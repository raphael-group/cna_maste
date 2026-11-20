// cnamaste's C entry point to OpenGM's inference, built by infra/opengm/build_opengm.sh.
//
// "The Potts form" below is cnamaste.potts's PottsProblem: E = offset + sum unary - sum J [x_a == x_b], offset outside.
//
// One call poses a pairwise Potts model in the Potts form and runs one OpenGM
// algorithm on it. The energy is sum_i unary[i, x_i] - sum_e J_e [x_a == x_b]:
// the unary is -h (cnamaste.potts's field, negated), each edge a
// PottsFunction with -J on the diagonal and 0 off it, so OpenGM's bound is a
// bound on that energy, with no constant to restore. Alpha expansion and the
// swap receive the metric form instead, 0 on the diagonal and J off it: their
// auxiliary-node construction (Boykov, Veksler and Zabih 2001) assumes
// V(a, a) = 0 and is wrong otherwise, which the first form showed as
// labellings a second expansion still lowered. The form differs by sum_e J_e, a
// constant, so the labelling is the same problem's; `value` is evaluated in
// the Potts form whatever the algorithm saw.
//
// Every algorithm is OpenGM's header-only code and needs no downloaded
// external. Alpha expansion and the alpha-beta swap need a min s-t cut, chosen
// per call (`Mincut`): Boost Graph's Boykov-Kolmogorov max flow through
// OpenGM's MinSTCutBoost (Boost Software License, header-only), or `Dinic`,
// written against the interface OpenGM's GraphCut calls (addEdge,
// calculateCut), in dinic.cxx. Kolmogorov's own maxflow library, research-only, is not used.
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/astar.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/dynamicprogramming.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/auxiliary/minstcutboost.hxx>

#include "dinic.cxx"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <queue>
#include <vector>

namespace {

typedef opengm::SimpleDiscreteSpace<std::size_t, std::size_t> Space;  // n_nodes, n_states per node, the same for all
typedef opengm::GraphicalModel<
    double, opengm::Adder,  // unary and pairwise terms sum; Minimizer, below, picks the direction
    OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>, opengm::PottsFunction<double>),  // the only two allowed
    Space>
    Model;

// Records the bound after each iteration an algorithm reports.
template <class INF>
struct Trace {
  std::vector<double>* bounds;
  void begin(INF&) {}
  std::size_t operator()(INF& inference) {        // once per iteration
    bounds->push_back(inference.bound());
    return opengm::visitors::VisitorReturnFlag::ContinueInf;  // never stop early
  }
  void end(INF&) {}
  void addLog(const std::string&) {}  // the visitor concept asks for these; nothing logs
  void log(const std::string&, double) {}
};

// Must stay in step with cnamaste.external.ALGORITHMS, which maps the names.
enum Algorithm { ICM = 0, LBP = 1, ASTAR = 2, TRWS = 3, DD = 4, EXPANSION = 5, SWAP = 6 };

// Must stay in step with cnamaste.external.MINCUTS: the min s-t cut expansion and the swap make each move with.
enum Mincut { DINIC = 0, BOYKOV_KOLMOGOROV = 1 };

typedef opengm::MinSTCutBoost<std::size_t, double, opengm::KOLMOGOROV> BoykovKolmogorov;  // Boost Graph's

typedef std::chrono::steady_clock Clock;  // monotonic: unaffected by any wall-clock change

double since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <class INF>
void read(const INF& inference, std::vector<std::size_t>& labels, double* bound) {
  inference.arg(labels);         // the labelling it settled on
  *bound = inference.bound();    // and its lower bound, where the algorithm keeps one
}

// Alpha expansion or the alpha-beta swap, each move one min s-t cut by `MinStCut`: the labelling, the bound trace and
// the seconds of the inference call alone.
template <class MinStCut>
void move_making(int algorithm, const Model& model, std::size_t steps, bool started,
                 const std::vector<std::size_t>& initial, std::vector<double>& bounds,
                 std::vector<std::size_t>& labels, double* bound, double* seconds) {
  typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCut> Cut;
  typename Cut::Parameter cut;
  Clock::time_point clock;
  if (algorithm == EXPANSION) {
    typedef opengm::AlphaExpansion<Model, Cut> Inference;
    typename Inference::Parameter parameter(steps, cut);
    if (started) {  // otherwise OpenGM picks its own initial labelling
      parameter.labelInitialType_ = Inference::Parameter::EXPLICIT_LABEL;
      parameter.label_ = initial;
    }
    Inference inference(model, parameter);
    Trace<Inference> visitor = {&bounds};
    clock = Clock::now();
    inference.infer(visitor);
    *seconds = since(clock);
    read(inference, labels, bound);
  } else {
    typedef opengm::AlphaBetaSwap<Model, Cut> Inference;
    typename Inference::Parameter parameter;
    parameter.parameter_ = cut;  // the swap takes the cut through its own parameter
    parameter.maxNumberOfIterations_ = steps;
    Inference inference(model, parameter);
    if (started) {
      inference.setStartingPoint(initial.begin());  // and the start after construction, unlike expansion
    }
    Trace<Inference> visitor = {&bounds};
    clock = Clock::now();
    inference.infer(visitor);
    *seconds = since(clock);
    read(inference, labels, bound);
  }
}

}  // namespace

extern "C" {

// Run `algorithm` on the model, expansion and the swap cutting with `mincut`, and write the labelling, its energy in the Potts form,
// the algorithm's bound, its per-iteration bound trace (at most
// `max_iterations` entries), the entries written, and the seconds of the
// inference call alone. Returns 0, or 1 with `error` holding the message.
int cnamaste_opengm_solve(int algorithm, int mincut, std::int64_t n_nodes, std::int64_t n_states, const double* unary,
                     std::int64_t n_edges, const std::int64_t* first, const std::int64_t* second,
                     const double* coupling, const std::int64_t* start, std::int64_t max_iterations,
                     double tolerance, std::int64_t* labels_out, double* value, double* bound,
                     double* trace, std::int64_t* taken, double* seconds, char* error,
                     std::int64_t error_size) {
  try {
    Space space(static_cast<std::size_t>(n_nodes), static_cast<std::size_t>(n_states));
    Model model(space);
    const std::size_t shape[] = {static_cast<std::size_t>(n_states)};  // a unary table is 1-D, n_states long
    for (std::int64_t node = 0; node < n_nodes; ++node) {
      opengm::ExplicitFunction<double> table(shape, shape + 1);
      for (std::int64_t state = 0; state < n_states; ++state) {
        table(static_cast<std::size_t>(state)) = unary[node * n_states + state];  // row-major, node by node
      }
      const std::size_t variable[] = {static_cast<std::size_t>(node)};
      model.addFactor(model.addFunction(table), variable, variable + 1);  // a factor over this node alone
    }
    // Expansion and the swap need V(a, a) = 0 and V(a, b) >= 0; the other
    // algorithms take the Potts form directly. The two differ by the constant
    // sum_e J_e, so the minimiser is the same and only `value`, recomputed in
    // the Potts form below, is reported.
    const bool metric = algorithm == EXPANSION || algorithm == SWAP;
    for (std::int64_t edge = 0; edge < n_edges; ++edge) {
      // PottsFunction(n1, n2, valueEqual, valueNotEqual): -J / 0 in the
      // Potts form, 0 / +J in the metric one.
      opengm::PottsFunction<double> potts(n_states, n_states, metric ? 0.0 : -coupling[edge],
                                          metric ? coupling[edge] : 0.0);
      const std::size_t ends[] = {static_cast<std::size_t>(first[edge]),
                                  static_cast<std::size_t>(second[edge])};
      model.addFactor(model.addFunction(potts), ends, ends + 2);  // a factor over the edge's two nodes
    }
    std::vector<std::size_t> labels(static_cast<std::size_t>(n_nodes), 0);
    std::vector<std::size_t> initial;
    if (start != nullptr) {
      initial.assign(start, start + n_nodes);  // warm start, where the caller gave one
    }
    std::vector<double> bounds;  // the visitor fills this; algorithms run without one leave it empty
    const std::size_t steps = static_cast<std::size_t>(max_iterations);
    Clock::time_point clock;
    *bound = -std::numeric_limits<double>::infinity();  // the answer for an algorithm that keeps no bound
    switch (algorithm) {
      case ICM: {  // one node at a time to its conditional best; no bound, no trace
        typedef opengm::ICM<Model, opengm::Minimizer> Inference;
        typename Inference::Parameter parameter(initial);  // the warm start, or empty
        Inference inference(model, parameter);
        clock = Clock::now();
        inference.infer();
        *seconds = since(clock);
        read(inference, labels, bound);
        break;
      }
      case LBP: {  // loopy belief propagation: min-sum here, since the accumulator is Minimizer
        typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Minimizer> Rules;
        // MaxDistance: convergence is the largest message change, against `tolerance`.
        typedef opengm::MessagePassing<Model, opengm::Minimizer, Rules, opengm::MaxDistance> Inference;
        typename Inference::Parameter parameter(steps, tolerance);
        Inference inference(model, parameter);
        Trace<Inference> visitor = {&bounds};
        clock = Clock::now();
        inference.infer(visitor);
        *seconds = since(clock);
        read(inference, labels, bound);
        break;
      }
      case ASTAR: {  // exact branch and bound: the optimum, at a cost that grows with the model
        typedef opengm::AStar<Model, opengm::Minimizer> Inference;
        Inference inference(model);  // no parameter and no visitor: it reports no per-iteration bound
        clock = Clock::now();
        inference.infer();
        *seconds = since(clock);
        read(inference, labels, bound);
        break;
      }
      case TRWS: {  // sequential tree-reweighted message passing: a lower bound it raises
        typedef opengm::TRWSi<Model, opengm::Minimizer> Inference;
        typename Inference::Parameter parameter(steps);
        // TRW-S's stopping criterion as OpenGM states it: the bound
        // rose by at most `tolerance |bound|`, or the gap closed to that.
        parameter.precision() = tolerance;
        parameter.isAbsolutePrecision() = false;
        parameter.minRelativeDualImprovement() = tolerance;
        Inference inference(model, parameter);
        Trace<Inference> visitor = {&bounds};
        clock = Clock::now();
        inference.infer(visitor);
        *seconds = since(clock);
        read(inference, labels, bound);
        break;
      }
      case DD: {  // dual decomposition: split into trees, solve each exactly, agree by subgradient
        typedef opengm::DDDualVariableBlock<marray::Marray<double> > Block;  // the dual variables
        typedef opengm::DualDecompositionBase<Model, Block>::SubGmType SubModel;
        typedef opengm::DynamicProgramming<SubModel, opengm::Minimizer> SubInference;  // exact on a tree
        typedef opengm::DualDecompositionSubGradient<Model, SubInference, Block> Inference;
        typename Inference::Parameter parameter;
        parameter.maximalNumberOfIterations_ = steps;
        parameter.minimalRelAccuracy_ = tolerance;
        Inference inference(model, parameter);
        Trace<Inference> visitor = {&bounds};
        clock = Clock::now();
        inference.infer(visitor);
        *seconds = since(clock);
        read(inference, labels, bound);
        break;
      }
      case EXPANSION:
      case SWAP: {  // move making: each step is one min s-t cut over a two-label subproblem
        if (mincut == BOYKOV_KOLMOGOROV) {
          move_making<BoykovKolmogorov>(algorithm, model, steps, start != nullptr, initial, bounds, labels, bound,
                                        seconds);
        } else if (mincut == DINIC) {
          move_making<cnamaste::Dinic>(algorithm, model, steps, start != nullptr, initial, bounds, labels, bound, seconds);
        } else {
          throw std::runtime_error("unknown min s-t cut");
        }
        // The solver's bound is in the metric form, shifted by sum_e J_e from
        // what `value` reports, so it is dropped rather than reported wrongly.
        *bound = -std::numeric_limits<double>::infinity();
        break;
      }
      default:
        throw std::runtime_error("unknown algorithm");
    }
    // The labelling's energy in the Potts form, nodes then edges in order --
    // recomputed here, so the number returned does not depend on which
    // parameterisation the algorithm was handed.
    *value = 0.0;
    for (std::int64_t node = 0; node < n_nodes; ++node) {
      labels_out[node] = static_cast<std::int64_t>(labels[static_cast<std::size_t>(node)]);
      *value += unary[node * n_states + labels_out[node]];
    }
    for (std::int64_t edge = 0; edge < n_edges; ++edge) {
      if (labels_out[first[edge]] == labels_out[second[edge]]) {
        *value -= coupling[edge];
      }
    }
    if (algorithm == ASTAR) {
      *bound = *value;  // exact: the minimum bounds itself
    }
    const std::size_t written = std::min(bounds.size(), steps);  // clamped: `trace` holds `steps` doubles
    std::copy(bounds.begin(), bounds.begin() + written, trace);
    *taken = static_cast<std::int64_t>(written);
    return 0;
  } catch (const std::exception& failure) {
    // Nothing may escape into ctypes: the message crosses as bytes instead.
    std::strncpy(error, failure.what(), static_cast<std::size_t>(error_size - 1));
    error[error_size - 1] = '\0';  // strncpy does not terminate a truncated copy
    return 1;
  }
}

// One min s-t cut by `mincut` on a directed graph, node 0 the source and node 1 the sink, as OpenGM's GraphCut poses
// it: `sink_side` 1 where a node is cut off from the source (0 for both terminals, as calculateCut marks them), and
// `value` the capacity crossing from the source's side to the sink's. Returns 0, or 1 with `error` holding the message.
int cnamaste_min_cut(int mincut, std::int64_t n_nodes, std::int64_t n_edges, const std::int64_t* tail,
                     const std::int64_t* head, const double* capacity, std::int64_t* sink_side, double* value,
                     char* error, std::int64_t error_size) {
  try {
    std::vector<bool> segmentation;
    const std::size_t nodes = static_cast<std::size_t>(n_nodes), edges = static_cast<std::size_t>(n_edges);
    if (mincut == BOYKOV_KOLMOGOROV) {
      BoykovKolmogorov cut(nodes, edges);
      for (std::size_t edge = 0; edge < edges; ++edge) {
        cut.addEdge(static_cast<std::size_t>(tail[edge]), static_cast<std::size_t>(head[edge]), capacity[edge]);
      }
      cut.calculateCut(segmentation);
    } else if (mincut == DINIC) {
      cnamaste::Dinic cut(nodes, edges);
      for (std::size_t edge = 0; edge < edges; ++edge) {
        cut.addEdge(static_cast<std::size_t>(tail[edge]), static_cast<std::size_t>(head[edge]), capacity[edge]);
      }
      cut.calculateCut(segmentation);
    } else {
      throw std::runtime_error("unknown min s-t cut");
    }
    // NB the source's side is 0 and the source itself, the sink's the nodes marked and the sink itself.
    *value = 0.0;
    for (std::size_t node = 0; node < nodes; ++node) {
      sink_side[node] = node == 1 || (node != 0 && segmentation[node]) ? 1 : 0;
    }
    for (std::size_t edge = 0; edge < edges; ++edge) {
      if (sink_side[tail[edge]] == 0 && sink_side[head[edge]] == 1) {
        *value += capacity[edge];
      }
    }
    return 0;
  } catch (const std::exception& failure) {
    std::strncpy(error, failure.what(), static_cast<std::size_t>(error_size - 1));
    error[error_size - 1] = '\0';
    return 1;
  }
}

}  // extern "C"
