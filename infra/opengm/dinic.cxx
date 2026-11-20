// Dinic's maximum flow, the min s-t cut `hmrf.opengm_mincut_solver: dinic` names: included by cnamaste_opengm.cxx,
// built by infra/opengm/build_opengm.sh, and checked against scipy.sparse.csgraph.maximum_flow's own Dinic by
// tests/extended/test_fn_labelling.py through cnamaste_min_cut.
#ifndef CNAMASTE_DINIC_CXX
#define CNAMASTE_DINIC_CXX

#include <algorithm>
#include <cstddef>
#include <limits>
#include <queue>
#include <vector>

namespace cnamaste {

// Dinic's maximum flow on double capacities, for OpenGM's GraphCut. Node 0 is
// the source and node 1 the sink; calculateCut marks a node true where it is
// not reachable from the source in the residual graph, as OpenGM's
// MinSTCutBoost does.
class Dinic {
 public:
  typedef std::size_t node_type;
  typedef double ValueType;

  Dinic(std::size_t n_nodes, std::size_t n_edges) : head_(n_nodes, -1), level_(n_nodes), cursor_(n_nodes) {
    to_.reserve(2 * n_edges);       // two arcs per edge: forward, then its reverse
    capacity_.reserve(2 * n_edges);  // reserved once, so no reallocation while building
    next_.reserve(2 * n_edges);
  }

  // Always a pair, forward at an even index and reverse at the odd one after
  // it, which is what makes `arc ^ 1` the reverse arc in augment().
  void addEdge(node_type tail, node_type head, ValueType capacity) {
    push(tail, head, capacity);
    push(head, tail, 0.0);  // reverse starts empty; augment() fills it as flow passes
  }

  void calculateCut(std::vector<bool>& segmentation) {
    while (layer()) {                                       // a phase: build the level graph
      std::copy(head_.begin(), head_.end(), cursor_.begin());  // reset the current-arc pointers
      while (augment(0, std::numeric_limits<double>::infinity()) > 0.0) {
      }                                                     // push a blocking flow through it
    }
    // At most V - 1 phases, because each one strictly lengthens the shortest
    // source-sink path, and at most E augmentations per phase, because each
    // saturates its bottleneck exactly. Neither bound involves the capacities,
    // so this terminates on doubles as it does on integers.
    layer();  // once more, for reachability in the *final* residual graph: this is the cut
    segmentation.assign(head_.size(), true);
    for (std::size_t node = 0; node < head_.size(); ++node) {
      segmentation[node] = level_[node] < 0;  // true = unreachable = the sink side, as MinSTCutBoost has it
    }
    segmentation[0] = false;  // source
    segmentation[1] = false;  // sink, which the final BFS cannot reach
  }

 private:
  // Forward star: `head_[v]` is v's newest arc, `next_[a]` the one before it,
  // so a node's arcs are a singly-linked list threaded through the arc arrays.
  void push(node_type tail, node_type head, ValueType capacity) {
    to_.push_back(static_cast<long>(head));
    capacity_.push_back(capacity);
    next_.push_back(head_[tail]);                   // link to what was first
    head_[tail] = static_cast<long>(to_.size()) - 1;  // and become first
  }

  // BFS from the source over arcs with residual capacity, numbering each node
  // by its distance. -1 is unreached. Returns whether the sink is still
  // reachable, i.e. whether an augmenting path remains.
  bool layer() {
    std::fill(level_.begin(), level_.end(), -1);
    std::queue<long> queue;
    level_[0] = 0;  // the source
    queue.push(0);
    while (!queue.empty()) {
      long node = queue.front();
      queue.pop();
      for (long arc = head_[node]; arc >= 0; arc = next_[arc]) {  // walk node's arc list
        if (capacity_[arc] > 0.0 && level_[to_[arc]] < 0) {
          level_[to_[arc]] = level_[node] + 1;
          queue.push(to_[arc]);
        }
      }
    }
    return level_[1] >= 0;  // the sink
  }

  // One augmenting path, descending the level graph. `limit` carries the
  // bottleneck so far.
  double augment(long node, double limit) {
    if (node == 1) {
      return limit;  // reached the sink: the path can take this much
    }
    // A reference, so advancing `arc` advances the saved cursor: an arc this
    // phase has exhausted is never rescanned (Dinic's current-arc rule).
    for (long& arc = cursor_[node]; arc >= 0; arc = next_[arc]) {
      long next = to_[arc];
      if (capacity_[arc] > 0.0 && level_[next] == level_[node] + 1) {  // strictly downhill only
        // std::min *selects*, it does not compute, so at the bottleneck arc
        // `sent` is that arc's capacity bit for bit, and `-=` below is an
        // exact zero (IEEE: x - x == 0). That is what saturates an arc per
        // augmentation and bounds the phase, on doubles as on integers.
        double sent = augment(next, std::min(limit, capacity_[arc]));
        if (sent > 0.0) {
          capacity_[arc] -= sent;
          capacity_[arc ^ 1] += sent;  // the reverse arc, so the flow can be undone
          return sent;
        }
      }
    }
    return 0.0;  // no path from here
  }

  std::vector<long> head_,   // per node: its first arc, or -1
      level_,                 // per node: BFS distance from the source, -1 unreached
      cursor_,                // per node: the arc augment() is up to this phase
      to_,                    // per arc: its head
      next_;                  // per arc: the next arc out of the same tail
  std::vector<double> capacity_;  // per arc: residual capacity
};

}  // namespace cnamaste

#endif  // CNAMASTE_DINIC_CXX
