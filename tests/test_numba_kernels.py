"""Every `njit` kernel outside `sandbox` names the test that referees it.

Referee: the decorated functions `ast` finds under `python/cnamaste`, against `REFEREES`; each named test function is
defined in its file. Fails where a kernel is added, renamed or removed without its referee, or a referee is renamed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "python" / "cnamaste"

PENDING = "test_hmm_inference (PR #19)"
"""The referee of the recursions and lattices PR- #19 tests, until it merges."""

REFEREES = {
    "hmm/forward_backward.py::get_bin_emission": "test_hmm_baum_welch.py::test_posteriors_and_llf_are_the_enumerated_paths",
    "hmm/forward_backward.py::get_phased_step": "test_hmm_baum_welch.py::test_posteriors_and_llf_are_the_enumerated_paths",
    "hmm/forward_backward.py::get_log_alpha": "test_hmm_baum_welch.py::test_posteriors_and_llf_are_the_enumerated_paths",
    "hmm/forward_backward.py::get_log_beta": "test_hmm_baum_welch.py::test_posteriors_and_llf_are_the_enumerated_paths",
    "hmm/forward_backward.py::get_viterbi": "test_hmm_baum_welch.py::test_viterbi_is_the_argmax_over_every_path",
    "hmm/forward_backward.py::get_uniform_switch_step": "test_hmm_baum_welch.py::test_viterbi_is_the_argmax_over_every_path",
    "emission.py::get_log_state_sums_by_clone": "test_emission.py::test_each_clone_s_log_mu_shift_is_its_bins_logsumexp",
    "ab_decode/decode.py::get_dispersion_loglike_gradient": "test_ab_decode.py::test_the_dispersion_objective_gradient_is_central_differences",
    "hmm_emission.py::get_bb_shape_params": "test_hmm_emission.py::test_the_shape_params_are_p_tau_and_one_minus_p_tau_at_each_row_s_state",
    "hmm_emission_tabled.py::count_table_size": "test_hmm_emission_tabled.py::test_the_count_table_size_is_one_plus_the_largest_count_or_zero",
    "hmm_emission_tabled.py::rdr_logpmf_tabled": "test_hmm_emission_tabled.py::test_the_tabled_element_kernels_are_hmm_nophasing_s_bitwise",
    "hmm_emission_tabled.py::baf_switch_logpmf_tabled": "test_hmm_emission_tabled.py::test_the_tabled_element_kernels_are_hmm_nophasing_s_bitwise",
    "hmm_emission_tabled.py::baf_logpmf_tabled": "test_hmm_emission_tabled.py::test_the_tabled_element_kernels_are_hmm_nophasing_s_bitwise",
    "hmm_emission_tabled.py::rdr_logpmf_at_count": "test_hmm_emission_tabled.py::test_the_tabled_element_kernels_are_hmm_nophasing_s_bitwise",
    "hmm_emission_tabled.py::rdr_logpmf_at_stored_baseline": "test_hmm_emission_tabled.py::test_the_tabled_element_kernels_are_hmm_nophasing_s_bitwise",
    "hmm_emission_tabled.py::baf_logpmf_at_stored_total": "test_hmm_emission_tabled.py::test_the_tabled_element_kernels_are_hmm_nophasing_s_bitwise",
    "hmm_emission_tabled.py::baf_logpmf_at_dense_total": "test_hmm_emission_tabled.py::test_the_tabled_element_kernels_are_hmm_nophasing_s_bitwise",
    "hmm_emission_tabled.py::rdr_field_dense": "test_hmm_emission_tabled.py::test_the_unary_is_scipy_s_log_pmf_summed_over_each_clone_s_path",
    "hmm_emission_tabled.py::rdr_field_csr": "test_hmm_emission_tabled.py::test_the_unary_is_scipy_s_log_pmf_summed_over_each_clone_s_path",
    "hmm_emission_tabled.py::rdr_field_sized_csr": "test_hmm_emission_tabled.py::test_the_coverage_summed_field_is_the_csr_field_to_rounding",
    "hmm_emission_tabled.py::rdr_field_stored_baseline": "test_hmm_emission_tabled.py::test_the_unary_is_scipy_s_log_pmf_summed_over_each_clone_s_path",
    "hmm_emission_tabled.py::baf_field_stored_total": "test_hmm_emission_tabled.py::test_the_unary_is_scipy_s_log_pmf_summed_over_each_clone_s_path",
    "hmm_emission_tabled.py::baf_field_dense_total": "test_hmm_emission_tabled.py::test_the_unary_is_scipy_s_log_pmf_summed_over_each_clone_s_path",
    "hmm_emission_tabled.py::get_state_parameters": "test_hmm_emission_tabled.py::test_the_state_parameters_are_the_closed_forms",
    "hmm_emission_tabled.py::get_count_tables": "test_hmm_emission_tabled.py::test_the_count_tables_are_lgamma",
    "hmm_emission_bb.py::bb_logpmf_element": "test_hmm_nophasing.py::test_every_bb_evaluator_is_scipy_s",
    "hmm_emission_nb.py::nb_logpmf_vector": "test_hmm_nophasing.py::test_every_nb_evaluator_is_scipy_s",
    "hmm_emission_nb.py::nb_logpmf_element": "test_hmm_nophasing.py::test_every_nb_evaluator_is_scipy_s",
    "hmm_emission_bb.py::bb_logpmf_vector": "test_hmm_nophasing.py::test_every_bb_evaluator_is_scipy_s",
    "hmm_emission_bb.py::bb_logpmf_vector_by_element": "test_hmm_nophasing.py::test_every_bb_evaluator_is_scipy_s",
    "hmm_emission_bb.py::bb_switch_logpmf_element": "test_hmm_nophasing.py::test_the_switch_is_the_swapped_count_log_pmf_less_the_original",
    "hmm_emission_bb.py::bb_switch_logpmf_vector": "test_hmm_nophasing.py::test_the_switch_is_the_swapped_count_log_pmf_less_the_original",
    "hmm_emission_nb.py::nb_logpmf_dense": "test_hmm_nophasing.py::test_every_nb_evaluator_is_scipy_s",
    "hmm_emission_bb.py::bb_logpmf_dense": "test_hmm_nophasing.py::test_every_bb_evaluator_is_scipy_s",
    "hmm_utils.py::digamma_difference_element": "test_ab_decode.py::test_the_dispersion_objective_gradient_is_central_differences",
    "hmm_utils.py::log_rising": "test_hmm_utils.py::test_log_rising_is_the_exact_sum_of_logs",
    "hmm_utils.py::log_factorial": "test_hmm_utils.py::test_log_factorial_is_scipy_s_gammaln",
    "hmm_utils.py::logsumexp": "test_hmm_utils.py::test_logsumexp_is_scipy_s",
    "normal_spot.py::_squared_deviation": "test_sparse.py::test_log1p_std_by_spot_is_numpy_from_dense_or_csr",
    "normal_spot.py::_squares_csr": "test_sparse.py::test_log1p_std_by_spot_is_numpy_from_dense_or_csr",
    "normal_spot.py::_squares_dense": "test_sparse.py::test_log1p_std_by_spot_is_numpy_from_dense_or_csr",
    "scripts/run_cnamaste.py::set_numba_seed": "test_run_cnamaste_overrides.py::test_set_numba_seed_seeds_numba_s_generator_as_numpy_s",
    "sparse.py::sum_over_stored_entries": "test_sparse.py::test_a_stored_entry_sum_is_the_full_row_sum_for_a_term_zero_at_absent_entries",
    "sparse.py::sum_over_stored_entries_and_columns": "test_sparse.py::test_the_stored_entries_and_columns_sum_is_the_dense_row_sum",
    "sparse.py::sum_over_row_dense": "test_sparse.py::test_a_dense_row_sum_is_the_ordered_loop",
    "sparse.py::sum_over_row_csr": "test_sparse.py::test_a_csr_row_sum_is_the_dense_row_sum",
    "spatio_genomic_counts.py::get_sized_label_sums": "test_spatio_genomic_counts.py::test_the_sized_baseline_is_its_dense_product_summed_and_counted",
    "sparse.py::aligned_csr_values": "test_sparse.py::test_aligned_csr_values_are_the_dense_values_at_the_driver_entries",
    "sim/laws.py::get_nb_quantile": "sim/test_sim_draw.py::test_a_zero_truncated_draw_is_the_nb_quantile",
    "sim/laws.py::draw_zero_truncated": "sim/test_sim_draw.py::test_a_zero_truncated_draw_is_the_nb_quantile",
    "sim/laws.py::get_table_sizes": "sim/test_sim_draw.py::test_dev_tree_easy_r0_is_its_pin",
}


def get_kernels() -> set[str]:
    """Return every `module::function` under `PACKAGE`, `sandbox` excluded, decorated `njit` or `jit`."""
    kernels = set()
    for path in PACKAGE.rglob("*.py"):
        if "sandbox" in path.relative_to(PACKAGE).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
                if name in ("njit", "jit"):
                    kernels.add(f"{path.relative_to(PACKAGE).as_posix()}::{node.name}")
    return kernels


def test_every_kernel_names_a_test_that_exists() -> None:
    """The kernels are `REFEREES`' keys; every referee but `PENDING` is a test function defined in its file."""
    assert get_kernels() == set(REFEREES)
    tests = Path(__file__).resolve().parent
    missing = []
    for referee in set(REFEREES.values()) - {PENDING}:
        file, name = referee.split("::")
        path = tests / file
        if not path.exists() or not re.search(rf"^def {name}\(", path.read_text(), re.MULTILINE):
            missing.append(referee)
    assert not missing, sorted(missing)
