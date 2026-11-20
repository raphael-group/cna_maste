"""Fit checks and fit counts that leave the outputs unchanged: a check logs a warning and counts, a count adds its
value, per run, under its name, the same name `run_capture_sim` records in the capture's `/metrics`."""

from collections import Counter

from cnamaste.config import start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

FIT_CHECKS = ("emptied_clone", "non_finite_llf", "labelling_below_min_spots")
"""`emptied_clone`: a labelling left a clone without spots (`hmrf.run_hmrf`); `non_finite_llf`: a fit's llf is not
finite (`hmm.baum_welch`); `labelling_below_min_spots`: a labelling holds a clone under `hmrf.min_spots_per_clone`."""

FIT_COUNTS = tuple(
    f"{name}_{stage}" for stage in ("baf", "rdr") for name in ("flip_backs", "oscillating_spots", "stable_spots")
) + ("mstep_calls", "mstep_unconverged", "mstep_precision_loss")
"""The HMRF outer loop's labelling turnover per stage (`hmrf.run_hmrf`): `flip_backs`, the spots that changed back
to a label held at an earlier iteration, summed over iterations; `oscillating_spots`, the distinct such spots;
`stable_spots`, the spots unchanged over the stage's last `hmrf.STABLE_ITERATIONS` iterations; and the M steps
(`mstep.get_mstep_minimum`): `mstep_calls`, `mstep_unconverged`, those not converged, and `mstep_precision_loss`,
those the line search ended."""

FIT_TIMES = ("mstep_seconds",)
"""`mstep_seconds`: the M steps' summed wall time, a count no two runs share."""

fit_check_counts: Counter[str] = Counter()
"""This run's count per check, `reset_fit_check_counts` at the run's start."""


def get_fit_checks() -> tuple[str, ...]:
    """Return the fit checks, `FIT_CHECKS`."""
    return FIT_CHECKS


def record_fit_check(name: str, message: str) -> None:
    """Log `name`'s warning and count it.

    Args:
        name: one of `FIT_CHECKS`.
        message: what the check found.

    Raises:
        ValueError: for a name not in `FIT_CHECKS`.
    """
    if name not in FIT_CHECKS:
        raise ValueError(f"fit check {name!r} is not one of {FIT_CHECKS}.")
    fit_check_counts[name] += 1
    logger.warning(f"fit check {name}: {message}")


def get_fit_times() -> tuple[str, ...]:
    """Return the fit counts that hold wall time, `FIT_TIMES`."""
    return FIT_TIMES


def record_fit_count(name: str, count: float) -> None:
    """Add `count` to `name`'s count, without a warning.

    Args:
        name: one of `FIT_COUNTS` or `FIT_TIMES`.
        count: the value.

    Raises:
        ValueError: for a name not in `FIT_COUNTS` or `FIT_TIMES`.
    """
    if name not in FIT_COUNTS + FIT_TIMES:
        raise ValueError(f"fit count {name!r} is not one of {FIT_COUNTS + FIT_TIMES}.")
    fit_check_counts[name] += count


def reset_fit_check_counts() -> None:
    """Zero every check's count."""
    fit_check_counts.clear()


def get_fit_check_counts() -> dict[str, float]:
    """Each check's and count's value this run, zero included.

    Returns:
        Name -> count, in `FIT_CHECKS`, `FIT_COUNTS` then `FIT_TIMES`' order.
    """
    return {name: fit_check_counts[name] for name in FIT_CHECKS + FIT_COUNTS + FIT_TIMES}
