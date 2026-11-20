import copy
from dataclasses import dataclass, fields
from typing import Any, Optional

import numpy as np

from cnamaste.config import start_time
from cnamaste.emission import CnaEmissionConfounders
from cnamaste.hmm.baum_welch import HmmFit, get_paths
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.logger import get_logger
from cnamaste.utils import LockableMixin

logger = get_logger(__name__, start_time=start_time)


@dataclass
class HMMProfile(LockableMixin):
    log_gamma: np.ndarray
    pred_cnv: np.ndarray


@dataclass
class CloneAssignment(LockableMixin):
    assignment_before_reindex: Optional[np.ndarray] = None
    prev_assignment: Optional[np.ndarray] = None
    new_assignment: Optional[np.ndarray] = None
    total_llf: float = np.nan

    @property
    def unique_clone_labels(self):
        if self.new_assignment is not None:
            return np.unique(self.new_assignment)
        return None

    @property
    def num_clones(self):
        unique_labels = self.unique_clone_labels
        return len(unique_labels) if unique_labels is not None else None


@dataclass
class HmrfFit(LockableMixin):
    parameters: CnaHmmParameters
    confounders: CnaEmissionConfounders
    log_transmat: np.ndarray
    profile: HMMProfile
    llf: float
    n_states: int
    assignment: CloneAssignment

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        # NB valid clone assignment must be 0...N labels
        if self.assignment is not None and getattr(self.assignment, "new_assignment", None) is not None:
            unique_labels = np.unique(self.assignment.new_assignment)

            if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
                raise ValueError(
                    f"Invalid clone assignment: {unique_labels}. "
                    "Clone labels must be consecutive integers starting from 0."
                )

        # NB log the shape of log_mu, log_gamma and new_assignment
        logger.info(
            f"Validated HmrfFit with log_mu.shape={np.shape(self.parameters.log_mu)},"
            f"log_gamma.shape={self.profile.log_gamma.shape}, "
            f"pred_cnv.shape={self.profile.pred_cnv.shape}, "
            f"new_assignment.shape={getattr(self.assignment, 'new_assignment', None).shape if getattr(self.assignment, 'new_assignment', None) is not None else None}, "
            f"num_clones={getattr(self.assignment, 'num_clones')}."
        )

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)

        if hasattr(self.profile, key):
            return getattr(self.profile, key)

        if hasattr(self.assignment, key):
            return getattr(self.assignment, key)

        raise KeyError(f"'{key}' not found in HmrfFit, HMMProfile, or CloneAssignment.")

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        elif hasattr(self.profile, key):
            setattr(self.profile, key, value)
        elif hasattr(self.assignment, key):
            setattr(self.assignment, key, value)
        else:
            raise KeyError(f"Cannot set unknown key '{key}'")

        self.validate()

        logger.debug(f"Successfully set '{key}' in HmrfFit with new value: {value}.")

    def __str__(self) -> str:
        lines = ["HmrfFit:"]

        def format_val(val: Any, is_param: bool, indent: str) -> str:
            if isinstance(val, np.ndarray):
                if is_param:
                    arr_str = np.array2string(val, threshold=np.inf, separator=", ")
                    indented_arr = indent + arr_str.replace("\n", "\n" + indent)
                    return f"\n{indented_arr}"
                else:
                    arr_str = np.array2string(val, threshold=10, edgeitems=2, separator=", ")
                    indented_arr = indent + arr_str.replace("\n", "\n" + indent)
                    return f"<ndarray shape={val.shape} dtype={val.dtype}>\n{indented_arr}"

            elif isinstance(val, float):
                return f"{val:.6f}" if not np.isnan(val) else "nan"

            return str(val)

        nested_fields = {"parameters", "confounders", "profile", "assignment"}

        for f in fields(self):
            if f.name not in nested_fields:
                val = getattr(self, f.name)
                lines.append(f"  {f.name}: {format_val(val, False, '    ')}")

        for name in ["parameters", "confounders"]:
            lines.append(f"  {name}:")
            for key, val in getattr(self, name)._asdict().items():
                val_str = format_val(val, True, "      ")
                lines.append(f"    {key}:{val_str}" if val_str.startswith("\n") else f"    {key}: {val_str}")

        for nested_name in ["profile", "assignment"]:
            nested_obj = getattr(self, nested_name)

            if nested_obj is None:
                lines.append(f"  {nested_name}: None")
                continue

            lines.append(f"  {nested_name} ({nested_obj.__class__.__name__}):")

            is_param = False

            for f in fields(nested_obj):
                val = getattr(nested_obj, f.name)
                val_str = format_val(val, is_param, "      ")

                if val_str.startswith("\n"):
                    lines.append(f"    {f.name}:{val_str}")
                else:
                    lines.append(f"    {f.name}: {val_str}")

        return "\n".join(lines)

    def copy(self, deep: bool = False) -> "HmrfFit":
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def keys(self):
        all_keys = []
        nested_fields = {"parameters", "confounders", "profile", "assignment"}

        for f in fields(self):
            if f.name not in nested_fields:
                all_keys.append(f.name)

        for sub_obj in [self.profile, self.assignment]:
            if sub_obj is not None:
                all_keys.extend(f.name for f in fields(sub_obj))

        return all_keys

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def get_npz_fields(self) -> dict[str, Any]:
        """The fit as `rdrbaf_final_nstates<n>_smp.npz` holds it: the parameters, `(n_states,)`, as `new_*`, and
        `new_log_mu_shift` `(n_clones,)`, None where every `log Z_c` is 0.

        Returns:
            Key -> value, in the file's order.
        """
        log_mu_shift = np.asarray(self.confounders.log_mu_shift).reshape(-1)
        parameters = {
            "new_log_mu": self.parameters.log_mu,
            "new_alphas": self.parameters.alphas,
            "new_p_binom": self.parameters.p_binom,
            "new_taus": self.parameters.taus,
            "new_log_startprob": self.parameters.log_startprob,
            "new_log_transmat": self.log_transmat,
            "new_log_mu_shift": log_mu_shift if np.any(log_mu_shift) else None,
        }
        return (
            {"llf": self.llf, "n_states": self.n_states}
            | parameters
            | {f.name: getattr(sub, f.name) for sub in (self.profile, self.assignment) for f in fields(sub)}
        )


def get_hmrf_fit(fit: HmmFit, log_transmat: np.ndarray) -> HmrfFit:
    """`baum_welch`'s fit as a `HmrfFit`, no assignment yet; `pred_cnv` its `get_paths`, `(n_clones, n_bins)`.

    Args:
        fit: the `HmmFit`.
        log_transmat: `(K, K)` the copy state transitions.

    Returns:
        The `HmrfFit`.
    """
    return HmrfFit(
        parameters=fit.parameters,
        confounders=fit.confounders,
        log_transmat=log_transmat,
        profile=HMMProfile(log_gamma=fit.log_gamma, pred_cnv=get_paths(fit.log_gamma)),
        assignment=CloneAssignment(),
        llf=fit.llf,
        n_states=log_transmat.shape[0],
    )
