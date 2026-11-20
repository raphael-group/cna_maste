from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from cnamaste.config import get_global_config, start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


class LockableMixin:
    """A dataclass that refuses attribute writes once `lock`ed, its nested `LockableMixin` fields with it."""

    _locked: bool = False

    def lock(self) -> None:
        object.__setattr__(self, "_locked", True)

        if is_dataclass(self):
            for f in fields(self):
                val = getattr(self, f.name)
                if isinstance(val, LockableMixin):
                    val.lock()

    def unlock(self) -> None:
        object.__setattr__(self, "_locked", False)

        if is_dataclass(self):
            for f in fields(self):
                val = getattr(self, f.name)
                if isinstance(val, LockableMixin):
                    val.unlock()

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False) and name != "_locked":
            raise RuntimeError(f"Instance is locked. Cannot modify attribute '{name}'.")
        super().__setattr__(name, value)


def find_plain_or_gz(path):
    """`path` where it exists, else `path.gz` where that exists, else None."""
    plain = Path(path)
    packed = plain.with_name(plain.name + ".gz")
    return plain if plain.exists() else packed if packed.exists() else None


def cast_clone_label(label, with_normal=False):
    num = label.lower().replace("clone", "").strip()
    num = int(num)

    if not (-1 <= num <= 3999):
        raise ValueError("Input must be an integer between -1 and 3999.")

    if num == -1:
        return "WARN"
    elif num == 0:
        if with_normal:
            return "Normal"
        else:
            return "Clone 0"
    else:
        lookup = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ]

        roman_numeral = ""

        for value, symbol in lookup:
            while num >= value:
                roman_numeral += symbol
                num -= value

        return f"Clone {roman_numeral}"


def pause(config=None):
    if config is None:
        config = get_global_config()

    if bool(config.run.pause):
        input("<Enter>")


# TODO mv to source, not in utils, if still used.
def get_intervals(pred_cnv):
    """Find contiguous intervals of equal state in `pred_cnv`, real copy states (Z) or integer (A, B) states.

    Args:
        pred_cnv: the state labels.

    Returns:
        The (start, end) index intervals and each interval's state label.
    """
    intervals, labs = [], []
    s = 0

    while s < len(pred_cnv):
        t = np.where(pred_cnv[s:] != pred_cnv[s])[0]
        if len(t) == 0:
            intervals.append((s, len(pred_cnv)))
            labs.append(pred_cnv[s])
            s = len(pred_cnv)
        else:
            # NB next label switch
            t = t[0]

            # NB add the interval (run start index to run end index)
            intervals.append((s, s + t))

            # NB add the corresponding state label for this new interval.
            labs.append(pred_cnv[s])

            # NB update the index.
            s = s + t

    return intervals, labs
