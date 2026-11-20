"""The H&E network as a further unary in cnamaste's label step: its Potts energy plus -w log pi.

`install` rebinds `hmrf.get_hmrf_labelling`, the label step `run_hmrf` calls each iteration. Before
each call the network is fit by cross-entropy to the labels entering the step (warm, after the first; not at all where
those labels and the clone count are the last fit's), and for that call only `hmrf.get_labelling_unary` returns cnamaste's
field plus w log pi, so the alpha expansion and the step's log-likelihood see the energy

    E(l) = -sum_n [field_{n, l_n} + w log pi_{n, l_n}] - beta sum_{n~n'} 1[l_n = l_n']

Everything else is cnamaste's: the pseudobulk fit, the solver, the integer decoding, and the merge within the step,
which sees cnamaste's field alone: merging is a decision about genotype, and with the unary in it clones of one
genotype on slides of different stain were kept apart. What keeps the one network from echoing the labels back is its
capacity, `num_features`.

Each label step writes `<output_dir>/he_unary.json` (the per-step log) and `<output_dir>/he_prior.npz` (the network as
that step used it: log pi per spot in run order, argmax and entropy per output pixel), so a run that fails later keeps
both.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cnamaste import hmrf
from cnamaste.sandbox import he

STEPS_FIRST = 300
"""Adam steps on the first fit, from a fresh network."""

STEPS_NEXT = 60
"""Adam steps on each later fit, warm from the last."""


class HEUnary:
    """The H&E unary for one run: the slides, each run spot's pixel, one warm network, and where it writes."""

    def __init__(self, images, pixels, order, *, weight, num_features, seed=0, output_dir=None, spots=None, meta=None):
        """Args: `images`, `pixels`, `order` as `he.he_inputs` returns them; `weight` w; `num_features` the network's
        capacity; `seed` its initialization; `output_dir` where each step writes, None for nowhere; `spots` the run's
        `barcodes` and `samples`, written beside log pi; `meta` written at the head of the log."""
        self.images, self.pixels, self.order = images, pixels, order
        self.weight, self.num_features, self.seed = weight, num_features, seed
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.spots, self.meta = spots or {}, meta or {}
        self.network = None  # NB (n_clones, params, adam, labels, log pi in network order) of the last fit
        self.log = []

    def log_prior(self, labels_run, n_clones):
        """(n_spots, n_clones) log pi in the run's spot order, the network refit to `labels_run` where they changed."""
        labels = np.empty(len(self.order), dtype=np.int64)
        labels[self.order] = labels_run
        last = self.network
        if last is not None and last[0] == n_clones and np.array_equal(last[3], labels):
            steps, lp = 0, last[4]
        else:
            warm = last is not None and last[0] == n_clones
            steps = STEPS_NEXT if warm else STEPS_FIRST
            params, adam, _ = he.fit(
                self.images,
                self.pixels,
                labels,
                n_clones,
                steps=steps,
                seed=self.seed,
                params=last[1] if warm else None,
                adam=last[2] if warm else None,
                num_features=self.num_features,
            )
            lp = np.asarray(he.log_prior(params, self.images, self.pixels))
            self.network = (n_clones, params, adam, labels, lp)
        top = np.sort(lp, axis=1)
        self.log.append(
            {
                "clones": n_clones,
                "steps": steps,
                "agreement": float(np.mean(lp.argmax(1) == labels)),
                "median_margin": float(np.median(top[:, -1] - top[:, -2])) if n_clones > 1 else 0.0,
            }
        )
        self.write()
        return lp[self.order]

    def write(self):
        """The log and the network as the last step used it, to `output_dir`: rewritten each step."""
        if self.output_dir is None or self.network is None:
            return
        (self.output_dir / "he_unary.json").write_text(json.dumps({**self.meta, "steps": self.log}, indent=1))
        arrays = {**self.spots, "log_prior": self.network[4][self.order]}
        names = list(dict.fromkeys(np.asarray(self.spots.get("samples", []))))
        for sid, (argmax, entropy) in zip(names, he.pixel_posterior(self.network[1], self.images), strict=False):
            arrays[f"pixel_{sid}"], arrays[f"entropy_{sid}"] = argmax, entropy
        np.savez(self.output_dir / "he_prior.npz", **arrays)


def install(unary):
    """Rebind `hmrf.get_hmrf_labelling` to add `unary`; returns the function that restores it."""
    original_step, original_field, original_merge = hmrf.get_hmrf_labelling, hmrf.get_labelling_unary, hmrf.merge_clones

    def step(*args, **kwargs):
        prev = args[3]
        n_clones = len(args[1]["pred_cnv"])
        log_pi = unary.log_prior(np.asarray(prev, dtype=np.int64), n_clones)

        def field(*f_args, **f_kwargs):
            return original_field(*f_args, **f_kwargs) + unary.weight * log_pi

        def merge(single_llf, *m_args, **m_kwargs):
            # NB a merge is a decision about genotype, which the slide cannot make: it sees cnamaste's field alone.
            return original_merge(single_llf - unary.weight * log_pi, *m_args, **m_kwargs)

        hmrf.get_labelling_unary, hmrf.merge_clones = field, merge
        try:
            return original_step(*args, **kwargs)
        finally:
            hmrf.get_labelling_unary, hmrf.merge_clones = original_field, original_merge

    hmrf.get_hmrf_labelling = step

    def restore():
        hmrf.get_hmrf_labelling = original_step

    return restore
