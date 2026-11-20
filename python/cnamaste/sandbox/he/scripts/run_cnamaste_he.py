"""`run_cnamaste` with the H&E network as a further unary in its label step (`cnamaste.sandbox.he.unary`).

    run_cnamaste_he <config> [-o key=value ...] [--he_weight W] [--he_prior_num_features F] [--he_seed S]

The configuration names the slide image, `he.resolution` (hires, the default, or lowres): each sample's
`<spaceranger_dir>/spatial/tissue_<resolution>_image.png`, read as `he.slide` reads it. `io.load_input_data`, as
`run_cnamaste` calls it, is wrapped to read the run's own spots in the run's order -- their coordinates, barcodes and
samples -- and to install the unary before the inference starts. Needs the `jax` extra.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cnamaste import inference
from cnamaste.io import get_sample_sheet
from cnamaste.sandbox import he
from cnamaste.sandbox.he.unary import HEUnary, install
from cnamaste.scripts import run_cnamaste as rc


def parse_args(argv=None):
    """`run_cnamaste`'s arguments and the unary's."""
    ap = argparse.ArgumentParser(description="cna-maste pipeline, with the H&E CNN as a further unary")
    ap.add_argument("config_path", type=str)
    ap.add_argument("--over_rides", "-o", action="append", default=[])
    ap.add_argument("--he_weight", type=float, default=1.0)
    # NB the network's hidden channels per layer, its capacity: 4 is 512 parameters for 4 clones.
    ap.add_argument("--he_prior_num_features", type=int, default=4)
    ap.add_argument("--he_seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    """Install the unary as the run's data are read, then run cnamaste."""
    a = parse_args(argv)
    loaded = inference.load_input_data

    def load_input_data(config, *args, **kwargs):
        out = loaded(config, *args, **kwargs)
        coords, barcodes, adata = out[0], out[1], out[2]
        sheet = get_sample_sheet(config.paths.sample_sheet)
        dirs = {str(s): Path(d) for s, d in zip(sheet["sample_id"], sheet["spaceranger_dir"], strict=True)}
        samples = np.asarray(adata.obs["sample"]).astype(str)
        resolution = config.he.resolution
        images, pixels, order = he.he_inputs(coords, samples, dirs, resolution)
        microns, factor = he.rescale(dirs[samples[0]] / "spatial", resolution)
        meta = {
            "weight": a.he_weight,
            "num_features": a.he_prior_num_features,
            "seed": a.he_seed,
            "slide": {"resolution": resolution, "microns_per_image_pixel": microns, "input_per_image_pixel": factor},
        }
        unary = HEUnary(
            images,
            pixels,
            order,
            weight=a.he_weight,
            num_features=a.he_prior_num_features,
            seed=a.he_seed,
            output_dir=config.paths.output_dir,
            spots={"barcodes": np.asarray(barcodes).astype(str), "samples": samples},
            meta=meta,
        )
        install(unary)
        return out

    inference.load_input_data = load_input_data
    rc.run_cnamaste(a.config_path, over_rides=a.over_rides)


if __name__ == "__main__":
    main()
