"""The H&E network: a small CNN over a slide, fit to a label set by cross-entropy, read as a clone prior per spot.

pi_nm = softmax_m f_psi(I)_{x_n m}; `fit` maximizes sum_nm 1[l_n = m] log pi_nm - nu |psi|^2. The network sees only the
image: what it can express is what varies with the slide, which is what makes it a prior rather than a classifier of
the counts. `unary` adds it to the label step; `scripts.run_cnamaste_he` runs cnamaste with it.

A slide is read as Space Ranger lays it out (`slide`): `spatial/tissue_<resolution>_image.png` (`he.resolution`), its
`tissue_<resolution>_scalef` and `microns_per_pixel`, resampled by an integer factor to the input pixel
`input_microns` states, in optical density, and ZCA-whitened over the tissue its spots cover. On Visium HD
(`bin_size_um` stated) that pixel is half a bin, so the network's output, at `STRIDE`, is one pixel per bin: a
labelling of the bin array. A spot at `coords` (x, y) -- Visium HD's full-resolution pixels, as
`io.get_spatial_positions` reads `tissue_positions.parquet` -- sits at image row y s, column x s, s the image's pixels
per full-resolution pixel.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree

STRIDE = 2
"""The first layer's stride: the network's output is at half the input's resolution."""

DILATIONS = (1, 2, 4, 1)
"""Each layer's dilation: a receptive field of 28 input pixels, ~450 um at `INPUT_MICRONS`, ~220 um on 16 um bins."""

WEIGHT_DECAY = 1e-3
"""nu, the network's weight decay."""

LEARNING_RATE = 0.01
"""Adam's step on the network."""

INPUT_MICRONS = 16.0
"""The network's input pixel, in um, on a Visium (spot) slide."""

RESOLUTIONS = ("hires", "lowres")
"""Space Ranger's two downsampled images; `he.resolution` names one."""


# --- the network -------------------------------------------------------------------------------------------------------


def init_cnn(key: jax.Array, n_clones: int, num_features: int) -> list[dict[str, jax.Array]]:
    """A fresh network: a stride-2 4x4 layer, two 3x3 layers and a 1x1 head, `num_features` channels wide.

    Args:
        key: jax's.
        n_clones: the head's outputs.
        num_features: the hidden channels, the network's capacity.

    Returns:
        Each layer's weights `w` (HWIO, float32: the network is the run's cost, and needs no float64) and bias `b`.
    """
    shapes = [
        (4, 4, 3, num_features),
        (3, 3, num_features, num_features),
        (3, 3, num_features, num_features),
        (1, 1, num_features, n_clones),
    ]
    params = []
    for k, s in zip(jax.random.split(key, len(shapes)), shapes, strict=True):
        w = jax.random.normal(k, s, jnp.float32) * jnp.sqrt(2.0 / (s[0] * s[1] * s[2]))
        params.append({"w": w, "b": jnp.zeros(s[-1], jnp.float32)})
    params[-1]["w"] = params[-1]["w"] * 0.1
    return params


def cnn(params: list[dict[str, jax.Array]], image: jax.Array) -> jax.Array:
    """(H, W, 3) -> (H / STRIDE, W / STRIDE, n_clones) logits, each layer at its `DILATIONS` entry."""
    x = image[None].astype(params[0]["w"].dtype)
    for i, (layer, d) in enumerate(zip(params, DILATIONS, strict=True)):
        stride = (STRIDE, STRIDE) if i == 0 else (1, 1)
        x = (
            jax.lax.conv_general_dilated(
                x, layer["w"], stride, "SAME", rhs_dilation=(d, d), dimension_numbers=("NHWC", "HWIO", "NHWC")
            )
            + layer["b"]
        )
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x[0]


def log_prior(params: list[dict[str, jax.Array]], images: list[jax.Array], pixels: list[jax.Array]) -> jax.Array:
    """(n_spots, n_clones) log pi: the logits at each spot's pixel, slice by slice, log-softmaxed in float64."""
    logits = jnp.concatenate(
        [cnn(params, img)[px[:, 0] // STRIDE, px[:, 1] // STRIDE] for img, px in zip(images, pixels, strict=True)]
    )
    return jax.nn.log_softmax(logits.astype(jnp.float64), axis=1)


def _objective(params, images, pixels, target):
    return jnp.sum(target * log_prior(params, images, pixels)) - WEIGHT_DECAY * sum(
        jnp.sum(layer["w"] ** 2) for layer in params
    )


_value_and_grad = jax.jit(jax.value_and_grad(_objective))


def fit(images, pixels, labels, n_clones, *, steps, seed=0, params=None, adam=None, num_features=4):
    """Fit the network to `labels` by cross-entropy: `steps` Adam ascent steps from `params`.

    Args: `images`, `pixels` per slice (`he_inputs`); `labels` (n_spots,) in the network's order; `n_clones`; `steps`;
    `seed` a fresh network's; `params`, `adam` the network and Adam's moments to continue from, None for fresh ones;
    `num_features` a fresh network's capacity.

    Returns:
        The network, Adam's moments and the final objective.
    """
    if params is None:
        params = init_cnn(jax.random.PRNGKey(seed), n_clones, num_features)
    if adam is None:
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        adam = {"m": zeros, "v": zeros, "i": 0}
    target = jax.nn.one_hot(jnp.asarray(labels), n_clones)
    value = jnp.asarray(np.nan)
    b1, b2 = 0.9, 0.999
    for _ in range(steps):
        value, grad = _value_and_grad(params, images, pixels, target)
        i = adam["i"] + 1
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, adam["m"], grad)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, adam["v"], grad)
        c1, c2 = 1 - b1**i, 1 - b2**i
        params = jax.tree_util.tree_map(
            lambda p, m, v, c1=c1, c2=c2: p + LEARNING_RATE * (m / c1) / (jnp.sqrt(v / c2) + 1e-8), params, m, v
        )
        adam = {"m": m, "v": v, "i": i}
    return params, adam, float(value)


def pixel_posterior(params, images):
    """Per slice, the prior's most probable clone and its entropy (nats) at every pixel of the network's output."""
    out = []
    for img in images:
        lp = jax.nn.log_softmax(cnn(params, img).astype(jnp.float64), axis=-1)
        out.append((np.asarray(jnp.argmax(lp, axis=-1)), np.asarray(-jnp.sum(jnp.exp(lp) * lp, axis=-1))))
    return out


# --- the slide ---------------------------------------------------------------------------------------------------------


def optical_density(image, k):
    """An H&E (H, W, 3 or 4) as the network reads it: averaged down `k`-fold, in optical density, standardized per
    channel (the stain's own linear space)."""
    img = image[..., :3]
    H, W = img.shape[0] // k, img.shape[1] // k
    img = img[: H * k, : W * k].reshape(H, k, W, k, 3).mean(axis=(1, 3))
    od = -np.log(np.clip(img, 1e-3, 1.0))
    return (od - od.mean(axis=(0, 1))) / (od.std(axis=(0, 1)) + 1e-6)


def whiten(od, tissue=None):
    """The optical density ZCA-whitened over the slide's `tissue` pixels (all where None): decorrelated channels at
    unit variance there. A global linear map, so local contrast is kept and the stain's balance between slides is
    removed; over the tissue only, so glass and debris do not set it.

    Args:
        od: (H, W, 3).
        tissue: (H, W) bool, or None.

    Returns:
        (H, W, 3) float32.
    """
    flat = np.asarray(od, dtype=np.float64).reshape(-1, 3)
    inside = flat if tissue is None else flat[tissue.ravel()]
    flat = flat - inside.mean(axis=0)
    lam, U = np.linalg.eigh(np.cov(inside.T))
    zca = U @ np.diag(1.0 / np.sqrt(lam + 1e-6)) @ U.T
    return jnp.asarray((flat @ zca).reshape(np.shape(od)).astype(np.float32))


def input_microns(scalefactors):
    """The network's input pixel in um: half a Visium HD bin (`bin_size_um`), so its output is one pixel per bin;
    else `INPUT_MICRONS`."""
    return scalefactors["bin_size_um"] / STRIDE if "bin_size_um" in scalefactors else INPUT_MICRONS


def rescale(spatial, resolution):
    """The slide image's um per pixel, and the network's input pixels per image pixel.

    Args:
        spatial: a slice's `spatial/` directory.
        resolution: one of `RESOLUTIONS`.

    Returns:
        (um per image pixel, 1 / k for the integer k that brings it closest to `input_microns`, or the integer u that
            does where the image is coarser).

    Raises:
        ValueError: for another resolution, or where `scalefactors_json.json` states no `microns_per_pixel`.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"he.resolution {resolution!r}: one of {RESOLUTIONS}")
    scalefactors = json.loads((Path(spatial) / "scalefactors_json.json").read_text())
    if "microns_per_pixel" not in scalefactors:
        raise ValueError(f"{spatial}/scalefactors_json.json states no microns_per_pixel")
    microns = scalefactors["microns_per_pixel"] / scalefactors[f"tissue_{resolution}_scalef"]
    target = input_microns(scalefactors)
    if microns > target:
        return float(microns), float(round(microns / target))
    return float(microns), 1.0 / max(1, round(target / microns))


def slide(spatial, resolution):
    """The network's input from `spatial` in optical density, and its pixels per full-resolution pixel. An image
    coarser than the input pixel is upsampled bilinearly: one output per bin, with no detail the image does not carry."""
    scalefactors = json.loads((Path(spatial) / "scalefactors_json.json").read_text())
    _, factor = rescale(spatial, resolution)
    image = mpimg.imread(Path(spatial) / f"tissue_{resolution}_image.png")[..., :3]
    if factor > 1:
        image = ndi.zoom(image, (factor, factor, 1), order=1)
    return optical_density(image, max(1, round(1 / factor))), scalefactors[f"tissue_{resolution}_scalef"] * factor


def tissue_mask(shape, px):
    """The input pixels within half a spot spacing (the median nearest-neighbour distance) of a spot at `px`."""
    mask = np.zeros(shape[:2], bool)
    mask[px[:, 0], px[:, 1]] = True
    spacing = float(np.median(cKDTree(px).query(px, k=2)[0][:, 1])) if len(px) > 1 else 1.0
    return ndi.binary_dilation(mask, iterations=max(1, int(np.ceil(spacing / 2))))


def he_inputs(coords, samples, spaceranger_dirs, resolution):
    """The slides, whitened over their spots' tissue, and each spot's input pixel, sample by sample.

    Args:
        coords: (n_spots, 2) `(x, y)` in full-resolution pixels.
        samples: (n_spots,) sample ids.
        spaceranger_dirs: sample id -> Space Ranger directory.
        resolution: `he.resolution`.

    Returns:
        (images, pixels, order): per sample its input image and its spots' (row, column) input pixels, spots grouped by
            sample and in run order within one; `order[i]` run spot i's index in that order.
    """
    coords, samples = np.asarray(coords, dtype=float), np.asarray(samples).astype(str)
    images, pixels, by_sample = [], [], []
    for sid in dict.fromkeys(samples):
        od, scale = slide(Path(spaceranger_dirs[sid]) / "spatial", resolution)
        idx = np.flatnonzero(samples == sid)
        xy = coords[idx] * scale
        H, W = od.shape[:2]
        px = np.stack([np.clip(xy[:, 1].astype(int), 0, H - 1), np.clip(xy[:, 0].astype(int), 0, W - 1)], axis=1)
        images.append(whiten(od, tissue_mask(od.shape, px)))
        pixels.append(jnp.asarray(px))
        by_sample.append(idx)
    network = np.concatenate(by_sample)
    order = np.empty(len(network), dtype=np.int64)
    order[network] = np.arange(len(network))
    return images, pixels, order
