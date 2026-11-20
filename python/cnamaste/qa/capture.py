"""A capture: one `run_cnamaste` run's recorded calls in HDF5, written by `tests/scripts/run_capture_sim.py`.

Each call is `/<NN_group>/<function>[_k]` holding `in` (`args`, `kwargs`), `out` and `rng`, with the attributes
`function` (`module:qualname`), `seconds` and `config` (an index into `/config`'s `configs`). A value is a dataset (an
array or scalar) or a group whose `kind` names its encoding; `Writer.put` and `get_value` are inverses. Datasets are
gzip-9 with shuffle, integers in the smallest exact dtype. An array equal to an earlier one, a stage's input equal to an
earlier stage's output included, is an HDF5 hard link to it; no digest is stored, and a replay compares values.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import inspect
import io
import json
import random
from pathlib import Path
from typing import Any

import anndata
import h5py
import jax
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import _helperlib

from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.qa.sims import get_project, get_resources
from cnamaste.spatio_genomic_counts import SpatioGenomicCounts

TEMPLATE = get_project() / "zenodo_sim_config.yaml"


def get_template() -> Path:
    """Return the template, `TEMPLATE`."""
    return TEMPLATE


ENTRY_POINT_SEED = 0
"""numpy's legacy generator, seeded before the entry point, as the end-to-end tests seed it."""


def get_entry_point_seed() -> int:
    """Return the entry point seed, 0."""
    return ENTRY_POINT_SEED


OUTPUT_DIR = "output"
"""`paths.output_dir`, relative to the scratch directory."""


def get_output_dir() -> str:
    """Return the output dir, "output"."""
    return OUTPUT_DIR


LINKED = ("sim", "ref")
"""The checkout's directories the scratch directory links to, so recorded paths are relative."""

GZIP = {"compression": "gzip", "compression_opts": 9, "shuffle": True}


def get_gzip() -> dict[str, Any]:
    """Return the gzip, `GZIP`."""
    return GZIP


FLOAT_RTOL = 1e-12
"""A replayed float's tolerance relative to the stored one; integers and strings compare exactly."""

SCALARS = {"bool": bool, "int": int, "float": float, "str": str}

STRING = h5py.string_dtype()

ANNDATA_FIELDS = ("X", "obs", "var", "layers", "obsm", "obsp", "uns")
"""What an `AnnData` is recorded by; `MAPPINGS` as dicts."""

MAPPINGS = ("layers", "obsm", "obsp", "uns")

LINK_BYTES = 1 << 20
"""An array this large is also checked against the transposes of earlier ones."""

ARRAY_KINDS = ("ndarray", "copy", "transposed")

SPARSE_KINDS = ("sparse", "transposed_sparse")

ArrayKey = tuple[str, tuple[int, ...], bytes]
"""An array's identity for linking repeats: its stored dtype, shape and bytes."""


def get_overrides(fixture: str) -> list[str]:
    """`run_cnamaste`'s `-o` overrides of `TEMPLATE` for `fixture`, its paths relative to the scratch directory.

    Args:
        fixture: the fixture's directory, `<name>_<hash>`.

    Returns:
        The `key=value` overrides.
    """
    paths = [f"paths.sample_sheet=sim/{fixture}/sample_sheet.tsv", f"paths.output_dir={OUTPUT_DIR}"]
    return paths + [f"references.{key}=ref/{name}" for key, name in get_resources().items()]


def stage_links(work: Path) -> None:
    """Link `LINKED` from the checkout into the scratch directory, and make `OUTPUT_DIR` there.

    Args:
        work: the scratch directory.
    """
    for name in LINKED:
        (work / name).symlink_to(get_project() / name, target_is_directory=True)
    (work / OUTPUT_DIR).mkdir()


def get_path_of(value: Any) -> str:
    """`module:qualname` of a function or class.

    Args:
        value: the function or class.

    Returns:
        The path `get_object` reads.
    """
    return f"{value.__module__}:{value.__qualname__}"


def get_object(path: str) -> Any:
    """The function or class at `module:qualname`; a static or class method as its underlying function.

    Args:
        path: as `get_path_of` writes it.

    Returns:
        The object.
    """
    module, _, qualname = path.partition(":")
    found: Any = importlib.import_module(module)
    for part in qualname.split("."):
        found = inspect.getattr_static(found, part)
    return found.__func__ if isinstance(found, (staticmethod, classmethod)) else found


def get_compact(a: np.ndarray) -> np.ndarray:
    """`a` in the smallest integer dtype that holds it exactly where it is integral, else `a`.

    Args:
        a: a numeric array.

    Returns:
        The array to store; `get_array` restores the dtype.
    """
    if a.dtype.kind == "f":
        finite = a.size and np.isfinite(a).all() and (a == np.round(a)).all() and not np.signbit(a[a == 0]).any()
        if not finite or np.abs(a).max() >= 2**53:
            return a
    elif a.dtype.kind not in "iu" or a.size == 0:
        return a
    low, high = int(a.min()), int(a.max())
    for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64):
        if np.iinfo(dtype).min <= low and high <= np.iinfo(dtype).max:
            return a.astype(dtype)
    return a


def get_rng_state() -> dict[str, Any]:
    """numpy's legacy, numba's and `random`'s generator states, as `set_rng_state` restores them."""
    numpy_state = np.random.get_state()
    numba_state = _helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())  # type: ignore[attr-defined]
    return {"numpy": numpy_state, "numba": numba_state, "random": random.getstate()}


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore the generator states `get_rng_state` returned.

    Args:
        state: the states.
    """
    np.random.set_state(state["numpy"])
    _helperlib.rnd_set_state(_helperlib.rnd_get_np_state_ptr(), state["numba"])  # type: ignore[attr-defined]
    random.setstate(state["random"])


class Writer:
    """Encodes values under an open HDF5 file, an array equal to an earlier one hard-linked to it.

    Args:
        h5: the file.
        compress: gzip-9 with shuffle, else uncompressed (a replay's in-memory encoding).
    """

    def __init__(self, h5: h5py.File, compress: bool) -> None:
        self.h5 = h5
        self.filters = GZIP if compress else {}
        self.seen: dict[ArrayKey, tuple[str, str]] = {}
        self.seen_sparse: dict[ArrayKey, str] = {}
        """`get_sparse_key` of each canonical CSR matrix -> its path."""
        """`get_array_key` -> the first such array's path and recorded dtype."""

    def put(self, group: h5py.Group, name: str, value: Any) -> None:
        """Write `value` as `group[name]`.

        Args:
            group: the parent.
            name: the node's name.
            value: the value.

        Raises:
            TypeError: where no encoding holds `value`.
        """
        if value is None:
            self.put_kind(group, name, "none")
        elif isinstance(value, (bool, int, float, str)) and type(value).__name__ in SCALARS:
            data = np.asarray(value, dtype=STRING) if isinstance(value, str) else np.asarray(value)
            group.create_dataset(name, data=data).attrs["kind"] = type(value).__name__
        elif isinstance(value, np.generic):
            group.create_dataset(name, data=value).attrs["kind"] = "numpy"
        elif isinstance(value, np.ndarray):
            self.put_array(group, name, value)
        elif isinstance(value, jax.Array):
            # NB a JAX array as its numpy copy: a replay passes numpy, which `jax.jit` takes alike.
            self.put_array(group, name, np.asarray(value))
        elif isinstance(value, (Path, enum.Enum)):
            node = self.put_kind(group, name, "path" if isinstance(value, Path) else "enum")
            node.attrs["value"] = str(value) if isinstance(value, Path) else json.dumps(value.value)
            node.attrs["class"] = get_path_of(type(value))
        elif isinstance(value, type) or inspect.isfunction(value) or inspect.isbuiltin(value):
            self.put_kind(group, name, "callable").attrs["path"] = get_path_of(value)
        elif isinstance(value, YAMLConfig):
            self.put_kind(group, name, "config").attrs["value"] = json.dumps(value.to_dict())
        elif isinstance(value, np.random.Generator):
            seed = value.bit_generator.seed_seq
            node = self.put_kind(group, name, "generator")
            node.attrs["class"] = get_path_of(type(value.bit_generator))
            node.attrs["state"] = json.dumps(value.bit_generator.state)
            node.attrs["seed"] = json.dumps([seed.entropy, seed.spawn_key, seed.pool_size, seed.n_children_spawned])
        elif sp.issparse(value):
            self.put_sparse(group, name, value)
        elif isinstance(value, SpatioGenomicCounts):
            self.put_items(group, name, "spatio_genomic_counts", {"values": value.get_bin_major()})
        elif isinstance(value, pd.DataFrame):
            labels = {"index": value.index, "columns": value.columns}
            dtypes = set(value.dtypes)
            if len(dtypes) == 1 and value.shape[1] > 1 and isinstance(dtype := dtypes.pop(), pd.SparseDtype):
                # NB a wide sparse frame, as its CSR matrix.
                matrix = sp.csr_matrix(value.sparse.to_coo())
                self.put_items(
                    group, name, "sparse_frame", labels | {"values": matrix}, fill=json.dumps(dtype.fill_value)
                )
            else:
                columns = {str(k): value.iloc[:, k].array for k in range(value.shape[1])}
                node = self.put_items(group, name, "frame", labels)
                self.put_items(node, "values", "dict", columns)
        elif isinstance(value, pd.Series):
            self.put_items(group, name, "series", {"values": value.array, "index": value.index}, label=repr(value.name))
        elif isinstance(value, pd.Index):
            self.put_index(group, name, value)
        elif isinstance(value, pd.api.extensions.ExtensionArray):
            self.put_extension(group, name, value)
        elif isinstance(value, anndata.AnnData):
            # NB anndata lists `X` among the layers under the key None.
            items = {f: getattr(value, f) for f in ANNDATA_FIELDS}
            items |= {f: {k: v for k, v in items[f].items() if k is not None} for f in MAPPINGS}
            self.put_items(group, name, "anndata", items)
        elif isinstance(value, tuple) and hasattr(value, "_fields"):
            fields = {field: getattr(value, field) for field in value._fields}
            self.put_items(group, name, "namedtuple", fields, **{"class": get_path_of(type(value))})
        elif dataclasses.is_dataclass(value):
            fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
            self.put_items(group, name, "dataclass", fields, **{"class": get_path_of(type(value))})
        elif isinstance(value, (list, tuple)) and is_array_list(value):
            lengths = np.array([len(item) for item in value], dtype=np.int64)
            concatenated = {"values": np.concatenate(value), "lengths": lengths}
            self.put_items(group, name, "array_list", concatenated, sequence=type(value).__name__)
        elif isinstance(value, (list, tuple, set, frozenset)):
            ordered = sorted(value) if isinstance(value, (set, frozenset)) else value
            self.put_items(group, name, type(value).__name__, dict(enumerate(ordered)))
        elif isinstance(value, dict):
            self.put_items(group, name, "dict", value)
        elif type(value).__module__.startswith("cnamaste.") and hasattr(value, "__dict__"):
            self.put_items(group, name, "object", vars(value), **{"class": get_path_of(type(value))})
        else:
            raise TypeError(f"{group.name}/{name}: no encoding for {type(value).__qualname__}")

    def put_kind(self, group: h5py.Group, name: str, kind: str) -> h5py.Group:
        """An empty group of `kind`.

        Args:
            group: the parent.
            name: the node's name.
            kind: the encoding.

        Returns:
            The group.
        """
        node = group.create_group(name)
        node.attrs["kind"] = kind
        return node

    def put_items(self, group: h5py.Group, name: str, kind: str, items: dict[Any, Any], **attrs: Any) -> h5py.Group:
        """A group of `kind` holding `items` as children `0`, `1`, ..., their keys in its `keys` attribute.

        Args:
            group: the parent.
            name: the node's name.
            kind: the encoding.
            items: key -> value; keys are str or int.
            **attrs: further attributes.

        Returns:
            The group.
        """
        node = self.put_kind(group, name, kind)
        node.attrs.update(attrs)
        try:
            node.attrs["keys"] = json.dumps([[k, type(k).__name__] for k in map(get_key, items)])
        except TypeError as refused:
            raise TypeError(f"{node.name}: {refused}") from None
        for index, value in enumerate(items.values()):
            self.put(node, str(index), value)
        return node

    def put_array(self, group: h5py.Group, name: str, value: np.ndarray) -> None:
        """An array: strings as UTF-8, other objects item by item, numbers compacted (`get_compact`); a repeat (or
        transpose) of an earlier array refers to it.

        Args:
            group: the parent.
            name: the node's name.
            value: the array.
        """
        if value.dtype.kind == "O" and not all(isinstance(x, str) for x in value.flat):
            self.put_items(group, name, "object_array", dict(enumerate(value.flat)), shape=json.dumps(value.shape))
            return
        key, path = get_array_key(value), f"{group.name}/{name}".lstrip("/")
        if key in self.seen:
            self.put_reference(group, name, "copy", *self.seen[key], value.dtype.str)
            return
        if value.ndim == 2 and value.nbytes >= LINK_BYTES and (transposed := get_array_key(value.T)) in self.seen:
            self.put_reference(group, name, "transposed", *self.seen[transposed], value.dtype.str)
            return
        self.seen[key] = (path, value.dtype.str)
        data = get_utf8(value) if value.dtype.kind in "OU" else get_compact(value)
        # NB one chunk per dataset: 3,068 auto-sized chunks cost 1.0 MB of compression on easy.
        filters = self.filters | {"chunks": data.shape} if self.filters and data.size > 16 else {}
        stored = group.create_dataset(name, data=data, **filters)
        stored.attrs.update({"kind": "ndarray", "dtype": value.dtype.str})

    def put_sparse(self, group: h5py.Group, name: str, value: sp.sparray | sp.spmatrix) -> None:
        """A sparse matrix as CSR; a transpose of an earlier one refers to it, and sorted indices are stored as their
        gaps within each row (`get_row_gaps`), which compress to about 0.65 of the indices.

        Args:
            group: the parent.
            name: the node's name.
            value: the matrix.
        """
        matrix = value.tocsr()
        attrs = {"shape": json.dumps(matrix.shape), "format": value.format, "index_dtype": matrix.indices.dtype.str}
        canonical = matrix.has_canonical_format
        transposed = get_sparse_key(matrix.T.tocsr()) if canonical and matrix.nnz else None
        if transposed in self.seen_sparse:
            self.put_kind(group, name, "transposed_sparse").attrs.update(attrs | {"of": self.seen_sparse[transposed]})
            return
        if canonical and matrix.nnz:
            self.seen_sparse[get_sparse_key(matrix)] = f"{group.name}/{name}".lstrip("/")
        sorted_indices = bool(matrix.has_sorted_indices)
        indices = get_row_gaps(matrix.indices, matrix.indptr) if sorted_indices else matrix.indices
        items = {"data": matrix.data, "indices": indices, "indptr": matrix.indptr}
        self.put_items(group, name, "sparse", items, row_gaps=sorted_indices, **attrs)

    def put_reference(self, group: h5py.Group, name: str, kind: str, path: str, stored: str, dtype: str) -> None:
        """A repeat of the array at `path`: a hard link to its dataset, else a group of `kind` naming it and the dtype.

        Args:
            group: the parent.
            name: the node's name.
            kind: `copy` or `transposed`.
            path: the earlier array's path.
            stored: the earlier array's recorded dtype.
            dtype: this array's recorded dtype.
        """
        if kind == "copy" and stored == dtype and isinstance(self.h5[path], h5py.Dataset):
            group[name] = self.h5[path]
        else:
            self.put_kind(group, name, kind).attrs.update({"of": path, "dtype": dtype})

    def put_index(self, group: h5py.Group, name: str, value: pd.Index) -> None:
        """An index: a `RangeIndex` by its range, any other by its values.

        Args:
            group: the parent.
            name: the node's name.
            value: the index.
        """
        if isinstance(value, pd.RangeIndex):
            node = self.put_kind(group, name, "range_index")
            node.attrs["range"] = json.dumps([value.start, value.stop, value.step])
        else:
            node = self.put_items(group, name, "index", {"values": value.array})
        node.attrs["label"] = repr(value.name)

    def put_extension(self, group: h5py.Group, name: str, value: Any) -> None:
        """A pandas column's array: numpy-backed by its values, a categorical by codes and categories, a nullable one
        by values and mask.

        Args:
            group: the parent.
            name: the node's name.
            value: the extension array.
        """
        dtype = str(value.dtype)
        if isinstance(value, pd.Categorical):
            items = {"codes": value.codes, "categories": value.categories}
            self.put_items(group, name, "categorical", items, ordered=bool(value.ordered))
        elif isinstance(value, pd.arrays.SparseArray):
            items = {"values": value.sp_values, "index": value.sp_index.indices}
            self.put_items(group, name, "sparse_column", items, size=len(value), fill=json.dumps(value.fill_value))
        elif isinstance(value, pd.arrays.NumpyExtensionArray):
            column = value.to_numpy()
            if column.dtype.kind == "O":
                nulls = np.asarray(pd.isnull(column), dtype=bool)
                null = "nan" if any(isinstance(x, float) for x in column[nulls]) else "none"
                types = {type(x).__name__ for x in column[~nulls]}
                scalar = types.pop() if len(types) == 1 and types <= set(SCALARS) else "object"
                if scalar != "object":
                    filled = np.where(nulls, SCALARS[scalar](), column)
                    column = filled.astype(str if scalar == "str" else SCALARS[scalar])
                items = {"values": column, "nulls": nulls}
                self.put_items(group, name, "object_column", items, null=null, type=scalar)
            else:
                self.put_array(group, name, column)
        else:
            mask = np.asarray(value.isna())
            items = {"values": value.to_numpy(dtype=object, na_value=None)[~mask].astype(value.dtype.numpy_dtype)}
            self.put_items(group, name, "masked", items | {"mask": mask}, dtype=dtype)


def is_array_list(value: list[Any] | tuple[Any, ...]) -> bool:
    """A sequence of two or more 1-D numeric arrays of one dtype, stored concatenated (`array_list`).

    Args:
        value: the sequence.

    Returns:
        Whether it is one.
    """
    arrays = [item for item in value if isinstance(item, np.ndarray) and item.ndim == 1 and item.dtype.kind in "biuf"]
    return len(value) > 1 and len(arrays) == len(value) and len({a.dtype for a in arrays}) == 1


def get_sparse_key(matrix: sp.csr_matrix) -> ArrayKey:
    """A canonical CSR matrix's identity for linking transposes: its data dtype, shape and arrays' bytes.

    Args:
        matrix: the matrix.

    Returns:
        The key.
    """
    parts = (matrix.data, matrix.indices.astype(np.int64), matrix.indptr.astype(np.int64))
    return (matrix.data.dtype.str, matrix.shape, b"".join(np.ascontiguousarray(a).tobytes() for a in parts))


def get_row_gaps(indices: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    """Sorted CSR column indices as gaps within each row: a row's first index, then each index less the one before.

    Args:
        indices: `(nnz,)` the column indices, sorted within each row.
        indptr: `(n_rows + 1,)` the row pointers.

    Returns:
        `(nnz,)` the gaps, in `indices`' dtype; `get_row_indices` inverts it.
    """
    gaps = np.diff(indices.astype(np.int64), prepend=0)
    starts = indptr[:-1][np.diff(indptr) > 0]
    gaps[starts] = indices[starts]
    return gaps.astype(indices.dtype)


def get_row_indices(gaps: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    """The column indices `get_row_gaps` coded: each row's cumulative sum of its gaps.

    Args:
        gaps: `(nnz,)` the gaps.
        indptr: `(n_rows + 1,)` the row pointers.

    Returns:
        `(nnz,)` the indices, in `gaps`' dtype.
    """
    summed = np.concatenate([[0], np.cumsum(gaps, dtype=np.int64)])
    indices: np.ndarray = (summed[1:] - np.repeat(summed[indptr[:-1]], np.diff(indptr))).astype(gaps.dtype)
    return indices


def get_array_key(value: np.ndarray) -> ArrayKey:
    """An array's identity for linking repeats: as stored (`get_compact`), so arrays equal but for an integer dtype link.

    Args:
        value: a numeric or string array.

    Returns:
        The stored dtype, the shape and the bytes.
    """
    if value.dtype.kind in "OU":
        return ("str", value.shape, "\x00".join(map(str, value.flat)).encode())
    data = np.ascontiguousarray(get_compact(value))
    return (data.dtype.str, value.shape, data.tobytes())


def get_key(key: Any) -> Any:
    """A mapping key as JSON holds it: str or int.

    Args:
        key: the key.

    Returns:
        The key.

    Raises:
        TypeError: where the key is neither.
    """
    if isinstance(key, (str, int, np.integer)) and not isinstance(key, bool):
        return key if isinstance(key, str) else int(key)
    raise TypeError(f"key {key!r}: only str and int keys are encoded")


def get_utf8(value: np.ndarray) -> np.ndarray:
    """A string array as fixed-length UTF-8, which gzip compresses; variable-length strings, which HDF5 keeps
    uncompressed in its global heap (2.5 MB on easy), only where a string ends in NUL or the array is empty.

    Args:
        value: an array of `str`.

    Returns:
        The array to store.
    """
    encoded = [str(item).encode() for item in value.flat]
    if not encoded or any(item.endswith(b"\x00") for item in encoded):
        return value.astype(STRING)
    return np.array(encoded, dtype=bytes).reshape(value.shape)


def get_array(node: h5py.Dataset) -> np.ndarray:
    """An array dataset decoded to its recorded dtype.

    Args:
        node: the dataset.

    Returns:
        The array.
    """
    dtype = np.dtype(node.attrs["dtype"])
    if dtype.kind in "OU" and node.dtype.kind == "S":
        strings = np.array([item.decode() for item in node[()].flat], dtype=object)
        return strings.reshape(node.shape).astype(dtype)
    if dtype.kind in "OU":
        return np.asarray(node.asstr()[()], dtype=object).astype(dtype)
    return np.asarray(node[()]).astype(dtype)


def get_items(node: h5py.Group) -> dict[Any, Any]:
    """A group's items by their recorded keys, decoded.

    Args:
        node: a group `Writer.put_items` wrote.

    Returns:
        key -> value.
    """
    return {key: get_value(node[str(i)]) for i, (key, _) in enumerate(json.loads(node.attrs["keys"]))}


def get_value(node: h5py.Group | h5py.Dataset) -> Any:
    """The value `Writer.put` wrote as `node`.

    Args:
        node: the dataset or group.

    Returns:
        The value.
    """
    kind = node.attrs["kind"]
    attrs = node.attrs
    if kind in SCALARS:
        return SCALARS[kind](node.asstr()[()] if kind == "str" else node[()])
    if kind == "numpy":
        return node[()]
    if kind == "ndarray":
        return get_array(node)
    if kind == "none":
        return None
    if kind == "path":
        return Path(attrs["value"])
    if kind == "enum":
        return get_object(attrs["class"])(json.loads(attrs["value"]))
    if kind == "callable":
        return get_object(attrs["path"])
    if kind == "config":
        return YAMLConfig(json.loads(attrs["value"]))
    if kind == "generator":
        entropy, spawn_key, pool_size, n_children_spawned = json.loads(attrs["seed"])
        seed = np.random.SeedSequence(
            entropy, spawn_key=spawn_key, pool_size=pool_size, n_children_spawned=n_children_spawned
        )
        bit_generator = get_object(attrs["class"])(seed)
        bit_generator.state = json.loads(attrs["state"])
        return np.random.Generator(bit_generator)
    if kind == "copy":
        return get_value(node.file[attrs["of"]]).astype(np.dtype(attrs["dtype"]))
    if kind == "transposed":
        return get_value(node.file[attrs["of"]]).T.astype(np.dtype(attrs["dtype"]))
    if kind == "transposed_sparse":
        matrix = get_value(node.file[attrs["of"]]).tocsr().T.tocsr()
        dtype = np.dtype(attrs["index_dtype"])
        matrix.indices, matrix.indptr = matrix.indices.astype(dtype), matrix.indptr.astype(dtype)
        return matrix.asformat(attrs["format"])
    if kind == "range_index":
        return pd.RangeIndex(*json.loads(attrs["range"]), name=ast.literal_eval(attrs["label"]))
    items = get_items(node)
    if kind == "array_list":
        arrays = np.split(items["values"], np.cumsum(items["lengths"])[:-1])
        return arrays if attrs["sequence"] == "list" else tuple(arrays)
    if kind == "sparse":
        indices = get_row_indices(items["indices"], items["indptr"]) if attrs["row_gaps"] else items["indices"]
        matrix = sp.csr_matrix((items["data"], indices, items["indptr"]), shape=json.loads(attrs["shape"]))
        matrix.indices, matrix.indptr = indices, items["indptr"]
        return matrix.asformat(attrs["format"])
    if kind == "spatio_genomic_counts":
        return SpatioGenomicCounts(items["values"].T)
    if kind == "sparse_frame":
        found = pd.DataFrame.sparse.from_spmatrix(items["values"], index=items["index"], columns=items["columns"])
        return found.astype(pd.SparseDtype(items["values"].dtype, json.loads(attrs["fill"])))
    if kind == "frame":
        values = get_items(node["values"])
        frame = pd.DataFrame({i: values[str(i)] for i in range(len(values))}, index=items["index"])
        frame.columns = items["columns"]
        return frame
    if kind == "series":
        return pd.Series(items["values"], index=items["index"], name=ast.literal_eval(attrs["label"]))
    if kind == "index":
        return pd.Index(items["values"], name=ast.literal_eval(attrs["label"]))
    if kind == "categorical":
        return pd.Categorical.from_codes(items["codes"], items["categories"], ordered=bool(attrs["ordered"]))
    if kind == "object_column":
        values = items["values"]
        column = np.empty(len(values), dtype=object)
        column[:] = values if attrs["type"] == "object" else values.tolist()
        column[items["nulls"]] = np.nan if attrs["null"] == "nan" else None
        return pd.array(column, dtype=object)
    if kind == "sparse_column":
        index = pd._libs.sparse.IntIndex(int(attrs["size"]), items["index"])
        return pd.arrays.SparseArray(items["values"], sparse_index=index, fill_value=json.loads(attrs["fill"]))
    if kind == "masked":
        column = np.zeros(items["mask"].shape, dtype=items["values"].dtype)
        column[~items["mask"]] = items["values"]
        return type(pd.array([], dtype=attrs["dtype"]))(column, items["mask"])
    if kind == "anndata":
        return anndata.AnnData(**items)
    if kind == "object_array":
        array = np.empty(len(items), dtype=object)
        array[:] = list(items.values())
        return array.reshape(json.loads(attrs["shape"]))
    if kind == "namedtuple":
        return get_object(attrs["class"])(**items)
    if kind == "dataclass":
        return get_object(attrs["class"])(**items)
    if kind == "object":
        found = get_object(attrs["class"])
        made = found.__new__(found)
        made.__dict__.update(items)
        return made
    if kind == "dict":
        return items
    sequence = list(items.values())
    return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[kind](sequence)


def get_replay_file(value: Any) -> h5py.File:
    """`value` encoded as `out` in an in-memory file, uncompressed: a replay's output, as `assert_same` reads it.

    Args:
        value: the value.

    Returns:
        The open file.
    """
    h5 = h5py.File(io.BytesIO(), "w")
    Writer(h5, compress=False).put(h5, "out", value)
    return h5


def assert_same(stored: h5py.Group | h5py.Dataset, replayed: h5py.Group | h5py.Dataset) -> None:
    """Assert two encoded values equal: floats to `FLOAT_RTOL` (NaN equal to NaN), all else exactly.

    Args:
        stored: the capture's node.
        replayed: the replay's node (`get_replay_file`).

    Raises:
        AssertionError: naming the first node that differs.
    """
    where = stored.name
    if stored.attrs["kind"] in SPARSE_KINDS or replayed.attrs["kind"] in SPARSE_KINDS:
        expected, found = get_value(stored), get_value(replayed)
        assert sp.issparse(found) and found.format == expected.format, f"{where}: {found!r:.80} replayed"
        assert found.shape == expected.shape and found.dtype == expected.dtype, f"{where}: shape or dtype differ"
        expected, found = expected.tocsr(), found.tocsr()
        assert (found.indptr == expected.indptr).all() and (found.indices == expected.indices).all(), where
        expected, found = expected.data, found.data
    elif stored.attrs["kind"] in ARRAY_KINDS or replayed.attrs["kind"] in ARRAY_KINDS:
        expected, found = get_value(stored), get_value(replayed)
        assert isinstance(found, np.ndarray) and found.dtype == expected.dtype, f"{where}: {found!r:.80} replayed"
    else:
        assert dict(stored.attrs) == dict(replayed.attrs), f"{where}: {dict(stored.attrs)} != {dict(replayed.attrs)}"
        if isinstance(stored, h5py.Group):
            assert sorted(stored) == sorted(replayed), f"{where}: children differ"
            for name in stored:
                assert_same(stored[name], replayed[name])
            return
        expected, found = np.asarray(get_value(stored)), np.asarray(get_value(replayed))
    if expected.dtype.kind in "fc":
        np.testing.assert_allclose(found, expected, rtol=FLOAT_RTOL, atol=0.0, equal_nan=True, err_msg=where)
    else:
        assert expected.shape == found.shape and (expected == found).all(), f"{where}: values differ"


class Capture:
    """One capture file, read only.

    Args:
        path: the file.
    """

    def __init__(self, path: Path) -> None:
        self.h5 = h5py.File(path, "r")
        config = self.h5["config"].attrs
        self.calls: list[str] = json.loads(config["stages"])
        """Every recorded call, `NN_group/function[_k]`, in the order each returned."""
        self.configs: list[str] = json.loads(config["configs"])

    def close(self) -> None:
        self.h5.close()

    def entry(self, call: str) -> h5py.Group:
        """The call's group: `in`, `out`, `rng` and its attributes.

        Args:
            call: `NN_group/function[_k]`.

        Returns:
            The group.
        """
        return self.h5[call]

    def stored(self, path: str) -> Any:
        """The value at `path`, decoded.

        Args:
            path: e.g. `05_baf/run_hmrf/out`.

        Returns:
            The value.
        """
        return get_value(self.h5[path])

    def decode(self, call: str) -> tuple[Any, list[Any], dict[str, Any]]:
        """The call's function and arguments, the global config and generators set as the call found them.

        Args:
            call: `NN_group/function[_k]`.

        Returns:
            The function, its positional and its keyword arguments.
        """
        entry = self.entry(call)
        set_global_config(YAMLConfig(json.loads(self.configs[int(entry.attrs["config"])])))
        rng = entry["rng"]
        numpy_state = ("MT19937", get_value(rng["numpy"]), *json.loads(rng.attrs["numpy"]))
        numba_state = (int(rng.attrs["numba_pos"]), [int(x) for x in get_value(rng["numba"])])
        random_state = (3, tuple(int(x) for x in get_value(rng["random"])), None)
        set_rng_state({"numpy": numpy_state, "numba": numba_state, "random": random_state})
        given = get_value(entry["in"])
        return get_object(entry.attrs["function"]), given["args"], given["kwargs"]


def put_rng(writer: Writer, group: h5py.Group, state: dict[str, Any]) -> None:
    """Write `get_rng_state`'s states as `group/rng`, as `Capture.decode` restores them; a state a call left unchanged
    is a hard link to the earlier one.

    Args:
        writer: the file's writer.
        group: the call's group.
        state: the states.
    """
    numpy_state, numba_state, random_state = state["numpy"], state["numba"], state["random"]
    rng = group.create_group("rng")
    writer.put_array(rng, "numpy", np.asarray(numpy_state[1], dtype=np.uint32))
    rng.attrs["numpy"] = json.dumps([int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])])
    writer.put_array(rng, "numba", np.asarray(numba_state[1], dtype=np.uint32))
    rng.attrs["numba_pos"] = int(numba_state[0])
    writer.put_array(rng, "random", np.asarray(random_state[1], dtype=np.uint64))
