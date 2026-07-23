from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TEXT_EXTENSIONS = {".txt", ".md", ".yaml", ".yml", ".toml", ".html", ".htm", ".log"}
MAX_TEXT_BYTES = 256 * 1024
MAX_TEXT_CHARS = 200_000
MAX_FRAME_ELEMENTS = 4096


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_text(path: Path) -> tuple[str, bool]:
    size = path.stat().st_size
    with path.open("rb") as source:
        content = source.read(MAX_TEXT_BYTES + 1)
    truncated = size > MAX_TEXT_BYTES or len(content) > MAX_TEXT_BYTES
    return content[:MAX_TEXT_BYTES].decode("utf-8", errors="replace"), truncated


def _sample_slices(shape: tuple[int, ...], budget: int = 1024) -> tuple[slice, ...]:
    if not shape:
        return ()
    limits = [min(int(size), 8 if index == 0 else 16) for index, size in enumerate(shape)]
    while math.prod(max(1, value) for value in limits) > budget:
        index = max(range(len(limits)), key=lambda item: limits[item])
        if limits[index] <= 1:
            break
        limits[index] = max(1, limits[index] // 2)
    return tuple(slice(0, value) for value in limits)


def _frame_slices(shape: tuple[int, ...], budget: int = MAX_FRAME_ELEMENTS) -> tuple[slice, ...]:
    """Build a bounded slice for one frame without materialising the full value."""
    if not shape:
        return ()
    limits = [min(int(size), 256) for size in shape]
    while math.prod(max(1, value) for value in limits) > budget:
        index = max(range(len(limits)), key=lambda item: limits[item])
        if limits[index] <= 1:
            break
        limits[index] = max(1, limits[index] // 2)
    return tuple(slice(0, value) for value in limits)


def _frame_payload(
    *,
    mode: str,
    field: str,
    index: int,
    frame_count: int,
    value: Any,
    dtype: str,
    full_value_shape: list[int],
    fields: list[dict[str, Any]],
) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "mode": mode,
        "field": field,
        "frame_index": index,
        "frame_count": frame_count,
        "value": _safe(value),
        "value_shape": list(array.shape),
        "full_value_shape": full_value_shape,
        "dtype": dtype,
        "fields": fields,
        "truncated": list(array.shape) != full_value_shape,
        "error": None,
    }


def _frame_hdf5(path: Path, index: int, field: str | None) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        fields: list[dict[str, Any]] = []

        def visitor(name, obj):
            if len(fields) >= 500 or not isinstance(obj, h5py.Dataset):
                return
            fields.append({
                "path": name,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "frame_count": int(obj.shape[0]) if obj.shape else 1,
            })

        handle.visititems(visitor)
        available = {item["path"] for item in fields}
        selected = field if field in available else (fields[0]["path"] if fields else None)
        if selected is None:
            raise ValueError("HDF5 文件中没有可查阅的数据字段")
        dataset = handle[selected]
        frame_count = int(dataset.shape[0]) if dataset.shape else 1
        resolved_index = min(max(0, index), max(0, frame_count - 1))
        full_shape = list(dataset.shape[1:]) if dataset.shape else []
        if dataset.shape:
            value = np.asarray(dataset[(resolved_index, *_frame_slices(tuple(dataset.shape[1:])))])
        else:
            value = np.asarray(dataset[()])
        return _frame_payload(
            mode="hdf5",
            field=selected,
            index=resolved_index,
            frame_count=frame_count,
            value=value,
            dtype=str(dataset.dtype),
            full_value_shape=full_shape,
            fields=fields,
        )


def _json_fields(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        return [{
            "path": str(key),
            "shape": [len(value)] if isinstance(value, list) else [],
            "dtype": type(value).__name__,
            "frame_count": len(value) if isinstance(value, list) else 1,
        } for key, value in list(parsed.items())[:500]]
    if isinstance(parsed, list):
        return [{"path": "$", "shape": [len(parsed)], "dtype": "list", "frame_count": len(parsed)}]
    return [{"path": "$", "shape": [], "dtype": type(parsed).__name__, "frame_count": 1}]


def _frame_json(path: Path, index: int, field: str | None) -> dict[str, Any]:
    text, truncated = _read_text(path)
    if truncated:
        raise ValueError("JSON 文件超过安全解析上限，无法可靠地按帧查阅")
    if path.suffix.lower() == ".jsonl":
        lines = [line for line in text.splitlines() if line.strip()]
        parsed: Any = [json.loads(line) for line in lines]
    else:
        parsed = json.loads(text)
    fields = _json_fields(parsed)
    available = {item["path"] for item in fields}
    selected = field if field in available else fields[0]["path"]
    source = parsed.get(selected) if isinstance(parsed, dict) else parsed
    frame_count = len(source) if isinstance(source, list) else 1
    resolved_index = min(max(0, index), max(0, frame_count - 1))
    value = source[resolved_index] if isinstance(source, list) and source else source
    array = np.asarray(value) if not isinstance(value, dict) else np.asarray(value, dtype=object)
    full_shape = list(array.shape)
    if isinstance(value, list) and array.dtype != object and array.size > MAX_FRAME_ELEMENTS:
        value = array[_frame_slices(tuple(array.shape))]
    return _frame_payload(
        mode="json",
        field=selected,
        index=resolved_index,
        frame_count=frame_count,
        value=value,
        dtype=type(source).__name__,
        full_value_shape=full_shape,
        fields=fields,
    )


def _frame_numpy(path: Path, index: int, field: str | None) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            fields = [{
                "path": key,
                "shape": list(archive[key].shape),
                "dtype": str(archive[key].dtype),
                "frame_count": int(archive[key].shape[0]) if archive[key].shape else 1,
            } for key in archive.files]
            available = {item["path"] for item in fields}
            selected = field if field in available else (fields[0]["path"] if fields else None)
            if selected is None:
                raise ValueError("NPZ 文件中没有数组")
            array = archive[selected]
            frame_count = int(array.shape[0]) if array.shape else 1
            resolved_index = min(max(0, index), max(0, frame_count - 1))
            full_shape = list(array.shape[1:]) if array.shape else []
            value = np.asarray(array[(resolved_index, *_frame_slices(tuple(array.shape[1:])))]) if array.shape else np.asarray(array[()])
            return _frame_payload(mode="numpy", field=selected, index=resolved_index, frame_count=frame_count, value=value, dtype=str(array.dtype), full_value_shape=full_shape, fields=fields)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    fields = [{"path": "$", "shape": list(array.shape), "dtype": str(array.dtype), "frame_count": int(array.shape[0]) if array.shape else 1}]
    frame_count = int(array.shape[0]) if array.shape else 1
    resolved_index = min(max(0, index), max(0, frame_count - 1))
    full_shape = list(array.shape[1:]) if array.shape else []
    value = np.asarray(array[(resolved_index, *_frame_slices(tuple(array.shape[1:])))]) if array.shape else np.asarray(array[()])
    return _frame_payload(mode="numpy", field="$", index=resolved_index, frame_count=frame_count, value=value, dtype=str(array.dtype), full_value_shape=full_shape, fields=fields)


def _frame_table(path: Path, index: int) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    requested = max(0, index)
    value = None
    frame_count = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as source:
        reader = csv.DictReader(source, delimiter=delimiter)
        for row_index, row in enumerate(reader):
            if row_index == requested:
                value = row
            frame_count = row_index + 1
    if value is None:
        if frame_count == 0:
            raise ValueError("表格中没有数据行")
        resolved = frame_count - 1
        with path.open("r", encoding="utf-8", errors="replace", newline="") as source:
            value = next(itertools.islice(csv.DictReader(source, delimiter=delimiter), resolved, None))
    else:
        resolved = requested
    fields = [{"path": "$", "shape": [frame_count], "dtype": "row", "frame_count": frame_count}]
    return _frame_payload(mode="table", field="$", index=resolved, frame_count=frame_count, value=value, dtype="row", full_value_shape=[], fields=fields)


def _frame_text(path: Path, index: int) -> dict[str, Any]:
    content, truncated = _read_text(path)
    blocks = [block.strip() for block in content.split("\n\n") if block.strip()] or [content]
    resolved_index = min(max(0, index), max(0, len(blocks) - 1))
    fields = [{"path": "$", "shape": [len(blocks)], "dtype": "paragraph", "frame_count": len(blocks)}]
    result = _frame_payload(mode="text", field="$", index=resolved_index, frame_count=len(blocks), value=blocks[resolved_index], dtype="paragraph", full_value_shape=[], fields=fields)
    result["truncated"] = truncated
    return result


def _frame_parquet(path: Path, index: int, field: str | None) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path)
    frame_count = source.metadata.num_rows
    fields = [{
        "path": arrow_field.name,
        "shape": [frame_count],
        "dtype": str(arrow_field.type),
        "frame_count": frame_count,
    } for arrow_field in source.schema_arrow]
    available = {item["path"] for item in fields}
    selected = field if field in available else (fields[0]["path"] if fields else None)
    if selected is None:
        raise ValueError("Parquet 文件中没有可查阅的数据字段")
    resolved_index = min(max(0, index), max(0, frame_count - 1))
    row_group = 0
    offset = resolved_index
    for group_index in range(source.num_row_groups):
        rows = source.metadata.row_group(group_index).num_rows
        if offset < rows:
            row_group = group_index
            break
        offset -= rows
    table = source.read_row_group(row_group, columns=[selected]).slice(offset, 1)
    row = table.to_pylist()[0] if table.num_rows else {}
    value = row.get(selected)
    array = np.asarray(value)
    full_shape = list(array.shape)
    if array.size > MAX_FRAME_ELEMENTS:
        value = array[_frame_slices(tuple(array.shape))]
    dtype = str(source.schema_arrow.field(selected).type)
    return _frame_payload(mode="parquet", field=selected, index=resolved_index, frame_count=frame_count, value=value, dtype=dtype, full_value_shape=full_shape, fields=fields)


def preview_file_frame(path: Path, relative_path: str, index: int = 0, field: str | None = None) -> dict[str, Any]:
    suffix = path.suffix.lower()
    base = {"name": path.name, "relative_path": relative_path, "extension": suffix}
    try:
        if suffix in {".h5", ".hdf5", ".h5df"}:
            result = _frame_hdf5(path, index, field)
        elif suffix in {".json", ".jsonl"}:
            result = _frame_json(path, index, field)
        elif suffix in {".npy", ".npz"}:
            result = _frame_numpy(path, index, field)
        elif suffix in {".csv", ".tsv"}:
            result = _frame_table(path, index)
        elif suffix == ".parquet":
            result = _frame_parquet(path, index, field)
        elif suffix in TEXT_EXTENSIONS or not suffix:
            result = _frame_text(path, index)
        else:
            raise ValueError("该文件类型没有可索引的逐帧数据")
        return {**base, **result}
    except Exception as exc:
        return {**base, "mode": "error", "error": str(exc)[:500], "fields": [], "truncated": False}


def _preview_json(path: Path) -> dict[str, Any]:
    text, truncated = _read_text(path)
    suffix = path.suffix.lower()
    parsed: Any = None
    if suffix == ".jsonl":
        values = []
        for line in text.splitlines()[:100]:
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError:
                values.append(line)
        parsed = values
        content = json.dumps(parsed, ensure_ascii=False, indent=2, default=str)
    else:
        try:
            parsed = json.loads(text)
            content = json.dumps(parsed, ensure_ascii=False, indent=2, default=str)
        except json.JSONDecodeError:
            content = text
    if len(content) > MAX_TEXT_CHARS:
        content = content[:MAX_TEXT_CHARS]
        truncated = True
    sections = []
    if isinstance(parsed, dict):
        for index, (key, value) in enumerate(list(parsed.items())[:200]):
            rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
            sections.append({
                "id": f"json-{index}",
                "title": str(key),
                "content": rendered[:50_000],
                "truncated": len(rendered) > 50_000,
                "kind": type(value).__name__,
            })
    elif isinstance(parsed, list):
        for index, value in enumerate(parsed[:200]):
            rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
            sections.append({
                "id": f"json-{index}",
                "title": f"[{index}]",
                "content": rendered[:30_000],
                "truncated": len(rendered) > 30_000,
                "kind": type(value).__name__,
            })
    elif parsed is not None:
        sections.append({"id": "json-0", "title": "Value", "content": content, "truncated": truncated, "kind": type(parsed).__name__})
    return {"mode": "json", "content": content, "sections": sections, "section_count": len(sections), "truncated": truncated}


def _preview_hdf5(path: Path, field: str | None) -> dict[str, Any]:
    import h5py

    datasets: list[dict[str, Any]] = []
    selected = None
    attributes: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        attributes = {str(key): _safe(value) for key, value in list(handle.attrs.items())[:50]}

        def visitor(name, obj):
            if len(datasets) >= 500 or not isinstance(obj, h5py.Dataset):
                return
            datasets.append({
                "path": name,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "compression": obj.compression,
            })

        handle.visititems(visitor)
        available = {item["path"] for item in datasets}
        selected = field if field in available else (datasets[0]["path"] if datasets else None)
    return {
        "mode": "hdf5",
        "datasets": datasets,
        "selected_field": selected,
        "frame_comparison": True,
        "attributes": attributes,
        "truncated": len(datasets) >= 500,
    }


def _preview_table(path: Path) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as source:
        reader = csv.DictReader(source, delimiter=delimiter)
        columns = list(reader.fieldnames or [])[:80]
        for row in reader:
            rows.append({column: row.get(column) for column in columns})
            if len(rows) >= 50:
                break
    return {"mode": "table", "columns": columns, "rows": rows, "truncated": len(rows) >= 50}


def _preview_parquet(path: Path, field: str | None) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path)
    row_count = source.metadata.num_rows
    first_batch = next(source.iter_batches(batch_size=1), None)
    first_row = first_batch.to_pylist()[0] if first_batch is not None and first_batch.num_rows else {}
    datasets = [{
        "path": arrow_field.name,
        "shape": [row_count, *list(np.asarray(first_row.get(arrow_field.name)).shape)] if first_row.get(arrow_field.name) is not None else [row_count],
        "dtype": str(arrow_field.type),
        "compression": "columnar",
    } for arrow_field in source.schema_arrow]
    available = {item["path"] for item in datasets}
    selected = field if field in available else (datasets[0]["path"] if datasets else None)
    return {
        "mode": "parquet",
        "datasets": datasets,
        "selected_field": selected,
        "row_count": row_count,
        "row_groups": source.num_row_groups,
        "truncated": False,
        "frame_comparison": True,
    }


def _preview_numpy(path: Path, field: str | None) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            arrays = [{"path": key, "shape": list(archive[key].shape), "dtype": str(archive[key].dtype)} for key in archive.files]
            selected = field if field in archive.files else (archive.files[0] if archive.files else None)
            values = np.asarray(archive[selected][_sample_slices(tuple(archive[selected].shape))]) if selected else None
            return {"mode": "numpy", "datasets": arrays, "selected_field": selected, "sample": _safe(values), "sample_shape": list(values.shape) if values is not None else None}
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    values = np.asarray(array[_sample_slices(tuple(array.shape))])
    return {"mode": "numpy", "datasets": [{"path": "$", "shape": list(array.shape), "dtype": str(array.dtype)}], "selected_field": "$", "sample": _safe(values), "sample_shape": list(values.shape)}


def preview_file(path: Path, relative_path: str, field: str | None = None) -> dict[str, Any]:
    suffix = path.suffix.lower()
    base = {
        "name": path.name,
        "relative_path": relative_path,
        "extension": suffix,
        "size_bytes": path.stat().st_size,
    }
    try:
        if suffix in {".json", ".jsonl"}:
            preview = _preview_json(path)
        elif suffix in {".h5", ".hdf5", ".h5df"}:
            preview = _preview_hdf5(path, field)
        elif suffix in {".csv", ".tsv"}:
            preview = _preview_table(path)
        elif suffix == ".parquet":
            preview = _preview_parquet(path, field)
        elif suffix in {".npy", ".npz"}:
            preview = _preview_numpy(path, field)
        elif suffix in TEXT_EXTENSIONS or not suffix:
            content, truncated = _read_text(path)
            preview = {"mode": "text", "content": content, "truncated": truncated}
        else:
            with path.open("rb") as source:
                header = source.read(256)
            preview = {"mode": "binary", "hex": header.hex(" "), "truncated": path.stat().st_size > 256}
        return {**base, **preview, "error": None}
    except Exception as exc:
        return {**base, "mode": "error", "error": str(exc)[:500], "truncated": False}
