from __future__ import annotations

import json
from pathlib import Path

import numpy as np


S1_REPAIR_SCHEMA = "alice/s1-isolated-spike-repair/v1"


def load_s1_repair(curation: dict | None) -> dict | None:
    summary = (curation or {}).get("s1_repair") or {}
    repaired_count = int(summary.get("repaired_frame_count") or 0)
    if repaired_count <= 0:
        return None
    inline = summary.get("patch")
    if isinstance(inline, dict):
        payload = inline
    else:
        artifact_path = str(summary.get("artifact_path") or "")
        if not artifact_path:
            raise RuntimeError("S1 repair metadata has no patch artifact")
        path = Path(artifact_path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read S1 repair patch: {path}") from exc
    if payload.get("schema") != S1_REPAIR_SCHEMA:
        raise RuntimeError("Unsupported S1 repair patch schema")
    if str(payload.get("dataset_id") or "") != str((curation or {}).get("dataset_id") or ""):
        raise RuntimeError("S1 repair patch dataset identity mismatch")
    if str(payload.get("episode_id") or "") != str((curation or {}).get("episode_id") or ""):
        raise RuntimeError("S1 repair patch episode identity mismatch")
    for entry in payload.get("entries") or []:
        sizes = [len(entry.get(key) or []) for key in ("source_rows", "flat_indices", "values")]
        if len(set(sizes)) != 1:
            raise RuntimeError("Malformed S1 repair patch entry")
    patch_signatures = payload.get("source_signatures") or []
    report_signatures = (curation or {}).get("source_signatures") or []
    if patch_signatures and report_signatures and patch_signatures != report_signatures:
        raise RuntimeError("S1 repair patch source-version lock mismatch")
    return payload


def apply_s1_repair(
    values: np.ndarray,
    repair: dict | None,
    source_relative: str,
    dataset_path: str,
    source_rows: np.ndarray,
) -> np.ndarray:
    output = np.asarray(values).copy()
    if repair is None or output.size == 0:
        return output
    wanted_source = str(source_relative).replace("\\", "/").casefold()
    wanted_dataset = str(dataset_path).strip("/").casefold()
    rows = np.asarray(source_rows, dtype=np.int64)
    if output.shape[0] != rows.size:
        raise ValueError("S1 repair row mapping does not match the exported values")
    flattened = output.reshape(output.shape[0], -1)
    row_positions: dict[int, np.ndarray] = {}
    for entry in repair.get("entries") or []:
        if str(entry.get("source_path") or "").replace("\\", "/").casefold() != wanted_source:
            continue
        if str(entry.get("dataset_path") or "").strip("/").casefold() != wanted_dataset:
            continue
        patch_rows = entry.get("source_rows") or []
        flat_indices = entry.get("flat_indices") or []
        patch_values = entry.get("values") or []
        if not (len(patch_rows) == len(flat_indices) == len(patch_values)):
            raise RuntimeError("Malformed S1 repair patch entry")
        for source_row, flat_index, value in zip(patch_rows, flat_indices, patch_values):
            index = int(flat_index)
            if index < 0 or index >= flattened.shape[1] or not np.isfinite(float(value)):
                raise RuntimeError("S1 repair patch contains an invalid target")
            source_row = int(source_row)
            positions = row_positions.get(source_row)
            if positions is None:
                positions = np.flatnonzero(rows == source_row)
                row_positions[source_row] = positions
            flattened[positions, index] = float(value)
    return output


def s1_repair_cell_count(
    repair: dict | None,
    source_relative: str | None = None,
    dataset_prefix: str | None = None,
) -> int:
    wanted_source = str(source_relative or "").replace("\\", "/").casefold()
    wanted_prefix = str(dataset_prefix or "").strip("/").casefold()
    return sum(
        len(entry.get("values") or [])
        for entry in (repair or {}).get("entries") or []
        if (not wanted_source or str(entry.get("source_path") or "").replace("\\", "/").casefold() == wanted_source)
        and (not wanted_prefix or str(entry.get("dataset_path") or "").strip("/").casefold().startswith(wanted_prefix))
    )
