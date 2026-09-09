#!/usr/bin/env python3
"""Add manufacturer-datasheet typical current measurements to the catalog.

The cached text files are generated from official PDFs.  This adapter is
deliberately conservative: it only accepts rows with an explicit current unit
and a ``Typ`` table value, and it keeps the datasheet's test conditions in the
feature label.  It does not infer a value for a sibling family or turn a
unitless mode count into a power measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CURRENT_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\D{1,10}(?P<unit>mA|uA|μA|µA|nA)\b", re.I)
FREQ_RE = re.compile(
    r"(?:system\s*clock|fHCLK|\bclock)\s*=\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)",
    re.I,
)
VOLTAGE_RE = re.compile(r"(?:VDD(?:A)?|VCORE|LDO voltage|voltage)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*V\b", re.I)
TABLE_RE = re.compile(r"^\s*Table\s+[0-9 -]+\.?\s*.*Power\s+consumption\s+characteristics", re.I)
ARTERY_TABLE_RE = re.compile(r"Typical current consumption in (Run|Sleep) mode", re.I)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def compact(value: str) -> str:
    value = value.strip()
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value or "0"


def frequency_hz(match: re.Match[str]) -> int:
    factors = {"ghz": 1_000_000_000, "mhz": 1_000_000, "khz": 1_000, "hz": 1}
    return round(float(match.group("value")) * factors[match.group("unit").lower()])


def clean_condition(lines: list[str]) -> str:
    return " ".join(" ".join(line.split()) for line in lines if line.strip())


def clean_value_text(lines: list[str]) -> str:
    """Join decimal digits split over PDF text lines (e.g. ``130.1`` + ``5``)."""
    text = clean_condition(lines)
    return re.sub(
        r"(?P<whole>\d+\.\d+)\s+(?P<tail>\d)(?=\s*\D{0,10}(?:mA|uA|μA|µA|nA)\b)",
        r"\g<whole>\g<tail>",
        text,
        flags=re.I,
    )


def line_has_mode(text: str) -> str | None:
    lowered = text.lower()
    if "wakeup" in lowered:
        return None
    if "deep-sleep" in lowered or "deep sleep" in lowered:
        return "deep-sleep"
    if "standby mode" in lowered:
        return "standby"
    if "sleep mode" in lowered:
        return "sleep"
    if "run mode" in lowered:
        return "run"
    return None


def product_line_for(stem: str, product_lines: set[str]) -> str | None:
    upper = stem.upper()
    candidates = [line for line in product_lines if upper.startswith(line.upper())]
    return max(candidates, key=len) if candidates else None


def source_id_for(vendor: str, product_line: str, filename: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", filename.lower()).strip("-")
    return f"{vendor}:datasheet:{product_line.lower()}:{safe}"


def feature(name: str, source_document_id: str) -> dict[str, str]:
    return {
        "type": "PowerOther",
        "name": name,
        "source_kind": "manufacturer_datasheet",
        "source_document_id": source_document_id,
        "verification_status": "manufacturer_datasheet",
        "category": "power",
    }


def applies_to_scope(device: dict[str, str], scope: str) -> bool:
    """Apply the common GigaDevice x-suffix table scopes exactly when known."""
    lowered = scope.lower()
    # Examples in official tables: GD32F103x4/6/8/B and GD32F103xC/D/E/F/G/I/K.
    groups = re.findall(r"x([0-9a-z](?:/[0-9a-z])+)", lowered)
    if not groups:
        return True
    suffix = device.get("device_name", "").lower().split(device.get("product_line", "").lower(), 1)[-1]
    if len(suffix) < 2:
        return False
    return suffix[1] in {item for item in groups[0].split("/") if item}


def parse_artery(path: Path, product_line: str) -> list[tuple[str, str, str, int]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    run_headings = [
        i for i, line in enumerate(lines)
        if (match := ARTERY_TABLE_RE.search(line)) and match.group(1).lower() == "run"
    ]
    if not run_headings:
        return []
    run_start = run_headings[-1]
    sleep_start = next((i for i in range(run_start + 1, len(lines)) if re.search(r"Typical current consumption in Sleep mode", lines[i], re.I)), None)
    if sleep_start is None:
        return []
    sections = [("run", run_start + 1, sleep_start), ("sleep", sleep_start + 1, len(lines))]
    results: list[tuple[str, str, str, int]] = []
    for mode, start, end in sections:
        first: tuple[str, str, str, int] | None = None
        for raw in lines[start:end]:
            text = " ".join(raw.split())
            match = re.match(r"^(\d+(?:\.\d+)?)\s*(GHz|MHz|kHz|Hz)\s+(.+)$", text, re.I)
            if not match:
                continue
            values = re.findall(r"\d+(?:\.\d+)?", match.group(3))
            if len(values) < 2:
                continue
            # Newer AT32 tables include the LDO voltage; older tables rely on
            # the common VDD=3.3 V, TA=25 C condition in the preamble.
            has_voltage = len(values) >= 3 and float(values[0]) <= 5
            if has_voltage:
                voltage = values[0]
                current_values = values[1:]
            else:
                voltage = "3.3"
                current_values = values
            if len(current_values) < 2:
                continue
            freq = frequency_hz(re.match(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)", text, re.I))  # type: ignore[arg-type]
            # Values are ordered as enabled, disabled.  F435/F437 and
            # F455/456/457 have one extra enabled column for the model split.
            if len(current_values) >= 3 and product_line in {"AT32F435", "AT32F437", "AT32F455", "AT32F456", "AT32F457"}:
                if product_line in {"AT32F435", "AT32F455", "AT32F456"}:
                    enabled = current_values[1] if product_line == "AT32F435" else current_values[0]
                else:
                    enabled = current_values[0] if product_line == "AT32F437" else current_values[1]
                disabled = current_values[-1]
            else:
                enabled, disabled = current_values[0], current_values[1]
            first = (compact(enabled), compact(disabled), voltage, freq)
            break
        if first:
            enabled, disabled, voltage, freq = first
            results.append((mode, "enabled", enabled, freq))
            results.append((mode, "disabled", disabled, freq))
            # Voltage is carried separately in the condition suffix by the
            # caller; returning it as a compact pseudo-record keeps parsing
            # independent of the CSV writer.
            results[-2] = (results[-2][0], f"{results[-2][1]}|{voltage}", results[-2][2], results[-2][3])
            results[-1] = (results[-1][0], f"{results[-1][1]}|{voltage}", results[-1][2], results[-1][3])
    return results


def parse_giga(path: Path, product_line: str) -> list[tuple[str, str, str, str, int, str, str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    headings = [i for i, line in enumerate(lines) if TABLE_RE.search(line) and i > 300]
    if not headings:
        return []
    results: list[tuple[str, str, str, str, int, str, str]] = []
    for position, heading in enumerate(headings):
        end = headings[position + 1] if position + 1 < len(headings) else min(len(lines), heading + 1_000)
        scope = clean_condition(lines[heading : min(end, heading + 2)])
        mode: str | None = None
        seen: dict[tuple[str, str], tuple[str, str, str, str, int, str, str]] = {}
        for index in range(heading + 1, end):
            raw = " ".join(lines[index].split())
            current_mode = line_has_mode(raw)
            if current_mode:
                mode = current_mode
            if mode not in {"run", "sleep"}:
                continue
            # PDF extraction may put ``System clock`` and ``=120 MHz`` on
            # adjacent lines.  Match across the next line without changing
            # the row/value pairing below.
            row_context = clean_condition(lines[index : min(end, index + 2)])
            freq_match = FREQ_RE.search(row_context)
            if not freq_match:
                continue
            value_match = None
            value_line = ""
            value_index = index
            for lookahead in range(index, min(end, index + 8)):
                for span in range(1, 4):
                    value_text = clean_value_text(lines[lookahead : min(end, lookahead + span)])
                    value_match = CURRENT_RE.search(value_text)
                    if value_match:
                        value_line = value_text
                        value_index = lookahead + span - 1
                        break
                if value_match:
                    break
            if not value_match:
                continue
            # Keep the condition for this row only.  Looking several lines
            # past the value can pull the next "disabled" row into an
            # enabled row (the PDF text extractor puts each cell on a line).
            condition_start = index
            # When the frequency is matched across two lines, the scan may
            # start on the previous row's value. Prefer the VDD line in the
            # current two-line context.
            found_vdd = False
            for forward in range(index, min(end, index + 2)):
                if re.search(r"VDD|VDDA|VCORE", " ".join(lines[forward].split()), re.I):
                    condition_start = forward
                    found_vdd = True
                    break
            if not found_vdd:
                for back in range(index - 1, max(heading, index - 5), -1):
                    previous = " ".join(lines[back].split())
                    if re.search(r"VDD|VDDA|VCORE", previous, re.I):
                        condition_start = back
                        break
            condition_lines = lines[condition_start : min(end, value_index + 1)]
            condition = clean_condition(condition_lines)
            status = "disabled" if "disabled" in condition.lower() else "enabled" if "enabled" in condition.lower() else "unspecified"
            # A page break can place the enabled/disabled cell after the
            # numeric Typ value. Recover it only when the row has no status.
            if status == "unspecified":
                for after in range(value_index + 1, min(end, value_index + 7)):
                    following = " ".join(lines[after].split()).lower()
                    if "disabled" in following:
                        status = "disabled"
                        break
                    if "enabled" in following:
                        status = "enabled"
                        break
            if status == "unspecified":
                continue
            freq = frequency_hz(freq_match)
            candidate = (mode, status, compact(value_match.group("value")), value_match.group("unit").replace("μ", "u").replace("µ", "u"), freq, condition, scope)
            key = (mode, status)
            if key not in seen or freq > seen[key][4]:
                seen[key] = candidate
        results.extend(seen.values())
    return results


def add_artery(rows: list[dict[str, str]], path: Path, product_line: str) -> int:
    source_document_id = source_id_for("artery", product_line, path.name)
    parsed = parse_artery(path, product_line)
    added = 0
    for row in rows:
        if row.get("manufacturer") != "Artery" or row.get("product_line") != product_line:
            continue
        features = json.loads(row.get("features_json") or "[]")
        for mode, status_voltage, value, freq in parsed:
            status, voltage = status_voltage.split("|", 1)
            unit = "mA"
            label = f"Typical supply current in {mode} mode, all peripherals {status}: {value} {unit} at {freq / 1_000_000:g} MHz, VDD/LDO {voltage} V, TA=25 C"
            item = feature(label, source_document_id)
            if not any(existing.get("name") == label for existing in features):
                features.append(item)
                added += 1
        row["features_json"] = json.dumps(features, ensure_ascii=False)
    return added


def add_giga(rows: list[dict[str, str]], path: Path, product_line: str) -> int:
    source_document_id = source_id_for("gigadevice", product_line, path.name)
    parsed = parse_giga(path, product_line)
    added = 0
    for row in rows:
        if row.get("manufacturer") != "GigaDevice" or row.get("product_line") != product_line:
            continue
        features = json.loads(row.get("features_json") or "[]")
        for mode, status, value, unit, freq, condition, scope in parsed:
            if not applies_to_scope(row, scope):
                continue
            voltage = VOLTAGE_RE.search(condition)
            voltage_text = f", {voltage.group(0)}" if voltage else ""
            label = f"Typical supply current in {mode} mode, all peripherals {status}: {value} {unit} at {freq / 1_000_000:g} MHz{voltage_text}; condition: {condition}"
            item = feature(label, source_document_id)
            if not any(existing.get("name") == label for existing in features):
                features.append(item)
                added += 1
        row["features_json"] = json.dumps(features, ensure_ascii=False)
    return added


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--artery-cache", type=Path, default=root / "cache" / "artery-documents" / "pdf")
    parser.add_argument("--gigadevice-cache", type=Path, default=root / "cache" / "gigadevice-datasheets" / "pdf")
    args = parser.parse_args()

    device_path = args.data_dir / "device-variants.csv"
    rows = read_csv(device_path)
    if not rows:
        raise SystemExit(f"no devices found: {device_path}")
    fields = list(rows[0].keys())
    product_lines = {row.get("product_line", "") for row in rows}
    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "documents": [], "features_added": 0}

    for path in sorted(args.artery_cache.glob("*.txt")):
        stem_match = re.match(r"ds-(at32[a-z0-9]+)\.pdf", path.name, re.I)
        if not stem_match:
            continue
        line = product_line_for(stem_match.group(1), product_lines)
        if not line:
            continue
        added = add_artery(rows, path, line)
        if added:
            report["documents"].append({"vendor": "Artery", "product_line": line, "file": path.name, "features_added": added})
            report["features_added"] += added

    for path in sorted(args.gigadevice_cache.glob("*.txt")):
        stem_match = re.match(r"(gd32[a-z0-9]+)-datasheet-", path.name, re.I)
        if not stem_match:
            continue
        line = product_line_for(stem_match.group(1), product_lines)
        if not line:
            continue
        added = add_giga(rows, path, line)
        if added:
            report["documents"].append({"vendor": "GigaDevice", "product_line": line, "file": path.name, "features_added": added})
            report["features_added"] += added

    write_csv(device_path, fields, rows)
    (args.data_dir / "power-augmentation-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"features_added": report["features_added"], "documents": len(report["documents"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
