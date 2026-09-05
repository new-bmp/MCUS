#!/usr/bin/env python3
"""Augment STMicroelectronics parts with typical current from official PDFs.

This importer is intentionally conservative.  It downloads only datasheet
URLs already attached to ST devices in the catalog, extracts tables whose
title says ``Typical current consumption``, and keeps rows with an explicit
current unit and clock frequency.  Values are applied to the product lines
covered by that exact datasheet; no sibling-family inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader


CURRENT_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mA|uA|μA|µA|nA)\b", re.I)
FREQ_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)\b", re.I)
TABLE_RE = re.compile(
    r"Typical\s+(?:and\s+maximum\s+)?current\s+consumption\s+in\s+"
    r"(?P<mode>Run|Sleep|Stop|Standby)\s+mode",
    re.I,
)
VOLTAGE_RE = re.compile(r"(?:V\s*DD(?:A)?|VCORE|voltage)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*V\b", re.I)
TEMP_RE = re.compile(r"T\s*A\s*=\s*(-?\d+(?:\.\d+)?)\s*°?\s*C", re.I)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def compact(value: str) -> str:
    value = value.strip()
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value or "0"


def frequency_hz(value: str, unit: str) -> int:
    return round(float(value) * {"ghz": 1_000_000_000, "mhz": 1_000_000, "khz": 1_000, "hz": 1}[unit.lower()])


def normalize_lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines()]


def extract_pdf_text(pdf_path: Path, txt_path: Path) -> str:
    if txt_path.exists() and txt_path.stat().st_mtime >= pdf_path.stat().st_mtime:
        return txt_path.read_text(encoding="utf-8", errors="ignore")
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    txt_path.write_text(text, encoding="utf-8")
    return text


def download(url: str, pdf_path: Path) -> bool:
    if pdf_path.exists() and pdf_path.stat().st_size > 100_000:
        return True
    try:
        # ST's CDN sometimes returns a non-standard 567 response to Python's
        # urllib while serving the same request to curl/PowerShell.  Prefer
        # curl when available, then fall back to urllib for other hosts.
        curl = subprocess.run(
            ["curl.exe", "-L", "--fail", "--silent", "--show-error", "--max-time", "45",
             "-A", "Mozilla/5.0 (MCUS catalog)", "-o", str(pdf_path), url],
            capture_output=True, text=True, timeout=55,
        )
        if curl.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 100_000:
            if pdf_path.read_bytes()[:4] == b"%PDF":
                return True
        if pdf_path.exists():
            pdf_path.unlink()
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MCUS catalog)"})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read()
        if not data.startswith(b"%PDF"):
            return False
        pdf_path.write_bytes(data)
        return True
    except Exception as exc:  # network sources are optional; report and continue
        if pdf_path.exists():
            pdf_path.unlink()
        print(f"skip download {url}: {exc}")
        return False


def parse_tables(text: str) -> list[tuple[str, str, str, int, str, str]]:
    """Return (mode, enabled/disabled, value, frequency, unit) rows.

    ST PDFs put the unit in a column header and often split it onto a line of
    its own.  Rows are therefore parsed only when a line starts with a clock
    frequency and has two numeric typical columns.  The highest-frequency
    row in each mode/status is retained by the caller.
    """
    lines = normalize_lines(text)
    headings = [
        (i, m.group("mode").lower())
        for i, line in enumerate(lines)
        # The table of contents and figure index repeat the same titles.  A
        # real table heading appears well after the front matter and starts
        # with its Table number.
        if i > 500 and re.match(r"^Table\s+\d+", line, re.I) and (m := TABLE_RE.search(line))
    ]
    result: list[tuple[str, str, str, int, str, str]] = []
    for pos, (heading, mode) in enumerate(headings):
        end = min(len(lines), heading + 320)
        # Stop at the next numbered table even when it is a different kind
        # of electrical-characteristics table, otherwise later rows can be
        # mistaken for frequency/current pairs.
        for candidate in range(heading + 1, end):
            if re.match(r"^Table\s+\d+", lines[candidate], re.I):
                end = candidate
                break
        block = lines[heading:end]
        # A table with a maximum column can still be used: use the first
        # explicitly labelled Typ values, never a Max column.
        unit = ""
        for line in block[:90]:
            match = re.search(r"\b(mA|uA|μA|µA|nA)\b", line, re.I)
            if match:
                unit = match.group(1).replace("μ", "u").replace("µ", "u")
                break
        if not unit:
            continue
        condition_parts: list[str] = []
        voltage = VOLTAGE_RE.search(" ".join(block[:100]))
        temperature = TEMP_RE.search(" ".join(block[:100]))
        if voltage:
            condition_parts.append(f"VDD={voltage.group(1)} V")
        if temperature:
            condition_parts.append(f"TA={temperature.group(1)} C")
        condition = ", ".join(condition_parts)
        candidates: dict[tuple[str, str], tuple[str, str, str, int, str, str]] = {}
        status: str | None = None
        for index, line in enumerate(block[1:], start=heading + 1):
            lower = line.lower()
            if "all peripherals enabled" in lower:
                status = "enabled"
            elif "all peripherals disabled" in lower:
                status = "disabled"
            # Most tables have the status in the column heading before rows.
            match = re.match(r"^(?P<freq>\d+(?:\.\d+)?)\s*(?P<funit>GHz|MHz|kHz|Hz)\s+(?P<rest>.+)$", line, re.I)
            if not match:
                continue
            numbers = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", match.group("rest"))
            if len(numbers) < 2:
                continue
            freq = frequency_hz(match.group("freq"), match.group("funit"))
            # In ST tables the two Typ columns are enabled then disabled.  If
            # the heading was not extracted, preserve both columns explicitly.
            values = numbers[:2]
            for row_status, value in (("enabled", values[0]), ("disabled", values[1])):
                key = (mode, row_status)
                row = (mode, row_status, compact(value), freq, unit, condition)
                if key not in candidates or freq > candidates[key][3]:
                    candidates[key] = row
        result.extend(candidates.values())
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "st-datasheets" / "pdf")
    parser.add_argument("--offline", action="store_true", help="parse only PDFs already present in the cache")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.data_dir / "device-variants.csv")
    fields = list(rows[0].keys())

    # Build URL -> exact product lines from catalog documents.
    documents: dict[str, set[str]] = {}
    for row in rows:
        if row.get("manufacturer") != "STMicroelectronics":
            continue
        try:
            entries = json.loads(row.get("documents_json") or "[]")
        except json.JSONDecodeError:
            entries = []
        for entry in entries:
            url = str(entry.get("name") or "")
            if "st.com/resource/en/datasheet/" in url.lower():
                documents.setdefault(url, set()).add(row.get("product_line", ""))

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "documents": [],
        "features_added": 0,
        "downloaded": 0,
        "failed_downloads": [],
        "cache_misses": [],
    }
    for url, product_lines in sorted(documents.items()):
        basename = Path(url.split("?", 1)[0]).name.lower()
        if not basename.endswith(".pdf"):
            basename += ".pdf"
        # Keep duplicate filenames from different URLs separate and make the
        # cache deterministic across runs.
        digest = hashlib.sha1(url.encode()).hexdigest()[:8]
        pdf_path = args.cache_dir / f"{basename[:-4]}-{digest}.pdf"
        if args.offline and not pdf_path.exists():
            report["cache_misses"].append(url)
            continue
        if not download(url, pdf_path):
            report["failed_downloads"].append(url)
            continue
        report["downloaded"] += 1
        txt_path = pdf_path.with_suffix(".txt")
        try:
            parsed = parse_tables(extract_pdf_text(pdf_path, txt_path))
        except Exception as exc:
            print(f"skip parse {pdf_path.name}: {exc}")
            continue
        if not parsed:
            continue
        source_id = f"st:datasheet:{hashlib.sha1(url.encode()).hexdigest()[:16]}"
        added = 0
        for row in rows:
            if row.get("manufacturer") != "STMicroelectronics" or row.get("product_line") not in product_lines:
                continue
            try:
                features = json.loads(row.get("features_json") or "[]")
            except json.JSONDecodeError:
                features = []
            # Rebuild this importer-owned subset so improved condition text
            # replaces older labels instead of creating duplicate readings.
            features = [
                item for item in features
                if not (
                    item.get("type") == "PowerOther"
                    and str(item.get("source_document_id", "")).startswith("st:datasheet:")
                )
            ]
            for mode, status, value, freq, unit, condition in parsed:
                condition_suffix = f"; condition: {condition}" if condition else ""
                label = (
                    f"Typical supply current in {mode} mode, all peripherals {status}: "
                    f"{value} {unit} at {freq / 1_000_000:g} MHz{condition_suffix}; official ST datasheet table"
                )
                item = {
                    "type": "PowerOther",
                    "name": label,
                    "source_kind": "manufacturer_datasheet",
                    "source_document_id": source_id,
                    "verification_status": "manufacturer_datasheet",
                    "category": "power",
                }
                if not any(existing.get("name") == label for existing in features):
                    features.append(item)
                    added += 1
            row["features_json"] = json.dumps(features, ensure_ascii=False)
        if parsed:
            report["documents"].append({
                "vendor": "STMicroelectronics",
                "product_lines": sorted(product_lines),
                "url": url,
                "file": pdf_path.name,
                "measurements": len(parsed),
                "features_added": added,
            })
            report["features_added"] += added

    write_csv(args.data_dir / "device-variants.csv", fields, rows)
    (args.data_dir / "st-power-augmentation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"documents": len(report["documents"]), "features_added": report["features_added"], "failed_downloads": len(report["failed_downloads"]), "cache_misses": len(report["cache_misses"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
