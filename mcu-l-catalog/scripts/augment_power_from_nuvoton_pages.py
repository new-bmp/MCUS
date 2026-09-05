#!/usr/bin/env python3
"""Import explicit Nuvoton dynamic-current specifications from cached pages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


POWER_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[µμu]A|mA)\s*/\s*MHz", re.I)
PART_RE = re.compile(r"\bpartNo\s*:\s*['\"]([^'\"]*)", re.I)
FREQ_RE = re.compile(r"(?:up\s+to|at|@|maximum(?:\s+of)?)\s*(\d+(?:\.\d+)?)\s*MHz", re.I)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def page_text(payload: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", payload)).replace("\xa0", " ").split())


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--pages-dir", type=Path, default=root / "cache" / "nuvoton-official" / "pages")
    args = parser.parse_args()
    rows = read_csv(args.data_dir / "device-variants.csv")
    fields = list(rows[0].keys())
    by_model = {row.get("device_name", "").upper(): row for row in rows if row.get("manufacturer") == "Nuvoton"}
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "pages_scanned": 0,
        "models_matched": 0,
        "features_added": 0,
        "evidence": [],
    }
    for path in sorted(args.pages_dir.glob("*.html")):
        report["pages_scanned"] += 1
        payload = path.read_text(encoding="utf-8", errors="ignore")
        match = PART_RE.search(payload)
        if not match:
            continue
        model = html.unescape(match.group(1)).strip().upper()
        row = by_model.get(model)
        if row is None:
            continue
        text = page_text(payload)
        values = []
        for current in POWER_RE.finditer(text):
            value = current.group("value")
            unit = current.group("unit").replace("μ", "u").replace("µ", "u")
            context = text[max(0, current.start() - 140) : min(len(text), current.end() + 140)]
            frequency = FREQ_RE.search(context)
            values.append((value, unit, frequency.group(1) if frequency else ""))
        unique = list(dict.fromkeys(values))
        if not unique:
            continue
        report["models_matched"] += 1
        try:
            features = json.loads(row.get("features_json") or "[]")
        except json.JSONDecodeError:
            features = []
        # Re-emit this source's records so wording changes cannot leave stale
        # qualifiers (for example, "maximum" being mistaken for a maximum
        # measurement) in the derived power classification.
        features = [
            item for item in features
            if not (
                item.get("source_kind") == "nuvoton_official_product_page"
            )
        ]
        source_id = f"nuvoton:product-page:{hashlib.sha1(path.name.encode()).hexdigest()[:16]}"
        added_for_model = 0
        for value, unit, frequency in unique:
            suffix = f", clock limit {frequency} MHz" if frequency else ""
            label = (
                f"Specified dynamic active current density: {value} {unit}/MHz{suffix}; "
                "official Nuvoton product page; density specification only (not a whole-chip current test)"
            )
            item = {
                "type": "PowerOther",
                "name": label,
                "source_kind": "nuvoton_official_product_page",
                "source_document_id": source_id,
                "verification_status": "manufacturer_product_page",
                "category": "power",
            }
            if not any(existing.get("name") == label for existing in features):
                features.append(item)
                added_for_model += 1
        row["features_json"] = json.dumps(features, ensure_ascii=False)
        if added_for_model:
            report["features_added"] += added_for_model
            report["evidence"].append({"model": model, "page": path.name, "values": unique, "features_added": added_for_model})

    write_csv(args.data_dir / "device-variants.csv", fields, rows)
    (args.data_dir / "nuvoton-power-augmentation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pages_scanned": report["pages_scanned"], "models_matched": report["models_matched"], "features_added": report["features_added"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
