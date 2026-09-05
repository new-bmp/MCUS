#!/usr/bin/env python3
"""Collect release-time and listed-price metadata from official selector snapshots.

The source selector calls its date ``releaseTime`` and its price ``price``.  The
date is retained as a release-time observation; the price is deliberately marked
as a listed reference price because the selector does not claim it is a launch
price.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "device_id",
    "release_date",
    "release_year",
    "release_quarter",
    "release_date_source",
    "launch_price",
    "launch_price_currency",
    "launch_price_status",
    "launch_price_source",
    "observed_at",
    "verification_status",
]

SELECTOR_URL = "https://products.espressif.com/#/product-comparison"


def parse_date(raw: object) -> tuple[str, int | None, str | None]:
    value = str(raw or "").strip()
    match = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", value)
    if not match:
        return "", None, None
    year, month, day = (int(part) for part in match.groups())
    try:
        date = datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return "", None, None
    return date.date().isoformat(), year, f"Q{(month - 1) // 3 + 1}"


def parse_price(raw: object) -> tuple[float | None, str]:
    value = str(raw or "").strip()
    match = re.match(r"^([A-Za-z]{3})\s*([0-9]+(?:\.[0-9]+)?)$", value)
    if not match:
        return None, ""
    return float(match.group(2)), match.group(1).upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "cache" / "espressif-products-en.json",
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=root / "data" / "combined" / "device-variants.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "combined" / "release-metadata.csv",
    )
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    selector_rows = {
        str(row.get("mpn") or "").strip().upper(): row
        for row in source.get("results", [])
        if str(row.get("mpn") or "").strip()
    }
    variants = []
    with args.variants.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("manufacturer") == "Espressif":
                variants.append(row)

    observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for variant in variants:
        mpn = str(variant.get("device_name") or "").strip().upper()
        selector = selector_rows.get(mpn)
        if not selector:
            continue
        release_date, release_year, release_quarter = parse_date(selector.get("releaseTime"))
        price, currency = parse_price(selector.get("price"))
        if not release_date and price is None:
            continue
        rows.append(
            {
                "device_id": variant["device_id"],
                "release_date": release_date,
                "release_year": release_year or "",
                "release_quarter": release_quarter or "",
                "release_date_source": SELECTOR_URL if release_date else "",
                "launch_price": price if price is not None else "",
                "launch_price_currency": currency,
                "launch_price_status": "official_product_selector_listed_price" if price is not None else "",
                "launch_price_source": SELECTOR_URL if price is not None else "",
                "observed_at": observed_at,
                "verification_status": "manufacturer_product_selector_snapshot",
            }
        )

    rows.sort(key=lambda row: row["device_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"source": str(args.source), "matched_devices": len(rows), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
