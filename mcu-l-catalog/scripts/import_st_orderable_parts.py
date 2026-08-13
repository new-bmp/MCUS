#!/usr/bin/env python3
"""Collect complete STM32 purchasing codes from official ST product pages."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


USER_AGENT = "MCU-L-Catalog/0.1 (+official product-page importer)"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


class MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.keywords: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        values = {key.lower(): value or "" for key, value in attrs}
        if values.get("name", "").lower() == "keywords":
            self.keywords.append(values.get("content", ""))


def fetch_text(url: str, cache_path: Path, *, refresh: bool, timeout: float, retries: int = 3) -> str:
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8", errors="replace")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            text = payload.decode("utf-8", errors="replace")
            cache_path.write_text(text, encoding="utf-8")
            return text
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error


def official_codes(html: str, device_name: str) -> list[str]:
    parser = MetaParser()
    parser.feed(html)
    haystack = ",".join(parser.keywords)
    candidates = set(re.findall(r"STM32[A-Z0-9]+", haystack.upper()))
    prefix = device_name.upper()
    return sorted(
        value
        for value in candidates
        if value.startswith(prefix) and len(value) > len(prefix)
    )


def split_suffix(device_name: str, part_number: str) -> dict[str, str]:
    suffix = part_number[len(device_name):]
    packing_code = "TR" if suffix.endswith("TR") else ""
    core = suffix[:-2] if packing_code else suffix
    return {
        "manufacturer_suffix": suffix,
        "package_code": core[:1],
        "temperature_grade_code": core[1:],
        "packing_code": packing_code,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=Path, default=root / "data" / "device-variants.csv")
    parser.add_argument("--output", type=Path, default=root / "data" / "orderable-parts.csv")
    parser.add_argument("--report", type=Path, default=root / "data" / "st-orderable-import-report.csv")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "st-product-pages")
    parser.add_argument("--series-prefix", default="STM32F1")
    parser.add_argument("--product-line")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    observed_at = utc_now()
    devices = [
        row
        for row in read_csv(args.devices)
        if row["manufacturer"] == "STMicroelectronics"
        and row["series"].upper().startswith(args.series_prefix.upper())
        and (not args.product_line or row["product_line"].upper() == args.product_line.upper())
        and re.fullmatch(r"STM32[A-Z0-9]+", row["device_name"].upper())
    ]

    def fetch_device(device: dict[str, str]) -> tuple[dict[str, str], str, list[str]]:
        name = device["device_name"].upper()
        url = f"https://www.st.com/en/microcontrollers-microprocessors/{name.lower()}.html"
        html = fetch_text(
            url,
            args.cache_dir / f"{name.lower()}.html",
            refresh=args.refresh,
            timeout=args.timeout,
        )
        return device, url, official_codes(html, name)

    rows: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch_device, device): device for device in devices}
        for future in as_completed(futures):
            device = futures[future]
            name = device["device_name"].upper()
            url = f"https://www.st.com/en/microcontrollers-microprocessors/{name.lower()}.html"
            try:
                _, url, codes = future.result()
                for code in codes:
                    suffix = split_suffix(name, code)
                    rows.append(
                        {
                            "orderable_part_id": f"st::{code.lower()}",
                            "device_id": device["device_id"],
                            "manufacturer": "STMicroelectronics",
                            "family": device["family"],
                            "series": device["series"],
                            "product_line": device["product_line"],
                            "device_name": name,
                            "part_number": code,
                            **suffix,
                            "package_name": "",
                            "temperature_range": "",
                            "packing_form": "tape_and_reel" if suffix["packing_code"] == "TR" else "tray_or_unspecified",
                            "lifecycle": "unknown",
                            "source_id": f"st-product-page:{name.lower()}",
                            "source_url": url,
                            "observed_at": observed_at,
                            "verification_status": "manufacturer_product_page",
                            "decode_status": "code_positions_parsed; descriptions_require_device_datasheet",
                        }
                    )
                report.append(
                    {
                        "device_name": name,
                        "source_url": url,
                        "status": "ok" if codes else "no_orderable_codes_found",
                        "orderable_part_count": len(codes),
                        "error": "",
                    }
                )
            except Exception as exc:
                report.append(
                    {
                        "device_name": name,
                        "source_url": url,
                        "status": "error",
                        "orderable_part_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    rows.sort(key=lambda item: item["part_number"])
    report.sort(key=lambda item: item["device_name"])
    fields = [
        "orderable_part_id", "device_id", "manufacturer", "family", "series",
        "product_line", "device_name", "part_number", "manufacturer_suffix",
        "package_code", "temperature_grade_code", "packing_code", "package_name",
        "temperature_range", "packing_form", "lifecycle", "source_id", "source_url",
        "observed_at", "verification_status", "decode_status",
    ]
    report_fields = ["device_name", "source_url", "status", "orderable_part_count", "error"]
    write_csv(args.output, fields, rows)
    write_csv(args.report, report_fields, report)

    sources_path = args.output.parent / "sources.csv"
    source_fields = [
        "source_id", "source_type", "publisher", "title", "url", "version",
        "observed_at", "verification_scope",
    ]
    existing_sources = read_csv(sources_path) if sources_path.exists() else []
    source_map = {row["source_id"]: row for row in existing_sources}
    for item in report:
        if item["status"] == "error":
            continue
        name = item["device_name"]
        source_map[f"st-product-page:{name.lower()}"] = {
            "source_id": f"st-product-page:{name.lower()}",
            "source_type": "manufacturer_product_page",
            "publisher": "STMicroelectronics",
            "title": f"{name} product page",
            "url": item["source_url"],
            "version": "live_web_page",
            "observed_at": observed_at,
            "verification_scope": "complete orderable codes explicitly listed in the page metadata",
        }
    write_csv(sources_path, source_fields, sorted(source_map.values(), key=lambda row: row["source_id"]))

    coverage_path = args.output.parent / "coverage-manifest.csv"
    if coverage_path.exists():
        coverage = read_csv(coverage_path)
        for item in coverage:
            if item.get("manufacturer") == "STMicroelectronics":
                item["orderable_part_count"] = str(len(rows))
                item["orderable_coverage_status"] = f"partial_official_pages:{args.series_prefix}"
                item["notes"] = (
                    f"{item.get('notes', '')} Official product-page ordering codes imported for "
                    f"{args.series_prefix}; other series remain pending."
                ).strip()
        if coverage:
            write_csv(coverage_path, list(coverage[0].keys()), coverage)

    print(
        json.dumps(
            {
                "devices_checked": len(devices),
                "orderable_parts": len(rows),
                "pages_with_no_codes": sum(row["status"] == "no_orderable_codes_found" for row in report),
                "errors": sum(row["status"] == "error" for row in report),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 1 if any(row["status"] == "error" for row in report) else 0


if __name__ == "__main__":
    sys.exit(main())
