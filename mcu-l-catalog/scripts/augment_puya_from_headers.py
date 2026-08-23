#!/usr/bin/env python3
"""Enrich Puya records from exact-device CMSIS headers in official DFPs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


DEVICE_FIELDS = [
    "device_id", "product_line_id", "manufacturer", "product_type",
    "architecture_class", "family", "series", "product_line", "device_name",
    "generic_device_name", "manufacturer_variant_code", "processor_cores",
    "max_clock_hz", "flash_bytes", "ram_bytes", "package_types", "pin_counts",
    "memory_regions_json", "features_json", "documents_json", "svd_files",
    "lifecycle", "source_id", "source_url", "source_version", "observed_at",
    "verification_status",
]
SOURCE_FIELDS = [
    "source_id", "source_type", "publisher", "title", "url", "version",
    "observed_at", "verification_scope",
]
PROVENANCE_FIELDS = [
    "record_type", "record_id", "field_name", "source_id", "source_url",
    "source_path", "source_value_json", "observed_at", "verification_status",
]
MANIFEST_FIELDS = ["source_id", "pack_url", "cache_path", "sha256", "bytes", "exact_headers"]
UNMATCHED_FIELDS = ["device_id", "device_name", "reason", "candidate_headers"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def append_semicolon(value: str, addition: str) -> str:
    values = [item for item in value.split(";") if item]
    if addition not in values:
        values.append(addition)
    return ";".join(values)


def pack_url(source: dict[str, str]) -> str:
    parsed = urllib.parse.urlparse(source["url"])
    filename = PurePosixPath(parsed.path).name
    stem = filename[:-5] if filename.lower().endswith(".pdsc") else filename
    return urllib.parse.urlunparse(
        parsed._replace(path=str(PurePosixPath(parsed.path).parent / f"{stem}.{source['version']}.pack"))
    )


def fetch(url: str, path: Path, *, refresh: bool, timeout: float, proxy: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.puyasemi.com":
        raise ValueError(f"unapproved Puya pack URL: {url}")
    if path.exists() and not refresh:
        return path.read_bytes()
    handlers: list[Any] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, headers={"User-Agent": "MCU-L-Catalog/0.5 (+official device-header importer)"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return payload
        except OSError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.75 * (2**attempt))
    assert last_error is not None
    raise last_error


INSTANCE = re.compile(
    r"^\s*#define\s+"
    r"(?P<name>(?:(?:ADC|DAC|COMP|OPAMP|USART|UART|LPUART|SPI|I2C|I3C|CAN|FDCAN|"
    r"TIM|LPTIM|HRTIM|DMA|BDMA|GPDMA)\d+|RNG\d*|RTC\d*|IWDG\d*|WWDG\d*))\s+"
    r"\(\([^\n]*?\*\)\s*(?P<base>[A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.MULTILINE,
)


def parse_header(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8", errors="replace")
    by_base: dict[str, set[str]] = defaultdict(set)
    for match in INSTANCE.finditer(text):
        by_base[match.group("base")].add(match.group("name"))

    grouped: dict[str, set[str]] = defaultdict(set)
    for base, names in by_base.items():
        # USART/UART aliases sometimes point at the same physical base.  Keep
        # it once and prefer the more capable USART label.
        if any(name.startswith("USART") for name in names):
            grouped["USART"].add(sorted(name for name in names if name.startswith("USART"))[0])
            continue
        name = sorted(names)[0]
        if name.startswith(("LPUART", "UART")):
            grouped["UART"].add(name)
        elif name.startswith(("LPTIM", "HRTIM", "TIM")):
            grouped["Timer"].add(name)
        elif name.startswith(("FDCAN", "CAN")):
            grouped["CAN"].add(name)
        elif name.startswith(("GPDMA", "BDMA", "DMA")):
            grouped["DMA"].add(name)
        elif name.startswith("ADC"):
            grouped["ADCUnits"].add(name)
        elif name.startswith("DAC"):
            grouped["DAC"].add(name)
        elif name.startswith("COMP"):
            grouped["COMP"].add(name)
        elif name.startswith("OPAMP"):
            grouped["OPAMP"].add(name)
        elif name.startswith("SPI"):
            grouped["SPI"].add(name)
        elif name.startswith("I2C"):
            grouped["I2C"].add(name)
        elif name.startswith("I3C"):
            grouped["I3C"].add(name)
        elif name.startswith("RNG"):
            grouped["RNG"].add(name)
        elif name.startswith("RTC"):
            grouped["RTC"].add(name)
        elif name.startswith(("IWDG", "WWDG")):
            grouped["WDT"].add(name)

    exact_types = (
        "Timer", "ADCUnits", "DAC", "SPI", "I2C", "I3C", "USART", "UART",
        "CAN", "DMA", "RNG", "RTC", "WDT", "COMP", "OPAMP",
    )
    return [{
        "type": feature_type,
        "name": ", ".join(sorted(grouped.get(feature_type, set()))) or "Official header exact instance count",
        "count": str(len(grouped.get(feature_type, set()))),
        "source_kind": "puya_device_header",
    } for feature_type in exact_types]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-pack", type=Path, default=root / "data/vendor-packs/puya")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache/puya/packs")
    parser.add_argument("--proxy", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    vendor_pack = args.vendor_pack.resolve()
    cache_dir = args.cache_dir.resolve()
    observed_at = utc_now()

    devices = read_csv(vendor_pack / "device-variants.csv")
    sources = read_csv(vendor_pack / "sources.csv")
    cmsis_sources = [row for row in sources if row.get("source_id", "").startswith("cmsis:Puya.")]
    header_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest: list[dict[str, Any]] = []
    source_additions: dict[str, dict[str, str]] = {}
    errors: list[dict[str, str]] = []

    for source in cmsis_sources:
        url = pack_url(source)
        cache_path = cache_dir / PurePosixPath(urllib.parse.urlparse(url).path).name
        try:
            payload = fetch(url, cache_path, refresh=args.refresh, timeout=args.timeout, proxy=args.proxy)
            source_id = "puya-device-header:" + source["source_id"].removeprefix("cmsis:Puya.")
            exact_count = 0
            with zipfile.ZipFile(cache_path) as archive:
                for name in archive.namelist():
                    if not name.lower().endswith(".h"):
                        continue
                    stem = PurePosixPath(name).stem.lower().replace(" ", "")
                    if not re.fullmatch(r"py32[a-z0-9]+", stem):
                        continue
                    record = {
                        "source_id": source_id, "source_url": url,
                        "source_version": source["version"], "archive_path": name,
                        "features": parse_header(archive.read(name)),
                    }
                    header_index[stem].append(record)
                    exact_count += 1
            digest = hashlib.sha256(payload).hexdigest()
            manifest.append({
                "source_id": source_id, "pack_url": url, "cache_path": str(cache_path),
                "sha256": digest, "bytes": len(payload), "exact_headers": exact_count,
            })
            source_additions[source_id] = {
                "source_id": source_id, "source_type": "manufacturer_device_header",
                "publisher": "Puya", "title": f"{source['title']} exact-device CMSIS headers",
                "url": url, "version": source["version"], "observed_at": observed_at,
                "verification_scope": (
                    "Peripheral instances explicitly defined by the official header whose filename exactly "
                    "matches the PDSC device name. Package pin counts and ADC external pins are not inferred."
                ),
            }
        except Exception as exc:
            errors.append({"source_id": source["source_id"], "error": str(exc)})

    provenance: list[dict[str, Any]] = []
    unmatched: list[dict[str, str]] = []
    matches = Counter()
    for row in devices:
        key = row["device_name"].lower().replace(" ", "")
        candidates = header_index.get(key, [])
        if len(candidates) != 1:
            unmatched.append({
                "device_id": row["device_id"], "device_name": row["device_name"],
                "reason": "no_exact_header_match" if not candidates else "duplicate_exact_header_match",
                "candidate_headers": ";".join(item["archive_path"] for item in candidates),
            })
            continue
        source = candidates[0]
        existing = [
            feature for feature in json.loads(row.get("features_json") or "[]")
            if feature.get("source_kind") != "puya_device_header"
        ]
        row["features_json"] = json.dumps(existing + source["features"], ensure_ascii=False, sort_keys=True)
        row["source_id"] = append_semicolon(row.get("source_id", ""), source["source_id"])
        row["source_version"] = append_semicolon(row.get("source_version", ""), source["source_version"])
        row["verification_status"] = "multi_source_manufacturer_device_header"
        matches[source["source_id"]] += 1
        provenance.append({
            "record_type": "device", "record_id": row["device_id"], "field_name": "features_json",
            "source_id": source["source_id"], "source_url": source["source_url"],
            "source_path": f"{source['source_url']}!/{source['archive_path']}",
            "source_value_json": json.dumps(source["features"], ensure_ascii=False, sort_keys=True),
            "observed_at": observed_at, "verification_status": "manufacturer_device_header",
        })

    retained = {row["source_id"]: row for row in sources if not row.get("source_id", "").startswith("puya-device-header:")}
    retained.update(source_additions)
    write_csv(vendor_pack / "device-variants.csv", DEVICE_FIELDS, devices)
    write_csv(vendor_pack / "sources.csv", SOURCE_FIELDS, sorted(retained.values(), key=lambda row: row["source_id"]))
    write_csv(vendor_pack / "header-field-provenance.csv", PROVENANCE_FIELDS, provenance)
    write_csv(vendor_pack / "header-unmatched.csv", UNMATCHED_FIELDS, unmatched)
    write_csv(vendor_pack / "header-source-manifest.csv", MANIFEST_FIELDS, manifest)
    report = {
        "status": "ok" if not errors else "partial", "official_packs": len(cmsis_sources),
        "pack_errors": errors, "input_devices": len(devices),
        "matched_devices": len(devices) - len(unmatched), "unmatched_devices": len(unmatched),
        "match_counts": dict(sorted(matches.items())), "provenance_records": len(provenance),
        "observed_at": observed_at,
        "accuracy_policy": [
            "The normalized header filename must exactly equal the PDSC device name.",
            "Counts use concrete peripheral instance macros and deduplicate aliases by physical base macro.",
            "A missing instance in the exact device header is recorded as zero.",
            "GPIO count, ADC external pins, and package data remain unknown without an official pinout source.",
        ],
    }
    (vendor_pack / "header-import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
