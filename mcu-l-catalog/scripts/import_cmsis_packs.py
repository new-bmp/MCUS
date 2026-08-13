#!/usr/bin/env python3
"""Import real MCU device names and metadata from the CMSIS-Pack index.

The importer intentionally stops at the device/variant layer. Complete
orderable part numbers are collected by manufacturer-specific adapters, never
constructed by combining possible suffix characters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


INDEX_URL = "https://www.keil.com/pack/index.pidx"
USER_AGENT = "MCU-L-Catalog/0.1 (+catalog metadata importer)"
PACKAGE_FEATURES = {
    "BGA", "CSP", "DFN", "DIP", "LGA", "QFN", "QFP", "SO", "SON",
    "SOP", "SSOP", "TQFP", "TSSOP", "UFBGA", "UFQFPN", "VFBGA",
    "WLCSP",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "unknown"


def parse_int(value: str | None) -> int | None:
    if not value:
        return None
    value = value.strip()
    try:
        return int(value, 0)
    except ValueError:
        return None


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fetch_bytes(
    url: str,
    cache_path: Path,
    *,
    refresh: bool,
    timeout: float,
    retries: int = 3,
) -> bytes:
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            cache_path.write_bytes(payload)
            return payload
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error


@dataclass
class Context:
    processors: dict[str, dict[str, str]] = field(default_factory=dict)
    memories: dict[str, dict[str, str]] = field(default_factory=dict)
    features: dict[str, dict[str, str]] = field(default_factory=dict)
    books: list[dict[str, str]] = field(default_factory=list)
    debug: dict[str, str] = field(default_factory=dict)

    def copy(self) -> "Context":
        return Context(
            processors={key: dict(value) for key, value in self.processors.items()},
            memories={key: dict(value) for key, value in self.memories.items()},
            features={key: dict(value) for key, value in self.features.items()},
            books=[dict(value) for value in self.books],
            debug=dict(self.debug),
        )


def merge_direct_metadata(node: ET.Element, inherited: Context) -> Context:
    result = inherited.copy()
    for child in node:
        tag = local_name(child.tag)
        attrs = dict(child.attrib)
        if tag == "processor":
            key = attrs.get("Pname") or "__default__"
            prior = result.processors.get(key, {})
            result.processors[key] = {**prior, **attrs}
        elif tag == "memory":
            key = attrs.get("id") or attrs.get("Dname") or attrs.get("name")
            if not key:
                key = hashlib.sha1(json.dumps(attrs, sort_keys=True).encode()).hexdigest()[:12]
            prior = result.memories.get(key, {})
            result.memories[key] = {**prior, **attrs}
        elif tag == "feature":
            key = "|".join((attrs.get("type", ""), attrs.get("name", "")))
            prior = result.features.get(key, {})
            result.features[key] = {**prior, **attrs}
        elif tag == "book":
            if attrs and attrs not in result.books:
                result.books.append(attrs)
        elif tag == "debug":
            result.debug.update(attrs)
    return result


def classify_memory(memories: Iterable[dict[str, str]]) -> tuple[int | None, int | None]:
    flash_total = 0
    ram_total = 0
    flash_found = False
    ram_found = False
    for memory in memories:
        identity = " ".join(
            (memory.get("id", ""), memory.get("Dname", ""), memory.get("name", ""))
        ).upper()
        size = parse_int(memory.get("size"))
        if size is None:
            continue
        access = memory.get("access", "").lower()
        is_default = memory.get("default") == "1" or memory.get("startup") == "1"
        is_program_memory = (
            "IROM" in identity
            or ("ROM" in identity and "IRAM" not in identity)
            or "FLASH" in identity
        )
        # Configuration/OTP/boot metadata regions are often also named FLASH.
        # Count executable/default program regions, while preserving every raw
        # region in memory_regions_json for audit.
        if is_program_memory and (
            "x" in access or is_default or not access
        ) and not any(token in identity for token in ("CONFIG", "USERCFG", "BOOTCFG", "OTP", "FUSE")):
            flash_total += size
            flash_found = True
        elif "IRAM" in identity or "RAM" in identity:
            ram_total += size
            ram_found = True
    return (flash_total if flash_found else None, ram_total if ram_found else None)


def product_type_for(series: str, device_name: str) -> str:
    key = f"{series} {device_name}".upper()
    if any(token in key for token in ("STM32WB", "STM32WL", "STM32WBA")):
        return "wireless_mcu"
    if any(token in key for token in ("STM32L", "STM32U")):
        return "low_power_mcu"
    if any(token in key for token in ("STM32H", "STM32N")):
        return "high_performance_mcu"
    return "general_purpose_mcu"


def normalize_manufacturer(value: str, package_vendor: str) -> str:
    value = value.split(":", 1)[0].strip() if value else ""
    aliases = {
        "STMicroelectronics": "STMicroelectronics",
        "STMicroelectronics International N.V.": "STMicroelectronics",
        "TexasInstruments": "Texas Instruments",
        "Texas Instruments": "Texas Instruments",
    }
    return aliases.get(value, value or package_vendor)


def derive_family(device_name: str, family_name: str) -> str:
    upper = device_name.upper()
    known_prefixes = (
        "STM32", "STM8", "EFM32", "EFR32", "LPC", "MCX", "KINETIS",
        "MSPM0", "TM4C", "MAX32", "NRF", "GD32", "APM32", "AT32",
    )
    for prefix in known_prefixes:
        if upper.startswith(prefix):
            return prefix.replace("NRF", "nRF")
    return family_name.removesuffix(" Series").strip() or device_name


def direct_children(node: ET.Element, wanted: str) -> list[ET.Element]:
    return [child for child in node if local_name(child.tag) == wanted]


def parse_pdsc(payload: bytes, pack: dict[str, str], observed_at: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = ET.fromstring(payload)
    package_vendor = root.findtext("vendor") or pack["pack_vendor"]
    package_name = root.findtext("name") or pack["pack_name"]
    source_id = f"cmsis:{package_vendor}.{package_name}@{pack['pack_version']}"
    devices_root = next((node for node in root if local_name(node.tag) == "devices"), None)
    if devices_root is None:
        return [], {
            "source_id": source_id,
            "source_type": "cmsis_pdsc",
            "publisher": package_vendor,
            "title": f"{package_vendor}.{package_name}",
            "url": pack["pdsc_url"],
            "version": pack["pack_version"],
            "observed_at": observed_at,
            "verification_scope": "device names and metadata present in the PDSC; not purchasing-code completeness",
        }

    rows: list[dict[str, Any]] = []

    def emit(
        node: ET.Element,
        context: Context,
        *,
        family_name: str,
        manufacturer: str,
        subfamilies: list[str],
        generic_device_name: str,
    ) -> None:
        node_tag = local_name(node.tag)
        device_name = node.attrib.get("Dvariant") if node_tag == "variant" else node.attrib.get("Dname")
        if not device_name:
            return

        processors = list(context.processors.values())
        memories = list(context.memories.values())
        features = list(context.features.values())
        flash_bytes, ram_bytes = classify_memory(memories)
        cores = sorted({item.get("Dcore", "") for item in processors if item.get("Dcore")})
        clocks = [parse_int(item.get("Dclock")) for item in processors]
        clocks = [item for item in clocks if item is not None]

        package_types: set[str] = set()
        pin_counts: set[int] = set()
        for feature in features:
            feature_type = feature.get("type", "").upper()
            if feature_type in PACKAGE_FEATURES:
                package_types.add(feature_type)
                pins = parse_int(feature.get("n"))
                if pins is not None:
                    pin_counts.add(pins)

        series = family_name.removesuffix(" Series").strip() or family_name
        if subfamilies:
            product_line = subfamilies[-1]
        elif node_tag == "variant" and generic_device_name:
            product_line = generic_device_name
        else:
            # When a PDSC places selectable devices directly under a family,
            # the family is the only source-backed grouping. Treating every
            # device name as its own product line would fabricate hierarchy.
            product_line = family_name.removesuffix(" Series").strip() or device_name
        catalog_family = derive_family(device_name, family_name)
        variant_code = device_name[len(product_line):] if device_name.startswith(product_line) else ""
        svd_files = sorted(
            {
                attrs.get("svd", "")
                for attrs in (context.debug, *processors)
                if attrs.get("svd")
            }
        )
        device_id = f"{slug(manufacturer)}::{slug(device_name)}"
        product_line_id = "::".join(
            (slug(manufacturer), slug(catalog_family), slug(series), slug(product_line))
        )
        rows.append(
            {
                "device_id": device_id,
                "product_line_id": product_line_id,
                "manufacturer": manufacturer,
                "product_type": product_type_for(series, device_name),
                "architecture_class": "+".join(cores) if cores else "unknown",
                "family": catalog_family,
                "series": series,
                "product_line": product_line,
                "device_name": device_name,
                "generic_device_name": generic_device_name or device_name,
                "manufacturer_variant_code": variant_code,
                "processor_cores": json.dumps(processors, ensure_ascii=False, sort_keys=True),
                "max_clock_hz": max(clocks) if clocks else "",
                "flash_bytes": flash_bytes if flash_bytes is not None else "",
                "ram_bytes": ram_bytes if ram_bytes is not None else "",
                "package_types": ";".join(sorted(package_types)),
                "pin_counts": ";".join(str(value) for value in sorted(pin_counts)),
                "memory_regions_json": json.dumps(memories, ensure_ascii=False, sort_keys=True),
                "features_json": json.dumps(features, ensure_ascii=False, sort_keys=True),
                "documents_json": json.dumps(context.books, ensure_ascii=False, sort_keys=True),
                "svd_files": ";".join(svd_files),
                "lifecycle": "unknown",
                "source_id": source_id,
                "source_url": pack["pdsc_url"],
                "source_version": pack["pack_version"],
                "observed_at": observed_at,
                "verification_status": "cmsis_pack_metadata",
            }
        )

    def walk(
        node: ET.Element,
        inherited: Context,
        *,
        family_name: str,
        manufacturer: str,
        subfamilies: list[str],
        generic_device_name: str = "",
    ) -> None:
        context = merge_direct_metadata(node, inherited)
        tag = local_name(node.tag)
        next_subfamilies = subfamilies
        next_generic = generic_device_name
        if tag == "subFamily":
            name = node.attrib.get("DsubFamily", "").strip()
            next_subfamilies = [*subfamilies, name] if name else subfamilies
        elif tag == "device":
            next_generic = node.attrib.get("Dname", "")

        nested = [
            child
            for child in node
            if local_name(child.tag) in {"subFamily", "device", "variant"}
        ]
        selectable_children = [child for child in nested if local_name(child.tag) == "variant"]
        if tag in {"device", "variant"} and (tag == "variant" or not selectable_children):
            emit(
                node,
                context,
                family_name=family_name,
                manufacturer=manufacturer,
                subfamilies=next_subfamilies,
                generic_device_name=next_generic,
            )
        for child in nested:
            walk(
                child,
                context,
                family_name=family_name,
                manufacturer=manufacturer,
                subfamilies=next_subfamilies,
                generic_device_name=next_generic,
            )

    for family in direct_children(devices_root, "family"):
        family_name = family.attrib.get("Dfamily", "").strip()
        manufacturer = normalize_manufacturer(family.attrib.get("Dvendor", ""), package_vendor)
        walk(
            family,
            Context(),
            family_name=family_name,
            manufacturer=manufacturer,
            subfamilies=[],
        )

    source = {
        "source_id": source_id,
        "source_type": "cmsis_pdsc",
        "publisher": package_vendor,
        "title": f"{package_vendor}.{package_name}",
        "url": pack["pdsc_url"],
        "version": pack["pack_version"],
        "observed_at": observed_at,
        "verification_scope": "device names and metadata present in the PDSC; not purchasing-code completeness",
    }
    return rows, source


def make_pdsc_url(base_url: str, vendor: str, name: str) -> str:
    if base_url.startswith("http://"):
        base_url = "https://" + base_url[len("http://"):]
    base_url = base_url if base_url.endswith("/") else base_url + "/"
    filename = f"{vendor}.{name}.pdsc"
    return urllib.parse.urljoin(base_url, filename)


def scope_matches(scope: str, row: dict[str, str]) -> bool:
    name = row["pack_name"].lower()
    if not name.endswith("_dfp"):
        return False
    if scope == "all-dfp":
        return True
    if scope == "stm32":
        return "stm32" in name and "stm32mp" not in name
    raise ValueError(f"unsupported scope: {scope}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-url", default=INDEX_URL)
    parser.add_argument("--scope", choices=("stm32", "all-dfp"), default="stm32")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[1] / "cache")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--include-deprecated", action="store_true")
    parser.add_argument("--max-packs", type=int)
    parser.add_argument(
        "--vendor",
        action="append",
        default=[],
        help="Limit selection to one or more exact pack-vendor names; repeat as needed.",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Reject selected PDSC URLs outside these HTTPS hosts; repeat as needed.",
    )
    args = parser.parse_args()

    observed_at = utc_now()
    index_payload = fetch_bytes(
        args.index_url,
        args.cache_dir / "index.pidx",
        refresh=args.refresh,
        timeout=args.timeout,
    )
    index_root = ET.fromstring(index_payload)
    index_timestamp = index_root.findtext("timestamp") or ""

    all_dfp_rows: list[dict[str, str]] = []
    selected: list[dict[str, str]] = []
    for pdsc in index_root.findall(".//pdsc"):
        vendor = pdsc.attrib.get("vendor", "")
        name = pdsc.attrib.get("name", "")
        if not name.lower().endswith("_dfp"):
            continue
        row = {
            "pack_vendor": vendor,
            "pack_name": name,
            "pack_version": pdsc.attrib.get("version", ""),
            "pdsc_url": make_pdsc_url(pdsc.attrib.get("url", ""), vendor, name),
            "deprecated": pdsc.attrib.get("deprecated", ""),
            "replacement": pdsc.attrib.get("replacement", ""),
            "index_timestamp": index_timestamp,
            "selected_for_import": "",
            "fetch_status": "not_selected",
            "device_count": "",
            "error": "",
        }
        all_dfp_rows.append(row)
        vendor_matches = not args.vendor or vendor.lower() in {value.lower() for value in args.vendor}
        if (
            scope_matches(args.scope, row)
            and vendor_matches
            and (args.include_deprecated or not row["deprecated"])
        ):
            row["selected_for_import"] = "yes"
            row["fetch_status"] = "pending"
            selected.append(row)

    selected.sort(key=lambda item: (item["pack_vendor"].lower(), item["pack_name"].lower()))
    if args.max_packs is not None:
        selected = selected[: args.max_packs]
        selected_ids = {(item["pack_vendor"], item["pack_name"]) for item in selected}
        for row in all_dfp_rows:
            if row["selected_for_import"] == "yes" and (row["pack_vendor"], row["pack_name"]) not in selected_ids:
                row["selected_for_import"] = ""
                row["fetch_status"] = "not_selected_by_limit"

    def fetch_pack(pack: dict[str, str]) -> tuple[dict[str, str], bytes]:
        parsed_url = urllib.parse.urlparse(pack["pdsc_url"])
        if args.allowed_host and (
            parsed_url.scheme != "https"
            or (parsed_url.hostname or "").lower() not in {host.lower() for host in args.allowed_host}
        ):
            raise ValueError(f"unapproved official PDSC URL: {pack['pdsc_url']}")
        cache_name = f"{pack['pack_vendor']}.{pack['pack_name']}.pdsc"
        payload = fetch_bytes(
            pack["pdsc_url"],
            args.cache_dir / "pdsc" / cache_name,
            refresh=args.refresh,
            timeout=args.timeout,
        )
        return pack, payload

    device_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch_pack, pack): pack for pack in selected}
        for future in as_completed(futures):
            pack = futures[future]
            try:
                _, payload = future.result()
                parsed_devices, source = parse_pdsc(payload, pack, observed_at)
                device_rows.extend(parsed_devices)
                sources.append(source)
                pack["fetch_status"] = "ok"
                pack["device_count"] = str(len(parsed_devices))
            except Exception as exc:  # keep a complete, reviewable failure manifest
                pack["fetch_status"] = "error"
                pack["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(
                    {
                        "pack_vendor": pack["pack_vendor"],
                        "pack_name": pack["pack_name"],
                        "pdsc_url": pack["pdsc_url"],
                        "error": pack["error"],
                    }
                )

    # A device can occasionally be repeated across transition packs. Preserve one
    # canonical row and make duplicates visible as validation-free source aliases.
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in sorted(device_rows, key=lambda item: (item["manufacturer"], item["device_name"], item["source_id"])):
        existing = deduplicated.get(row["device_id"])
        if existing is None:
            deduplicated[row["device_id"]] = row
        elif row["source_id"] not in existing["source_id"].split(";"):
            existing["source_id"] += ";" + row["source_id"]
            existing["source_url"] += ";" + row["source_url"]
            existing["source_version"] += ";" + row["source_version"]
    device_rows = sorted(deduplicated.values(), key=lambda item: (item["manufacturer"], item["series"], item["product_line"], item["device_name"]))

    product_line_map: dict[str, dict[str, Any]] = {}
    for device in device_rows:
        row = product_line_map.setdefault(
            device["product_line_id"],
            {
                "product_line_id": device["product_line_id"],
                "manufacturer": device["manufacturer"],
                "product_type": device["product_type"],
                "architecture_class": device["architecture_class"],
                "family": device["family"],
                "series": device["series"],
                "product_line": device["product_line"],
                "device_variant_count": 0,
                "source_ids": set(),
                "observed_at": observed_at,
                "verification_status": "cmsis_pack_metadata",
            },
        )
        row["device_variant_count"] += 1
        row["source_ids"].update(device["source_id"].split(";"))
    product_lines: list[dict[str, Any]] = []
    for row in product_line_map.values():
        row["source_ids"] = ";".join(sorted(row["source_ids"]))
        product_lines.append(row)
    product_lines.sort(key=lambda item: (item["manufacturer"], item["family"], item["series"], item["product_line"]))

    manufacturers = sorted({row["manufacturer"] for row in device_rows})
    coverage_rows: list[dict[str, Any]] = []
    for manufacturer in manufacturers:
        scoped_devices = [row for row in device_rows if row["manufacturer"] == manufacturer]
        scoped_lines = [row for row in product_lines if row["manufacturer"] == manufacturer]
        scoped_sources = {source_id for row in scoped_devices for source_id in row["source_id"].split(";")}
        coverage_rows.append(
            {
                "manufacturer": manufacturer,
                "source_scope": args.scope,
                "pack_count": len(scoped_sources),
                "family_count": len({row["family"] for row in scoped_lines}),
                "series_count": len({row["series"] for row in scoped_lines}),
                "product_line_count": len(scoped_lines),
                "device_variant_count": len(scoped_devices),
                "orderable_part_count": 0,
                "last_observed": observed_at,
                "device_coverage_status": "indexed_from_selected_cmsis_packs",
                "orderable_coverage_status": "not_imported",
                "notes": "CMSIS metadata is not proof of complete manufacturer purchasing-code coverage.",
            }
        )

    index_fields = [
        "pack_vendor", "pack_name", "pack_version", "pdsc_url", "deprecated",
        "replacement", "index_timestamp", "selected_for_import", "fetch_status",
        "device_count", "error",
    ]
    product_line_fields = [
        "product_line_id", "manufacturer", "product_type", "architecture_class",
        "family", "series", "product_line", "device_variant_count", "source_ids",
        "observed_at", "verification_status",
    ]
    device_fields = [
        "device_id", "product_line_id", "manufacturer", "product_type",
        "architecture_class", "family", "series", "product_line", "device_name",
        "generic_device_name", "manufacturer_variant_code", "processor_cores",
        "max_clock_hz", "flash_bytes", "ram_bytes", "package_types", "pin_counts",
        "memory_regions_json", "features_json", "documents_json", "svd_files",
        "lifecycle", "source_id", "source_url", "source_version", "observed_at",
        "verification_status",
    ]
    source_fields = [
        "source_id", "source_type", "publisher", "title", "url", "version",
        "observed_at", "verification_scope",
    ]
    coverage_fields = [
        "manufacturer", "source_scope", "pack_count", "family_count", "series_count",
        "product_line_count", "device_variant_count", "orderable_part_count",
        "last_observed", "device_coverage_status", "orderable_coverage_status", "notes",
    ]
    error_fields = ["pack_vendor", "pack_name", "pdsc_url", "error"]

    write_csv(args.output_dir / "cmsis-pack-index.csv", index_fields, all_dfp_rows)
    write_csv(args.output_dir / "product-lines.csv", product_line_fields, product_lines)
    write_csv(args.output_dir / "device-variants.csv", device_fields, device_rows)
    write_csv(args.output_dir / "sources.csv", source_fields, sorted(sources, key=lambda item: item["source_id"]))
    write_csv(args.output_dir / "coverage-manifest.csv", coverage_fields, coverage_rows)
    write_csv(args.output_dir / "import-errors.csv", error_fields, errors)

    print(
        json.dumps(
            {
                "scope": args.scope,
                "selected_packs": len(selected),
                "successful_packs": sum(row["fetch_status"] == "ok" for row in selected),
                "failed_packs": len(errors),
                "product_lines": len(product_lines),
                "device_variants": len(device_rows),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
