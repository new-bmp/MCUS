#!/usr/bin/env python3
"""Augment Puya records from the manufacturer's live product tables.

Puya publish an explicit, complete ordering-code table on each PY32 product
page.  This importer uses those rows as the authority for package options and
the data-sheet URL.  It deliberately does *not* decode an ordering-code
suffix.  A catalog device can receive package options only when all of the
following are explicitly compatible with the official table row: product
line, core, maximum clock, Flash and SRAM.  The result is therefore a set of
officially offered package alternatives for that catalog variant, never a
guessed package for a particular suffix.

Family data sheets are attached only when the same official PDF is linked by
every ordering-code row on that product page.  This makes a product-line PDF
usable for generic CMSIS records without pretending it is a unique SKU page.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = "https://www.puyasemi.com"
SEEDS = (f"{ROOT}/en/", f"{ROOT}/en/py32_series.html")
USER_AGENT = "MCUS/1.4 (+https://github.com/new-bmp/MCUS; Puya official importer)"
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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def cache_name(url: str) -> str:
    tail = urllib.parse.urlsplit(url).path.strip("/").replace("/", "-") or "index"
    tail = re.sub(r"[^a-z0-9-]+", "-", tail.lower()).strip("-")
    return f"{tail}-{hashlib.sha256(url.encode()).hexdigest()[:12]}.html"


def decode_page(payload: bytes) -> str:
    """Decode Puya pages without corrupting Chinese download-path segments."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        # The English pages are otherwise ASCII, but some currently emit the
        # Chinese PDF directory in GB18030.  Decoding it correctly is required
        # for a working, percent-encoded URL after urljoin().
        return payload.decode("gb18030")


def fetch(url: str, cache_dir: Path, timeout: float, refresh: bool) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "www.puyasemi.com":
        raise ValueError(f"unapproved Puya URL: {url}")
    path = cache_dir / cache_name(url)
    if path.exists() and not refresh:
        return decode_page(path.read_bytes())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return decode_page(payload)
        except OSError as exc:  # pragma: no cover - remote availability
            last_error = exc
            if attempt < 2:
                time.sleep(0.75 * (2**attempt))
    assert last_error is not None
    raise last_error


def product_links(payload: str, base_url: str) -> list[str]:
    result: list[str] = []
    for raw in re.findall(r"\bhref\s*=\s*['\"]([^'\"]+)['\"]", payload, re.I):
        url = urllib.parse.urljoin(base_url, html.unescape(raw))
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "www.puyasemi.com":
            continue
        # Keep only the product-family page.  Detail pages such as
        # /en/py32f031/2692.html repeat a single ordering code and are not
        # needed once its parent table is available.
        if not re.fullmatch(r"/en/py32[a-z0-9_-]*\.html", parsed.path, re.I):
            continue
        normalized = urllib.parse.urlunsplit(("https", parsed.netloc, parsed.path, "", ""))
        result.append(normalized)
    return sorted(set(result))


def balanced_array(payload: str, key: str) -> str:
    """Return the JavaScript array following ``key: [`` without eval()."""
    separator = r"=" if key == "defaultData" else r":"
    match = re.search(rf"\b{re.escape(key)}\s*{separator}\s*\[", payload, re.I)
    if not match:
        return ""
    start = payload.find("[", match.start())
    depth = 0
    quote = ""
    escaped = False
    for index in range(start, len(payload)):
        character = payload[index]
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = ""
            continue
        if character in "'\"":
            quote = character
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return payload[start + 1:index]
    return ""


def object_literals(payload: str) -> list[str]:
    result: list[str] = []
    start: int | None = None
    depth = 0
    quote = ""
    escaped = False
    for index, character in enumerate(payload):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = ""
            continue
        if character in "'\"":
            quote = character
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                result.append(payload[start:index + 1])
                start = None
    return result


def js_value(record: str, field: str) -> str:
    # Product-table values use simple quoted JavaScript strings.  We preserve
    # their text rather than evaluating the remote script.
    match = re.search(
        rf"(?:\"{re.escape(field)}\"|\b{re.escape(field)}\b)\s*:\s*(['\"])(.*?)\1",
        record,
        re.S,
    )
    if not match:
        return ""
    return html.unescape(match.group(2).replace("\\/", "/").replace("\\'", "'").replace('\\"', '"')).strip()


def official_url(base_url: str, href: str) -> str:
    """Return an ASCII-safe absolute URL for the official download control."""
    parsed = urllib.parse.urlsplit(urllib.parse.urljoin(base_url, href))
    return urllib.parse.urlunsplit((
        parsed.scheme,
        parsed.netloc,
        urllib.parse.quote(parsed.path, safe="/%:@!$&'()*+,;=-._~"),
        parsed.query,
        parsed.fragment,
    ))


def table_records(payload: str, page_url: str) -> list[dict[str, str]]:
    # Newer product pages call the grid value ``data``; several otherwise
    # identical official pages call it ``defaultData``.
    data_blocks = [balanced_array(payload, "data"), balanced_array(payload, "defaultData")]
    records: list[dict[str, str]] = []
    seen_parts: set[str] = set()
    for literal in object_literals("\n".join(data_blocks)):
        part_html = js_value(literal, "PartNo")
        part_match = re.search(r">\s*(PY32[A-Z0-9-]+)\s*<", part_html, re.I)
        if not part_match:
            continue
        part_number = part_match.group(1).upper()
        if part_number in seen_parts:
            continue
        seen_parts.add(part_number)
        datasheet_html = js_value(literal, "datasheet")
        href = re.search(r"\bhref\s*=\s*['\"]([^'\"]+)", datasheet_html, re.I)
        records.append({
            "part_number": part_number,
            "core": js_value(literal, "Core").upper(),
            "max_clock_mhz": js_value(literal, "MaxCLKMHz"),
            "flash_kb": js_value(literal, "FlashKB"),
            "sram_kb": js_value(literal, "SRAMKB"),
            "package": js_value(literal, "PackageType").upper().replace(" ", ""),
            "datasheet_url": official_url(page_url, html.unescape(href.group(1))) if href else "",
        })
    return records


def as_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalized_core(value: str) -> str:
    """Compare only the explicit core label, not a package suffix or family."""
    value = value.upper().replace("CORTEX-", "").replace("CORTEX", "")
    return re.sub(r"[^A-Z0-9+]", "", value)


def matching_memory_core_rows(device: dict[str, str], records: list[dict[str, str]]) -> list[dict[str, str]]:
    line = device.get("product_line", "").upper()
    flash = as_int(device.get("flash_bytes", ""))
    ram = as_int(device.get("ram_bytes", ""))
    core = normalized_core(device.get("architecture_class", ""))
    if not line or flash is None or ram is None:
        return []
    result = []
    for item in records:
        if not item["part_number"].startswith(line):
            continue
        if as_int(item["flash_kb"]) != flash // 1024 or as_int(item["sram_kb"]) != ram // 1024:
            continue
        item_core = normalized_core(item["core"])
        # M0+ / M4F are vendor shorthands for the architecture stored in the
        # catalog.  This maps only an explicit core label, never a part code.
        if core and item_core and not (core.endswith(item_core.rstrip("F")) or item_core.startswith(core.replace("M", "M"))):
            continue
        result.append(item)
    return result


def matching_rows(device: dict[str, str], records: list[dict[str, str]]) -> list[dict[str, str]]:
    clock = as_int(device.get("max_clock_hz", ""))
    if clock is None:
        return []
    return [
        item for item in matching_memory_core_rows(device, records)
        if as_int(item["max_clock_mhz"]) == clock // 1_000_000
    ]


def update_processor_clock(row: dict[str, str], clock_hz: int) -> None:
    """Keep the core descriptor's explicit Dclock consistent with the row."""
    try:
        processors = json.loads(row.get("processor_cores") or "[]")
    except json.JSONDecodeError:
        return
    if not isinstance(processors, list):
        return
    for processor in processors:
        if isinstance(processor, dict):
            processor["Dclock"] = str(clock_hz)
    row["processor_cores"] = json.dumps(processors, ensure_ascii=False, sort_keys=True)


def add_document(row: dict[str, str], document: dict[str, str]) -> bool:
    try:
        documents = json.loads(row.get("documents_json") or "[]")
    except json.JSONDecodeError:
        documents = []
    if not isinstance(documents, list):
        documents = []
    # Replace only the record owned by this importer.  This repairs an older
    # bad URL caused by decoding Puya's GB18030 path as UTF-8, without touching
    # manual links supplied by any other official source.
    original = list(documents)
    documents = [
        item for item in documents
        if not (isinstance(item, dict) and item.get("verification_status") == document["verification_status"])
    ]
    existing = next((item for item in documents if isinstance(item, dict) and item.get("url") == document["url"]), None)
    if existing is None:
        documents.append(document)
        changed = True
    else:
        before = dict(existing)
        existing.update(document)
        changed = existing != before
    changed = changed or documents != original
    if changed:
        row["documents_json"] = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    return changed


def append_semicolon(value: str, addition: str) -> str:
    values = [item for item in value.split(";") if item]
    if addition not in values:
        values.append(addition)
    return ";".join(values)


def pin_counts(packages: Iterable[str]) -> list[str]:
    values = {match.group(1) for package in packages if (match := re.search(r"(\d{1,3})$", package))}
    return sorted(values, key=int)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "puya-official")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    devices = read_csv(data_dir / "device-variants.csv")
    sources = read_csv(data_dir / "sources.csv")
    puya_devices = [row for row in devices if row.get("manufacturer") == "Puya"]

    queue = deque(SEEDS)
    queued = set(SEEDS)
    pages: dict[str, str] = {}
    errors: list[str] = []
    while queue:
        url = queue.popleft()
        try:
            payload = fetch(url, args.cache_dir / "pages", args.timeout, args.refresh)
            pages[url] = payload
            for link in product_links(payload, url):
                if link not in queued:
                    queued.add(link)
                    queue.append(link)
        except Exception as exc:  # pragma: no cover - remote availability
            errors.append(f"{url}: {type(exc).__name__}: {exc}")

    page_records = {url: table_records(payload, url) for url, payload in pages.items()}
    product_pages: dict[str, tuple[str, list[dict[str, str]]]] = {}
    for url, records in page_records.items():
        for line in {row["product_line"].upper() for row in puya_devices}:
            line_records = [row for row in records if row["part_number"].startswith(line)]
            if line_records:
                previous = product_pages.get(line)
                if previous is None or len(line_records) > len(previous[1]):
                    product_pages[line] = (url, line_records)

    observed = utc_now()
    source_map = {row.get("source_id", ""): row for row in sources}
    provenance: list[dict[str, str]] = []
    package_updates = package_corrections = document_updates = clock_corrections = 0
    no_compatible_rows: list[str] = []
    per_line_matches: Counter[str] = Counter()
    for row in puya_devices:
        page = product_pages.get(row.get("product_line", "").upper())
        if not page:
            continue
        page_url, all_line_records = page
        source_id = "puya:official-product-table:" + row["product_line"].lower()
        source_map[source_id] = {
            "source_id": source_id,
            "source_type": "manufacturer_product_table",
            "publisher": "Puya",
            "title": f"Puya {row['product_line']} official ordering-code table",
            "url": page_url,
            "version": "live",
            "observed_at": observed,
            "verification_scope": (
                "Exact ordering-code product table. Package options require explicit agreement on product line, "
                "core, maximum clock, Flash and SRAM; no suffix decoding is performed."
            ),
        }

        # A data sheet shared by every full ordering-code row is the official
        # product-line manual.  It is safe for every generic record of that
        # line, unlike a PDF visible for only one ordering code.
        urls = {item["datasheet_url"] for item in all_line_records if item["datasheet_url"]}
        if len(urls) == 1 and all(item["datasheet_url"] for item in all_line_records):
            url = next(iter(urls))
            changed = add_document(row, {
                "title": f"Puya {row['product_line']} data sheet",
                "url": url,
                "kind": "datasheet",
                "verification_status": "official_product_table_shared_datasheet",
            })
            document_updates += int(changed)
            if changed:
                provenance.append({
                    "record_type": "device", "record_id": row["device_id"], "field_name": "documents_json",
                    "source_id": source_id, "source_url": page_url, "source_path": "all exact product-table rows",
                    "source_value_json": json.dumps({"datasheet_url": url}, ensure_ascii=False),
                    "observed_at": observed, "verification_status": "official_product_table_shared_datasheet",
                })

        # Older CMSIS packs occasionally retain a pre-production clock value.
        # Correct it only when the official table has a unanimous maximum
        # clock for every row with this explicit product line, Flash, SRAM and
        # core.  A divergent or absent set remains untouched and auditable.
        same_memory_core = matching_memory_core_rows(row, all_line_records)
        official_clocks = {as_int(item["max_clock_mhz"]) for item in same_memory_core}
        official_clocks.discard(None)
        old_clock = as_int(row.get("max_clock_hz", ""))
        if len(official_clocks) == 1:
            official_clock = next(iter(official_clocks)) * 1_000_000
            if old_clock != official_clock:
                row["max_clock_hz"] = str(official_clock)
                update_processor_clock(row, official_clock)
                clock_corrections += 1
                provenance.append({
                    "record_type": "device", "record_id": row["device_id"], "field_name": "max_clock_hz",
                    "source_id": source_id, "source_url": page_url,
                    "source_path": "unanimous exact product-line / core / Flash / SRAM ordering-code rows",
                    "source_value_json": json.dumps([
                        {"part_number": item["part_number"], "max_clock_mhz": item["max_clock_mhz"]}
                        for item in same_memory_core
                    ], ensure_ascii=False, sort_keys=True),
                    "observed_at": observed, "verification_status": "official_product_table_unanimous_variant_value",
                })

        matches = matching_rows(row, all_line_records)
        if not matches:
            no_compatible_rows.append(row["device_name"])
            row["source_id"] = append_semicolon(row.get("source_id", ""), source_id)
            continue
        compatible_urls = {item["datasheet_url"] for item in matches if item["datasheet_url"]}
        if len(compatible_urls) == 1 and all(item["datasheet_url"] for item in matches):
            compatible_url = next(iter(compatible_urls))
            changed = add_document(row, {
                "title": f"Puya {row['device_name']} compatible-variant data sheet",
                "url": compatible_url,
                "kind": "datasheet",
                "verification_status": "official_product_table_compatible_datasheet",
            })
            document_updates += int(changed)
            if changed:
                provenance.append({
                    "record_type": "device", "record_id": row["device_id"], "field_name": "documents_json",
                    "source_id": source_id, "source_url": page_url,
                    "source_path": "exact compatible ordering-code rows",
                    "source_value_json": json.dumps({
                        "datasheet_url": compatible_url,
                        "part_numbers": [item["part_number"] for item in matches],
                    }, ensure_ascii=False, sort_keys=True),
                    "observed_at": observed, "verification_status": "official_product_table_compatible_datasheet",
                })
        packages = sorted({item["package"] for item in matches if item["package"]})
        pins = pin_counts(packages)
        if packages and (row.get("package_types") != ";".join(packages) or row.get("pin_counts") != ";".join(pins)):
            if row.get("package_types"):
                package_corrections += 1
            else:
                package_updates += 1
            row["package_types"] = ";".join(packages)
            row["pin_counts"] = ";".join(pins)
            value = [{"part_number": item["part_number"], "package": item["package"]} for item in matches]
            for field in ("package_types", "pin_counts"):
                provenance.append({
                    "record_type": "device", "record_id": row["device_id"], "field_name": field,
                    "source_id": source_id, "source_url": page_url,
                    "source_path": "exact compatible ordering-code rows",
                    "source_value_json": json.dumps(value, ensure_ascii=False, sort_keys=True),
                    "observed_at": observed, "verification_status": "official_product_table_compatible_rows",
                })
        per_line_matches[row["product_line"]] += 1
        row["source_id"] = append_semicolon(row.get("source_id", ""), source_id)
        row["source_version"] = append_semicolon(row.get("source_version", ""), "live")
        row["verification_status"] = "multi_source_manufacturer_product_table"

    write_csv(data_dir / "device-variants.csv", DEVICE_FIELDS, devices)
    write_csv(data_dir / "sources.csv", SOURCE_FIELDS, sorted(source_map.values(), key=lambda item: item.get("source_id", "")))
    write_csv(data_dir / "puya-official-field-provenance.csv", PROVENANCE_FIELDS, provenance)
    report = {
        "generated_at": observed,
        "pages_crawled": len(pages),
        "product_pages_with_exact_rows": len(product_pages),
        "puya_catalog_devices": len(puya_devices),
        "devices_with_compatible_official_rows": sum(per_line_matches.values()),
        "package_updates": package_updates,
        "package_corrections": package_corrections,
        "clock_corrections": clock_corrections,
        "document_updates": document_updates,
        "remaining_without_compatible_rows": no_compatible_rows,
        "errors": errors,
        "accuracy_policy": [
            "Only https://www.puyasemi.com/en/ product tables are used.",
            "No package, pin count, or capability is decoded from an ordering-code suffix.",
        "Package alternatives are retained only from official rows matching product line, core, clock, Flash and SRAM.",
        "A catalog clock is corrected only when all official rows matching product line, core, Flash and SRAM agree on one value.",
            "A product-line data sheet is attached only if every official ordering-code row links the same PDF.",
        ],
    }
    (data_dir / "puya-official-augmentation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "pages_crawled", "product_pages_with_exact_rows", "puya_catalog_devices",
        "devices_with_compatible_official_rows", "package_updates", "package_corrections", "clock_corrections", "document_updates",
    )}, ensure_ascii=False, indent=2))
    if errors:
        print(json.dumps({"errors": errors}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
