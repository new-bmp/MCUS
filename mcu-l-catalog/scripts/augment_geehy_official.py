#!/usr/bin/env python3
"""Augment Geehy devices from the official selector API and product pages.

The selector returns complete orderable part numbers and their package names.
Catalog entries imported from CMSIS packs are sometimes generic device names,
so a generic row may legitimately map to more than one official package.  This
script records the union returned by the API and never decodes a name suffix.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = "https://global.geehy.com"
SELECTOR_URL = f"{ROOT}/getModel"
SELECTOR_TYPES = (1, 2, 3, 4, 5, 9, 10, 11)
USER_AGENT = "MCUS/1.0 (+https://github.com/new-bmp/MCUS; Geehy official importer)"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def clean_text(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).replace("\xa0", " ").split())


def safe_url(base: str, raw: str) -> str:
    absolute = urllib.parse.urljoin(base, html.unescape(raw))
    parts = urllib.parse.urlsplit(absolute)
    path = urllib.parse.quote(urllib.parse.unquote(parts.path), safe="/()[],'-._~")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def request_bytes(request: urllib.request.Request, timeout: float) -> bytes:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - remote availability
            last_error = exc
            if attempt < 2:
                time.sleep(0.8 * (2**attempt))
    assert last_error is not None
    raise last_error


def selector_rows(selector_type: int, cache_dir: Path, timeout: float, refresh: bool) -> list[dict[str, Any]]:
    cache_path = cache_dir / f"selector-type-{selector_type}.json"
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        filters = [
            {"name": "modelselect_type", "value": str(selector_type)},
            {"order": ""},
        ]
        body = urllib.parse.urlencode({"arr": json.dumps(filters, separators=(",", ":"))}).encode()
        request = urllib.request.Request(
            SELECTOR_URL,
            data=body,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Accept": "application/json",
            },
        )
        payload = json.loads(request_bytes(request, timeout).decode("utf-8"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if payload.get("code") != 1 or not isinstance(payload.get("list"), list):
        raise ValueError(f"unexpected selector response for type {selector_type}")
    return [item for item in payload["list"] if isinstance(item, dict)]


def product_page_url(product_line: str, device_name: str) -> str:
    page_name = "BMP561" if product_line == "BMP" else product_line
    if product_line == "BMP" and not device_name.upper().startswith("BMP561"):
        page_name = device_name
    return f"{ROOT}/product/fifth/{urllib.parse.quote(page_name, safe='-_')}"


def fetch_page(url: str, cache_dir: Path, timeout: float, refresh: bool) -> str:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = cache_dir / f"{slug(url.rsplit('/', 1)[-1])}-{digest}.html"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8", errors="replace")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    payload = request_bytes(request, timeout)
    text = payload.decode("utf-8", errors="replace")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def page_documents(payload: str, page_url: str) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for row in re.findall(r"<tr\b[^>]*>.*?</tr>", payload, flags=re.I | re.S):
        row_text = clean_text(row)
        kind = ""
        if re.search(r"\bdata\s*sheet\b|\bdatasheet\b", row_text, re.I):
            kind = "datasheet"
        elif re.search(r"\buser\s*manual\b|\breference\s*manual\b", row_text, re.I):
            kind = "reference_manual"
        if not kind:
            continue
        link = re.search(r"\bhref\s*=\s*['\"]([^'\"]+\.pdf(?:\?[^'\"]*)?)['\"]", row, re.I)
        if not link:
            continue
        anchor = re.search(r"<a\b[^>]*>(.*?)</a>", row, re.I | re.S)
        title = clean_text(anchor.group(1)) if anchor else row_text
        url = safe_url(page_url, link.group(1))
        cells = [clean_text(value) for value in re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.I | re.S)]
        version = next((value for value in cells if re.fullmatch(r"V?\d+(?:\.\d+){1,3}", value, re.I)), "")
        record = {
            "title": title or ("Datasheet" if kind == "datasheet" else "User Manual"),
            "url": url,
            "version": version,
            "kind": kind,
            "verification_status": "official_product_page",
        }
        if url not in {item["url"] for item in documents}:
            documents.append(record)
    return documents


def add_document(row: dict[str, str], document: dict[str, str]) -> bool:
    try:
        current = json.loads(row.get("documents_json") or "[]")
    except json.JSONDecodeError:
        current = []
    if not isinstance(current, list):
        current = []
    existing = next(
        (item for item in current if isinstance(item, dict) and item.get("url") == document["url"]),
        None,
    )
    if existing is not None:
        before = dict(existing)
        existing.update(document)
        changed = existing != before
    else:
        current.append(document)
        changed = True
    row["documents_json"] = json.dumps(current, ensure_ascii=False, sort_keys=True)
    return changed


def pin_counts(packages: list[str]) -> list[str]:
    values = set()
    for package in packages:
        match = re.search(r"(\d{2,3})$", package)
        if match:
            values.add(match.group(1))
    return sorted(values, key=int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/combined"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/geehy-official"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    data = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    devices = read_csv(data / "device-variants.csv")
    parts = read_csv(data / "orderable-parts.csv")
    sources = read_csv(data / "sources.csv")
    errors: list[str] = []
    selector: list[dict[str, Any]] = []
    for selector_type in SELECTOR_TYPES:
        try:
            selector.extend(selector_rows(selector_type, cache_dir, args.timeout, args.refresh))
        except Exception as exc:  # pragma: no cover - remote availability
            errors.append(f"selector type {selector_type}: {type(exc).__name__}: {exc}")

    selector_by_code = {
        str(item.get("product_code") or "").upper(): item
        for item in selector
        if item.get("product_code") and item.get("package")
    }
    geehy_rows = [row for row in devices if row.get("manufacturer") == "Geehy"]
    page_cache: dict[str, tuple[str, list[dict[str, str]]]] = {}
    for row in geehy_rows:
        key = row.get("product_line", "")
        if key in page_cache:
            continue
        url = product_page_url(key, row.get("device_name", ""))
        try:
            payload = fetch_page(url, cache_dir / "pages", args.timeout, args.refresh)
            page_cache[key] = (url, page_documents(payload, url))
        except Exception as exc:  # pragma: no cover - remote availability
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
        time.sleep(0.03)

    observed = utc_now()
    source_map = {row.get("source_id", ""): row for row in sources}
    existing_parts = {row.get("part_number", "").upper() for row in parts}
    package_updates = 0
    package_corrections = 0
    document_updates = 0
    orderable_part_updates = 0
    matched_devices = 0
    for row in geehy_rows:
        name = row.get("device_name", "").upper()
        matches = [
            (code, item) for code, item in selector_by_code.items()
            if code.startswith(name)
        ]
        packages = sorted({str(item.get("package") or "").strip() for _, item in matches if item.get("package")})
        if matches:
            matched_devices += 1
        if packages:
            package_value = ";".join(packages)
            pins = ";".join(pin_counts(packages))
            if row.get("package_types") != package_value or (pins and row.get("pin_counts") != pins):
                if row.get("package_types"):
                    package_corrections += 1
                else:
                    package_updates += 1
                row["package_types"] = package_value
                if pins:
                    row["pin_counts"] = pins
            row["verification_status"] = "manufacturer_product_selector_api"

        page_url, documents = page_cache.get(row.get("product_line", ""), ("", []))
        for document in documents:
            document_updates += int(add_document(row, document))
        source_id = "geehy:official-selector-api"
        ids = [value for value in row.get("source_id", "").split(";") if value]
        row["source_id"] = ";".join(dict.fromkeys([*ids, source_id]))
        if page_url:
            page_source_id = f"geehy:product-page:{slug(row.get('product_line') or name)}"
            row["source_id"] = ";".join(dict.fromkeys([*row["source_id"].split(";"), page_source_id]))
            source_map[page_source_id] = {
                "source_id": page_source_id,
                "source_type": "manufacturer_product_page",
                "publisher": "Geehy",
                "title": f"Geehy {row.get('product_line') or name} official product page",
                "url": page_url,
                "version": "live",
                "observed_at": observed,
                "verification_scope": "Official direct datasheet and user-manual links for this product line.",
            }

        for code, item in matches:
            if code in existing_parts:
                continue
            parts.append({
                "orderable_part_id": f"geehy::{slug(code)}",
                "device_id": row.get("device_id", ""),
                "manufacturer": "Geehy",
                "family": row.get("family", ""),
                "series": row.get("series", ""),
                "product_line": row.get("product_line", ""),
                "device_name": row.get("device_name", ""),
                "part_number": code,
                "manufacturer_suffix": code[len(name):],
                "package_code": "",
                "temperature_grade_code": "",
                "packing_code": "",
                "package_name": str(item.get("package") or ""),
                "temperature_range": "",
                "packing_form": "",
                "lifecycle": "unknown",
                "source_id": source_id,
                "source_url": page_url or SELECTOR_URL,
                "observed_at": observed,
                "verification_status": "manufacturer_product_selector_api",
                "decode_status": "exact_official_model;official_selector_package",
            })
            existing_parts.add(code)
            orderable_part_updates += 1

    source_map["geehy:official-selector-api"] = {
        "source_id": "geehy:official-selector-api",
        "source_type": "manufacturer_product_selector_api",
        "publisher": "Geehy",
        "title": "Geehy official product selector",
        "url": f"{ROOT}/product/select",
        "version": "live",
        "observed_at": observed,
        "verification_scope": "Complete orderable part number and package returned by the official selector API.",
    }

    write_csv(data / "device-variants.csv", devices)
    write_csv(data / "orderable-parts.csv", parts)
    write_csv(data / "sources.csv", sorted(source_map.values(), key=lambda item: item.get("source_id", "")))
    remaining = [row["device_name"] for row in geehy_rows if not row.get("package_types")]
    report = {
        "generated_at": observed,
        "selector_rows": len(selector_by_code),
        "product_pages": len(page_cache),
        "matched_devices": matched_devices,
        "package_updates": package_updates,
        "package_corrections": package_corrections,
        "document_updates": document_updates,
        "orderable_part_updates": orderable_part_updates,
        "remaining_missing_packages": remaining,
        "errors": errors,
    }
    (data / "geehy-official-augmentation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
