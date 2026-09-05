#!/usr/bin/env python3
"""Augment MM32 devices from exact rows on MindMotion product pages.

MindMotion's public product pages contain complete model-number rows with pin
count, package, package size, and temperature range.  The same pages expose
stable download endpoints behind their JavaScript download buttons.  This
importer records only values attached to an exact model row; it never decodes
package suffixes or copies package data between sibling devices.
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
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = "https://www.mindmotion.com.cn"
SEED = f"{ROOT}/products/mm32mcu/"
USER_AGENT = "MCUS/1.0 (+https://github.com/new-bmp/MCUS; MindMotion official importer)"
PACKAGE_NAMES = {
    "BGA", "CSP", "DFN", "DIP", "LGA", "LQFP", "QFN", "QFP", "SOP",
    "SSOP", "TQFP", "TSSOP", "UFBGA", "UFQFPN", "VFBGA", "WLCSP",
}
PACKAGE_VALUE_RE = re.compile(
    r"\b(BGA|CSP|DFN|DIP|LGA|LQFP|QFN|QFP|SOP|SSOP|TQFP|TSSOP|UFBGA|UFQFPN|VFBGA|WLCSP)"
    r"\s*-?\s*(\d{1,3})\b",
    re.I,
)


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


def clean_text(value: str) -> str:
    value = re.sub(r"<script\b.*?</script>|<style\b.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(value).replace("\xa0", " ").split())


def cache_name(url: str) -> str:
    path = urllib.parse.urlparse(url).path.strip("/")
    tail = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")[-120:] or "root"
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return f"{tail}-{digest}.html"


def fetch(url: str, cache_dir: Path, timeout: float, refresh: bool) -> str:
    path = cache_dir / cache_name(url)
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8", errors="replace")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            break
        except Exception as exc:  # pragma: no cover - remote availability
            last_error = exc
            if attempt < 2:
                time.sleep(0.8 * (2**attempt))
    else:
        assert last_error is not None
        raise last_error
    text = payload.decode("utf-8", errors="replace")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def product_links(payload: str, base_url: str) -> list[str]:
    links = []
    for raw in re.findall(r"\bhref\s*=\s*['\"]([^'\"]+)['\"]", payload, re.I):
        url = urllib.parse.urljoin(base_url, html.unescape(raw))
        parsed = urllib.parse.urlsplit(url)
        if parsed.netloc.lower() != "www.mindmotion.com.cn":
            continue
        if not parsed.path.lower().startswith("/products/mm32mcu/"):
            continue
        if not re.fullmatch(r"/products/mm32mcu/[A-Za-z0-9_./-]*", parsed.path):
            continue
        normalized = urllib.parse.urlunsplit(("https", parsed.netloc, parsed.path, "", ""))
        links.append(normalized if normalized.endswith("/") else normalized + "/")
    return sorted(set(links))


def package_rows(payload: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for block in re.findall(r"<dl\b[^>]*>.*?</dl>", payload, re.I | re.S):
        cells = [clean_text(value) for value in re.findall(r"<dd\b[^>]*>(.*?)</dd>", block, re.I | re.S)]
        model = next(
            (match.group(0).upper() for value in cells for match in [re.search(r"\bMM32[A-Z0-9]+\b", value, re.I)] if match),
            "",
        )
        if not model:
            continue
        package_index = next(
            (index for index, value in enumerate(cells) if value.strip().upper() in PACKAGE_NAMES),
            -1,
        )
        if package_index <= 0:
            continue
        pin_value = cells[package_index - 1].strip()
        if not pin_value.isdigit() or not 4 <= int(pin_value) <= 512:
            continue
        package_family = cells[package_index].strip().upper()
        size = cells[package_index + 1].strip() if package_index + 1 < len(cells) else ""
        temperature = cells[package_index + 2].strip() if package_index + 2 < len(cells) else ""
        drawing = re.search(
            r"IfCz\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]1['\"]\s*\)",
            block,
            re.I,
        )
        rows[model] = {
            "package": f"{package_family}{pin_value}",
            "pins": pin_value,
            "size": size,
            "temperature": temperature,
            "drawing_url": f"{ROOT}/download1.aspx?itemid={drawing.group(2)}&typeid=1" if drawing else "",
            "drawing_title": clean_text(drawing.group(1)) if drawing else "",
        }
    # Legacy product pages use ordinary comparison tables.  Their final
    # Package column contains values such as LQFP48 for the complete part
    # number in the same row.  Parse only exact model rows and explicit package
    # tokens; column positions and part-number suffixes are never decoded.
    for block in re.findall(r"<tr\b[^>]*>.*?</tr>", payload, re.I | re.S):
        cells = [clean_text(value) for value in re.findall(r"<td\b[^>]*>(.*?)</td>", block, re.I | re.S)]
        model = next(
            (match.group(0).upper() for value in cells for match in [re.fullmatch(r"MM32[A-Z0-9]+", value, re.I)] if match),
            "",
        )
        if not model:
            continue
        packages = []
        pins = []
        for value in cells:
            for match in PACKAGE_VALUE_RE.finditer(value):
                package = f"{match.group(1).upper()}{match.group(2)}"
                if package not in packages:
                    packages.append(package)
                if match.group(2) not in pins:
                    pins.append(match.group(2))
        if packages:
            rows.setdefault(model, {
                "package": ";".join(packages),
                "pins": ";".join(pins),
                "size": "",
                "temperature": "",
                "drawing_url": "",
                "drawing_title": "",
            })
    return rows


def exact_model_blocks(payload: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    candidates = [
        *re.findall(r"<dl\b[^>]*>.*?</dl>", payload, re.I | re.S),
        *re.findall(r"<tr\b[^>]*>.*?</tr>", payload, re.I | re.S),
    ]
    for block in candidates:
        match = re.search(r"\bMM32[A-Z0-9]+\b", clean_text(block), re.I)
        if match:
            blocks.setdefault(match.group(0).upper(), block)
    return blocks


def package_drawing(model: str, block: str) -> dict[str, str] | None:
    match = re.search(
        r"IfCz\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]1['\"]\s*\)",
        block,
        re.I,
    )
    if not match:
        return None
    return {
        "title": clean_text(match.group(1)) or f"{model} package drawing",
        "url": f"{ROOT}/download1.aspx?itemid={match.group(2)}&typeid=1",
        "kind": "package_drawing",
        "verification_status": "official_exact_package_drawing",
    }


def page_documents(payload: str) -> list[dict[str, str]]:
    records = []
    pattern = re.compile(
        r"<a\b[^>]*onclick\s*=\s*['\"]IfCz\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]5['\"]\s*\)['\"][^>]*>(.*?)</a>",
        re.I | re.S,
    )
    for filename, item_id, body in pattern.findall(payload):
        title = clean_text(body)
        evidence = f"{filename} {title}".lower()
        if filename.lower().endswith(".pdf") and (
            re.search(r"(?:^|/)ds[_-]", filename, re.I)
            or "datasheet" in evidence
            or "data sheet" in evidence
            or "产品手册" in title
        ):
            kind = "datasheet"
        elif filename.lower().endswith(".pdf") and (
            re.search(r"(?:^|/)um[_-]", filename, re.I)
            or "user manual" in evidence
            or "用户手册" in title
        ):
            kind = "reference_manual"
        else:
            continue
        records.append({
            "title": title or clean_text(filename),
            "url": f"{ROOT}/download1.aspx?itemid={item_id}&typeid=5",
            "kind": kind,
            "verification_status": "official_product_page_download",
        })
    unique = {}
    for record in records:
        unique.setdefault(record["url"], record)
    return list(unique.values())


def add_document(row: dict[str, str], document: dict[str, str]) -> bool:
    try:
        documents = json.loads(row.get("documents_json") or "[]")
    except json.JSONDecodeError:
        documents = []
    if not isinstance(documents, list):
        documents = []
    existing = next(
        (item for item in documents if isinstance(item, dict) and item.get("url") == document.get("url")),
        None,
    )
    if existing is not None:
        before = dict(existing)
        existing.update(document)
        changed = existing != before
    else:
        documents.append(document)
        changed = True
    if changed:
        row["documents_json"] = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    return changed


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "mindmotion-official")
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    devices = read_csv(data / "device-variants.csv")
    sources = read_csv(data / "sources.csv")
    catalog_names = {
        row["device_name"].upper()
        for row in devices
        if row.get("manufacturer") == "MindMotion"
    }

    queue = deque([SEED])
    queued = {SEED}
    pages: dict[str, str] = {}
    errors: list[str] = []
    while queue:
        url = queue.popleft()
        try:
            payload = fetch(url, cache_dir / "pages", args.timeout, args.refresh)
            pages[url] = payload
            for link in product_links(payload, url):
                if link not in queued:
                    queued.add(link)
                    queue.append(link)
        except Exception as exc:  # pragma: no cover - remote availability
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
        time.sleep(0.03)

    evidence: dict[str, dict[str, Any]] = {}
    for url, payload in pages.items():
        rows = package_rows(payload)
        documents = page_documents(payload)
        model_blocks = exact_model_blocks(payload)
        for model in model_blocks:
            if model not in catalog_names:
                continue
            current = evidence.setdefault(model, {"page_url": url, "documents": []})
            if model in rows:
                current.update(rows[model])
            current["page_url"] = url
            for document in documents:
                if document["url"] not in {item["url"] for item in current["documents"]}:
                    current["documents"].append(document)
            drawing = package_drawing(model, model_blocks[model])
            if drawing and drawing["url"] not in {item["url"] for item in current["documents"]}:
                current["documents"].append(drawing)

    observed = utc_now()
    source_map = {row.get("source_id", ""): row for row in sources}
    package_updates = 0
    package_corrections = 0
    document_updates = 0
    matched = 0
    for row in devices:
        item = evidence.get(row.get("device_name", "").upper())
        if not item or row.get("manufacturer") != "MindMotion":
            continue
        matched += 1
        package = item.get("package", "")
        pins = item.get("pins", "")
        if package and (row.get("package_types") != package or row.get("pin_counts") != pins):
            if row.get("package_types"):
                package_corrections += 1
            else:
                package_updates += 1
            row["package_types"] = package
            row["pin_counts"] = pins
        for document in item["documents"]:
            document_updates += int(add_document(row, document))
        page_url = item["page_url"]
        source_id = "mindmotion:product-page:" + hashlib.sha256(page_url.encode()).hexdigest()[:16]
        existing_ids = [value for value in row.get("source_id", "").split(";") if value]
        row["source_id"] = ";".join(dict.fromkeys([*existing_ids, source_id]))
        row["verification_status"] = "multi_source_manufacturer_product_page"
        source_map[source_id] = {
            "source_id": source_id,
            "source_type": "manufacturer_product_page",
            "publisher": "MindMotion",
            "title": f"MindMotion official product table for {row.get('product_line') or row['device_name']}",
            "url": page_url,
            "version": "live",
            "observed_at": observed,
            "verification_scope": "Exact model row, package, pin count, package size, and official document download controls.",
        }

    write_csv(data / "device-variants.csv", devices)
    write_csv(data / "sources.csv", sorted(source_map.values(), key=lambda item: item.get("source_id", "")))
    missing = [
        row["device_name"] for row in devices
        if row.get("manufacturer") == "MindMotion" and not row.get("package_types")
    ]
    report = {
        "generated_at": observed,
        "pages_crawled": len(pages),
        "exact_catalog_models_found": matched,
        "package_updates": package_updates,
        "package_corrections": package_corrections,
        "document_updates": document_updates,
        "remaining_missing_packages": missing,
        "errors": errors,
    }
    report_path = data / "mindmotion-official-augmentation-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
