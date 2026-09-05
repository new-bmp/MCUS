#!/usr/bin/env python3
"""Augment exact Nuvoton parts from the official sitemap and selector API.

Only a product page whose final path segment exactly equals the catalog device
name is accepted.  Package values come from that page's Package section or
from the exact ``partNo`` row returned by Nuvoton's selection-guide endpoint.
No ordering-code suffix is decoded.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = "https://www.nuvoton.com"
SITEMAP_URL = f"{ROOT}/sitemap.xml"
SELECTOR_URL = (
    f"{ROOT}/system/modules/com.thesys.project.nuvoton/pages/"
    "selection-guide/ajax/selectionPage.json"
)
USER_AGENT = "MCUS/1.0 (+https://github.com/new-bmp/MCUS; Nuvoton official importer)"
PACKAGE_RE = re.compile(
    r"\b((?:LQFP|TQFP|QFP|QFN|WQFN|VQFN|TSSOP|SSOP|SOP|BGA|FBGA|"
    r"LFBGA|TFBGA|VFBGA|UFBGA|WLCSP|CSP|LGA|DFN|PDIP)\s*-?\s*\d{1,3})\b",
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


def cache_name(url: str, suffix: str) -> str:
    path = urllib.parse.urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1] or "index"
    stem = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-") or "index"
    return f"{stem}-{hashlib.sha256(url.encode()).hexdigest()[:14]}{suffix}"


def fetch(url: str, path: Path, timeout: float, refresh: bool) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return payload
        except Exception as exc:  # pragma: no cover - remote availability
            last_error = exc
            if attempt < 3:
                time.sleep(1.0 * (2**attempt))
    assert last_error is not None
    raise last_error


def clean_text(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).replace("\xa0", " ").split())


def sitemap_product_pages(payload: bytes) -> dict[str, str]:
    text = payload.decode("utf-8", errors="replace")
    result: dict[str, str] = {}
    for raw in re.findall(r"<loc>([^<]+)</loc>", text, re.I):
        url = html.unescape(raw.strip())
        parsed = urllib.parse.urlsplit(url)
        if parsed.netloc.lower() != "www.nuvoton.com" or "/products/" not in parsed.path.lower():
            continue
        segments = [value for value in parsed.path.split("/") if value]
        if len(segments) < 2 or segments[-1].lower() != "index.html":
            continue
        model = segments[-2].upper()
        if re.fullmatch(r"[A-Z][A-Z0-9-]{3,30}", model):
            result.setdefault(model, url)
    return result


def page_variable(payload: str, key: str) -> str:
    block = re.search(r"\bvar\s+PROD_PAGE_VARS\s*=\s*\{(.*?)\}\s*;?", payload, re.S)
    scope = block.group(1) if block else payload
    match = re.search(rf"\b{re.escape(key)}\s*:\s*['\"]([^'\"]*)['\"]", scope)
    return html.unescape(match.group(1)).strip() if match else ""


def page_packages(payload: str) -> list[str]:
    candidates: list[str] = []
    sections = re.findall(
        r"<p\b[^>]*>.*?(?:<strong\b[^>]*>\s*)?Package(?:\s*</strong>)?.*?</p>(.{0,1800}?)(?:</ul>|</table>|<p\b)",
        payload,
        re.I | re.S,
    )
    for section in sections:
        for match in PACKAGE_RE.finditer(clean_text(section)):
            package = re.sub(r"\s*-?\s*", "", match.group(1)).upper()
            if package not in candidates:
                candidates.append(package)
    return candidates


def fetch_form_json(
    url: str,
    form: dict[str, str],
    path: Path,
    timeout: float,
    refresh: bool,
) -> dict[str, Any]:
    if path.exists() and not refresh:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, dict) and isinstance(parsed.get("itemList"), list):
            return parsed
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(form).encode("utf-8"),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Connection": "close",
        },
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            parsed = json.loads(payload.decode("utf-8-sig"))
            if not isinstance(parsed, dict) or not isinstance(parsed.get("itemList"), list):
                raise ValueError(str(parsed.get("message") or "selector returned no item list"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return parsed
        except Exception as exc:  # pragma: no cover - remote availability
            last_error = exc
            if attempt < 3:
                time.sleep(1.0 * (2**attempt))
    assert last_error is not None
    raise last_error


def selector_series_records(
    family: str,
    series: str,
    folder: str,
    cache_dir: Path,
    timeout: float,
    refresh: bool,
) -> dict[str, dict[str, Any]]:
    if not family or not folder:
        return {}
    form = {"family": family, "page": "1", "start": "0", "limit": "5000"}
    if series:
        form["ProductSeries"] = series
    series_folder = folder.rstrip("/") + "/"
    url = f"{SELECTOR_URL}?currentFolder={urllib.parse.quote(series_folder, safe='/')}"
    key = f"{series_folder}#{family}#{series}"
    path = cache_dir / cache_name(ROOT + key, ".json")
    parsed = fetch_form_json(url, form, path, timeout, refresh)
    rows = parsed.get("itemList") or []
    result: dict[str, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        model = str(item.get("PartNo") or "").upper()
        if model:
            result[model] = item
    return result


def pin_counts(packages: list[str]) -> list[str]:
    values = set()
    for package in packages:
        match = re.search(r"(\d{1,3})$", package)
        if match:
            values.add(match.group(1))
    return sorted(values, key=int)


def add_document(row: dict[str, str], record: dict[str, str]) -> bool:
    try:
        documents = json.loads(row.get("documents_json") or "[]")
    except json.JSONDecodeError:
        documents = []
    if not isinstance(documents, list):
        documents = []
    existing = next(
        (item for item in documents if isinstance(item, dict) and item.get("url") == record["url"]),
        None,
    )
    if existing is not None:
        before = dict(existing)
        existing.update(record)
        changed = existing != before
    else:
        documents.append(record)
        changed = True
    if changed:
        row["documents_json"] = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    return changed


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "nuvoton-official")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    devices = read_csv(data / "device-variants.csv")
    sources = read_csv(data / "sources.csv")
    sitemap = fetch(SITEMAP_URL, cache_dir / "sitemap.xml", args.timeout, args.refresh)
    product_pages = sitemap_product_pages(sitemap)
    catalog_rows = [row for row in devices if row.get("manufacturer") == "Nuvoton"]
    exact_pages = {
        row["device_name"].upper(): product_pages[row["device_name"].upper()]
        for row in catalog_rows
        if row.get("device_name", "").upper() in product_pages
    }

    def load_evidence(model: str, url: str) -> tuple[str, dict[str, Any] | None, list[str]]:
        item_errors: list[str] = []
        try:
            page_path = cache_dir / "pages" / cache_name(url, ".html")
            page = fetch(url, page_path, args.timeout, args.refresh).decode("utf-8", errors="replace")
            if page_variable(page, "partNo").upper() not in {"", model}:
                raise ValueError("page partNo does not match URL model")
            packages = page_packages(page)
            return model, {
                "url": url,
                "packages": packages,
                "datasheet": "",
                "package_size": "",
                "family": page_variable(page, "family"),
                "series": page_variable(page, "series"),
                "folder": page_variable(page, "currentFolder"),
            }, item_errors
        except Exception as exc:  # pragma: no cover - remote availability
            item_errors.append(f"{model} page: {type(exc).__name__}: {exc}")
            return model, None, item_errors

    evidence: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(load_evidence, model, url): model
            for model, url in sorted(exact_pages.items())
        }
        for future in as_completed(futures):
            model, item, item_errors = future.result()
            errors.extend(item_errors)
            if item is not None:
                evidence[model] = item

    selector_groups: dict[tuple[str, str, str], list[str]] = {}
    for model, item in evidence.items():
        group = (item["family"], item["series"], item["folder"])
        if group[0] and group[2]:
            series_folder = group[2].rstrip("/").rsplit("/", 1)[0] + "/"
            selector_groups.setdefault((group[0], group[1], series_folder), []).append(model)

    selector_matches = 0
    with ThreadPoolExecutor(max_workers=min(max(1, args.workers), 6)) as executor:
        futures = {
            executor.submit(
                selector_series_records,
                family,
                series,
                folder,
                cache_dir / "selector-series",
                args.timeout,
                args.refresh,
            ): (family, series, folder, models)
            for (family, series, folder), models in selector_groups.items()
        }
        for future in as_completed(futures):
            family, series, _folder, models = futures[future]
            try:
                records = future.result()
            except Exception as exc:  # pragma: no cover - remote availability
                errors.append(f"{family} / {series} selector: {type(exc).__name__}: {exc}")
                continue
            for model in models:
                selector = records.get(model)
                if not selector:
                    continue
                selector_matches += 1
                package_value = str(selector.get("PackageType") or selector.get("Package") or "")
                selector_packages = [
                    re.sub(r"\s*-?\s*", "", match.group(1)).upper()
                    for match in PACKAGE_RE.finditer(package_value)
                ]
                if selector_packages:
                    evidence[model]["packages"] = list(dict.fromkeys(selector_packages))
                link = str(selector.get("link") or "").strip()
                if link:
                    evidence[model]["datasheet"] = urllib.parse.urljoin(ROOT, link)
                evidence[model]["package_size"] = str(selector.get("PackageSize") or "")

    observed = utc_now()
    source_map = {row.get("source_id", ""): row for row in sources}
    package_updates = 0
    package_corrections = 0
    document_updates = 0
    for row in catalog_rows:
        model = row.get("device_name", "").upper()
        item = evidence.get(model)
        if not item:
            continue
        packages = item["packages"]
        pins = pin_counts(packages)
        if packages and (
            row.get("package_types") != ";".join(packages)
            or (pins and row.get("pin_counts") != ";".join(pins))
        ):
            if row.get("package_types"):
                package_corrections += 1
            else:
                package_updates += 1
            row["package_types"] = ";".join(packages)
            if pins:
                row["pin_counts"] = ";".join(pins)
        document_updates += int(add_document(row, {
            "title": f"Nuvoton {row['device_name']} official product page",
            "url": item["url"],
            "kind": "product_page",
            "verification_status": "official_exact_product_page",
        }))
        if item["datasheet"]:
            document_updates += int(add_document(row, {
                "title": f"Nuvoton {row['device_name']} data sheet",
                "url": item["datasheet"],
                "kind": "datasheet",
                "verification_status": "official_exact_selector_datasheet",
            }))
        source_id = f"nuvoton:product-page:{model.lower()}"
        existing = [value for value in row.get("source_id", "").split(";") if value]
        row["source_id"] = ";".join(dict.fromkeys([*existing, source_id]))
        row["verification_status"] = "multi_source_manufacturer_product_page"
        source_map[source_id] = {
            "source_id": source_id,
            "source_type": "manufacturer_product_page",
            "publisher": "Nuvoton",
            "title": f"Nuvoton {row['device_name']} official product page",
            "url": item["url"],
            "version": "live",
            "observed_at": observed,
            "verification_scope": (
                "Exact part-number product page and, when available, its exact selection-guide row: "
                "package type, package size, and official data-sheet link."
            ),
        }

    write_csv(data / "device-variants.csv", devices)
    write_csv(data / "sources.csv", sorted(source_map.values(), key=lambda row: row.get("source_id", "")))
    remaining = [row["device_name"] for row in catalog_rows if not row.get("package_types")]
    report = {
        "generated_at": observed,
        "sitemap_product_pages": len(product_pages),
        "exact_catalog_pages": len(exact_pages),
        "pages_processed": len(evidence),
        "selector_series_queries": len(selector_groups),
        "selector_exact_matches": selector_matches,
        "package_updates": package_updates,
        "package_corrections": package_corrections,
        "document_updates": document_updates,
        "remaining_missing_packages": remaining,
        "errors": errors,
    }
    (data / "nuvoton-official-augmentation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "sitemap_product_pages", "exact_catalog_pages", "pages_processed",
        "selector_series_queries", "selector_exact_matches",
        "package_updates", "package_corrections", "document_updates",
    )}, ensure_ascii=False, indent=2))
    if errors:
        print(json.dumps({"errors": errors}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
