#!/usr/bin/env python3
"""Fill package and document data from exact official sources.

The adapters use GigaDevice's document center, Artery's document API, and
Espressif's selector/sitemap. PDFs are downloaded where package-code tables
must be parsed. A model is updated only when the manufacturer publishes both
the exact device/selector record and the corresponding package-code mapping.
No package is inferred from a sibling device or from a Cartesian suffix rule.

The same command also parses exact package features in any locally cached
CMSIS PDSC.  This makes the augmenter useful for future vendor packs without
weakening the evidence boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


USER_AGENT = "MCUS/1.0 (+https://github.com/new-bmp/MCUS; official package importer)"
GIGA_ROOT = "https://www.gd32mcu.com"
ARTERY_ROOT = "https://www.arterytek.com"
ARTERY_DOCUMENT_INDEX = f"{ARTERY_ROOT}/json/Document.json"
ESPRESSIF_DOCUMENT_SITEMAP = (
    "https://documentation.espressif.com/api/sitemap/get/TECHNICAL_FILE_EN"
)
EXACT_OFFICIAL_PACKAGES = {
    "STM32WBA5MMGHXU": {
        "manufacturer": "STMicroelectronics",
        "package": "SiP-LGA76 8x12.5 mm",
        "pins": "76",
        "title": "STM32WBA5MMG data sheet",
        "url": "https://www.st.com/resource/en/datasheet/stm32wba5mmg.pdf",
        "source_id": "stmicroelectronics:datasheet:stm32wba5mmg",
        "verification_scope": (
            "Exact STM32WBA5MMGHxU ordering-information entry: H = LGA; "
            "package table specifies a 76-pin 8 x 12.5 mm SiP-LGA."
        ),
    },
}
GIGA_QUERIES = (
    "GD32C1", "GD32E1", "GD32E2", "GD32E5", "GD32F1", "GD32F2", "GD32F3",
    "GD32F4", "GD32F5", "GD32L2", "GD32W5", "GD32VF1", "GD32A", "GD32G5",
    "GD32H7", "GD32M5", "GD32VW5",
)
PACKAGE_RE = re.compile(
    r"\b(?P<family>GD32[A-Z0-9]+?)(?P<code>[A-Z])x(?:[A-Z0-9]+)?\s+"
    r"(?P<package>(?:LQFP|TQFP|QFN|BGA|WLCSP|TSSOP|SOP|SSOP|QFP|LGA|UFBGA|UFQFPN|CSP|DFN)\s*-?\s*\d+"
    r"(?:\s*/\s*(?:LQFP|TQFP|QFN|BGA|WLCSP|TSSOP|SOP|SSOP|QFP|LGA|UFBGA|UFQFPN|CSP|DFN)\s*-?\s*\d+)*)\b",
    re.IGNORECASE,
)
PACKAGE_FEATURE_TYPES = {
    "BGA", "CSP", "DFN", "DIP", "LGA", "QFN", "QFP", "SO", "SON", "SOP",
    "SSOP", "TQFP", "TSSOP", "UFBGA", "UFQFPN", "VFBGA", "WLCSP", "LQFP",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "document"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fetch(url: str, cache_path: Path, timeout: float) -> tuple[bytes, str, int]:
    if cache_path.exists():
        payload = cache_path.read_bytes()
        return payload, url, 200
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                final_url = response.geturl()
                status = response.status
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
            return payload, final_url, status
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.6 * (2**attempt))
    assert last_error is not None
    raise last_error


def parse_datasheet_index(payload: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    # The page uses a repeated <li><dl> block.  Restrict extraction to that
    # block so sidebar links are never mistaken for a document.
    for block in re.findall(r'<li[^>]*class="cl"[^>]*>(.*?)</li>', payload, re.IGNORECASE | re.DOTALL):
        name_match = re.search(r'class="data-name"[^>]*>(.*?)</dd>', block, re.IGNORECASE | re.DOTALL)
        href_match = re.search(r'href="([^"]*document_id[^\"]*)"', block, re.IGNORECASE)
        version_match = re.search(r'class="data-version"[^>]*>(.*?)</dd>', block, re.IGNORECASE | re.DOTALL)
        if not name_match or not href_match:
            continue
        name = re.sub(r"<[^>]+>", " ", name_match.group(1))
        name = " ".join(name.split())
        if "datasheet" not in name.lower():
            continue
        version = " ".join(re.sub(r"<[^>]+>", " ", version_match.group(1)).split()) if version_match else ""
        link = urllib.parse.urljoin(GIGA_ROOT, href_match.group(1))
        records.append({"title": name, "version": version, "page_url": link})
    unique: dict[str, dict[str, str]] = {}
    for record in records:
        unique.setdefault(record["page_url"], record)
    return list(unique.values())


def extract_pdf_text(payload: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    import io
    try:
        reader = PdfReader(io.BytesIO(payload))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def extract_pdf_text_cached(payload: bytes, cache_path: Path) -> str:
    text_path = cache_path.with_suffix(cache_path.suffix + ".txt")
    if text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="replace")
    text = extract_pdf_text(payload)
    if text:
        text_path.write_text(text, encoding="utf-8")
    return text


def extract_package_mappings(text: str) -> dict[str, dict[str, str]]:
    text = re.sub(r"[ \t]+", " ", text)
    mappings: dict[str, dict[str, str]] = {}
    for match in PACKAGE_RE.finditer(text):
        family = match.group("family").upper()
        code = match.group("code").upper()
        package = re.sub(r"\s+", "", match.group("package")).upper().replace("/", ";")
        key = f"{family}:{code}"
        mappings[key] = {"family": family, "code": code, "package": package}
    return mappings


def list_value(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    return value if isinstance(value, list) else [value]


def artery_document_records(payload: bytes) -> dict[str, list[dict[str, str]]]:
    parsed = json.loads(payload.decode("utf-8-sig"))
    by_product: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in list_value(parsed.get("data") if isinstance(parsed, dict) else parsed):
        if not isinstance(item, dict):
            continue
        applications = {
            str(value.get("applicationType") or "").upper()
            for value in list_value(item.get("documentApplicationList"))
            if isinstance(value, dict)
        }
        kind = "datasheet" if "DS" in applications else "reference_manual" if "RM" in applications else ""
        if not kind:
            continue
        attachments = [
            value for value in list_value(item.get("documentAttachmentList"))
            if isinstance(value, dict) and str(value.get("filePath") or "").lower().endswith(".pdf")
        ]
        if not attachments:
            continue
        english = [value for value in attachments if value.get("enFileName")]
        selected = english[0] if english else attachments[0]
        url = urllib.parse.urljoin(ARTERY_ROOT, str(selected["filePath"]))
        title = str(item.get("enDocName") or item.get("cnDocName") or item.get("docNo") or "Artery document").strip()
        version = str(item.get("enVersion") or item.get("cnVersion") or "").strip()
        doc_no = str(item.get("docNo") or "").strip()
        for support in list_value(item.get("documentSupportProductList")):
            if not isinstance(support, dict):
                continue
            product = str(support.get("enProduct") or support.get("cnProduct") or "").strip().upper()
            if product:
                by_product[product].append({
                    "title": title,
                    "url": url,
                    "version": version,
                    "kind": kind,
                    "doc_no": doc_no,
                    "verification_status": "official_document_api",
                })
    return by_product


def extract_artery_package_mappings(text: str) -> dict[str, dict[str, str]]:
    lowered = text.lower()
    position = max(
        lowered.rfind("part numbering"),
        text.rfind("型号说明"),
        text.rfind("订货代码"),
    )
    if position < 0:
        return {"pins": {}, "packages": {}}
    section = re.sub(r"[ \t]+", " ", text[position: position + 5000])
    pin_match = re.search(
        r"Pin count\s+(.*?)(?:Internal Flash memory size|Flash memory size)",
        section,
        re.IGNORECASE | re.DOTALL,
    )
    package_match = re.search(
        r"Package(?: type)?\s+(.*?)(?:Temperature range)",
        section,
        re.IGNORECASE | re.DOTALL,
    )
    pins = {
        code.upper(): count
        for code, count in re.findall(r"\b([A-Z])\s*=\s*(\d+)\s*pins?\b", pin_match.group(1) if pin_match else "", re.IGNORECASE)
    }
    packages = {
        code.upper(): name.upper().replace(" ", "")
        for code, name in re.findall(
            r"\b([A-Z])\s*=\s*(LQFP|TQFP|QFN|TSSOP|SSOP|SOP|BGA|LGA|WLCSP|CSP|DFN)\b",
            package_match.group(1) if package_match else "",
            re.IGNORECASE,
        )
    }
    if not pins:
        pin_match = re.search(
            r"引脚数目?\s+(.*?)(?:内部闪存|闪存存储器)",
            section,
            re.DOTALL,
        )
        pins = {
            code.upper(): count
            for code, count in re.findall(r"\b([A-Z])\s*=\s*(\d+)\s*脚", pin_match.group(1) if pin_match else "")
        }
    if not packages:
        package_match = re.search(r"(?:^|\n)封装\s+(.*?)(?:温度范围)", section, re.DOTALL)
        packages = {
            code.upper(): name.upper().replace(" ", "")
            for code, name in re.findall(
                r"\b([A-Z])\s*=\s*(LQFP|TQFP|QFN|TSSOP|SSOP|SOP|BGA|LGA|WLCSP|CSP|DFN)\b",
                package_match.group(1) if package_match else "",
                re.IGNORECASE,
            )
        }
    return {"pins": pins, "packages": packages}


def espressif_selector_rows(cache_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename, product_kind in (("chip-products-en.json", "chip"), ("module-products-en.json", "module")):
        path = cache_dir / filename
        if not path.exists():
            continue
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        for item in list_value(parsed.get("tableData") if isinstance(parsed, dict) else []):
            if isinstance(item, dict):
                rows.append({**item, "_product_kind": product_kind})
    return rows


def normalized_product(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def espressif_datasheet_urls(payload: bytes) -> list[str]:
    root = ET.fromstring(payload)
    urls = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "loc" or not node.text:
            continue
        url = node.text.strip()
        if re.search(r"datasheet[^/]*\.html$", url, re.IGNORECASE):
            urls.append(url)
    return sorted(set(urls))


def match_espressif_datasheets(item: dict[str, Any], urls: list[str]) -> list[str]:
    name = normalized_product(str(item.get("name") or item.get("mpn") or ""))
    series = normalized_product(str(item.get("series") or ""))
    if not name:
        return []
    aliases = {name}
    if name.endswith("u"):
        aliases.add(name[:-1])
    if name.endswith("-i"):
        aliases.add(name[:-2])
    suffixes = {
        alias[len(series) + 1:]
        for alias in aliases
        if series and alias.startswith(series + "-")
    }
    exact: list[tuple[int, str]] = []
    series_docs: list[str] = []
    for url in urls:
        document = normalized_product(Path(urllib.parse.urlparse(url).path).stem)
        document = re.sub(r"-datasheet(?:-en)?$", "", document)
        if series and document in {series, f"0a-{series}"}:
            series_docs.append(url)
        score = 0
        if any(alias and alias in document for alias in aliases):
            score = 100 + max(len(alias) for alias in aliases if alias in document)
        elif series and series in document and any(suffix and suffix in document for suffix in suffixes):
            score = 80 + max(len(suffix) for suffix in suffixes if suffix in document)
        if score:
            exact.append((score, url))
    result: list[str] = []
    if exact:
        result.append(max(exact, key=lambda value: (value[0], -len(value[1])))[1])
    for url in series_docs:
        if url not in result:
            result.append(url)
    return result[:2]


def package_pin_counts(package_values: list[str]) -> list[str]:
    pins = set()
    for package in package_values:
        match = re.search(r"(?:^|[^0-9])(\d{2,3})(?:\s*(?:PIN|BALL))?$", package, re.IGNORECASE)
        if match:
            pins.add(match.group(1))
    return sorted(pins, key=int)


def pdsc_package_evidence(cache_dir: Path) -> dict[str, list[dict[str, str]]]:
    evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    pdsc_paths = cache_dir.rglob("*.pdsc") if cache_dir.exists() else []
    for path in pdsc_paths:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        def walk(node: ET.Element, inherited: list[dict[str, str]]) -> None:
            current = inherited + [child.attrib for child in node if child.tag.rsplit("}", 1)[-1] == "feature"]
            name = node.attrib.get("Dname") or node.attrib.get("Dvariant")
            if name:
                packages = [item for item in current if item.get("type", "").upper() in PACKAGE_FEATURE_TYPES]
                if packages:
                    evidence[name.upper()].extend(packages)
            for child in node:
                walk(child, current)
        walk(root, [])
    return evidence


def clean_package(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).upper()


def add_doc(
    row: dict[str, str],
    title: str,
    url: str,
    version: str = "",
    *,
    kind: str = "datasheet",
    verification_status: str = "official_datasheet",
) -> None:
    try:
        docs = json.loads(row.get("documents_json") or "[]")
    except json.JSONDecodeError:
        docs = []
    if not isinstance(docs, list):
        docs = []
    existing = next((item for item in docs if isinstance(item, dict) and item.get("url") == url), None)
    record = {
        "title": title,
        "url": url,
        "version": version,
        "kind": kind,
        "verification_status": verification_status,
    }
    if existing is None:
        docs.append(record)
    else:
        existing.update({key: value for key, value in record.items() if value})
    row["documents_json"] = json.dumps(docs, ensure_ascii=False, sort_keys=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "gigadevice-datasheets")
    parser.add_argument("--artery-cache", type=Path, default=root / "cache" / "artery-documents")
    parser.add_argument("--espressif-cache", type=Path, default=root / "cache" / "espressif")
    parser.add_argument("--pdsc-cache", type=Path, default=root / "cache" / "pdsc")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    devices = read_csv(data / "device-variants.csv")
    orderable_parts = read_csv(data / "orderable-parts.csv")
    sources = read_csv(data / "sources.csv")
    if not devices:
        raise SystemExit("device-variants.csv is empty")

    index_entries: list[dict[str, str]] = []
    index_errors: list[str] = []
    for query in GIGA_QUERIES:
        url = f"{GIGA_ROOT}/cn/download/5?kw={urllib.parse.quote(query)}"
        try:
            payload, _, _ = fetch(url, args.cache_dir / "index" / f"{query}.html", args.timeout)
            index_entries.extend(parse_datasheet_index(payload.decode("utf-8", errors="replace")))
        except Exception as exc:  # pragma: no cover - remote availability
            index_errors.append(f"{query}: {type(exc).__name__}: {exc}")
    unique_entries = {item["page_url"]: item for item in index_entries}

    documents: list[dict[str, Any]] = []
    mappings: dict[str, dict[str, str]] = {}
    for index, item in enumerate(unique_entries.values(), 1):
        try:
            pdf_path = args.cache_dir / "pdf" / f"{slug(item['title'])}-{index}.pdf"
            payload, final_url, status = fetch(item["page_url"], pdf_path, args.timeout)
            text = extract_pdf_text_cached(payload, pdf_path)
            for key, mapping in extract_package_mappings(text).items():
                mapping = {**mapping, "url": final_url, "title": item["title"], "version": item["version"],
                           "sha256": hashlib.sha256(payload).hexdigest()}
                mappings.setdefault(key, mapping)
            documents.append({**item, "url": final_url, "status": status,
                              "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
        except Exception as exc:  # pragma: no cover - remote availability
            documents.append({**item, "url": item["page_url"], "status": "error", "error": str(exc)})

    artery_errors: list[str] = []
    artery_docs: dict[str, list[dict[str, str]]] = {}
    artery_package_maps: dict[str, dict[str, Any]] = {}
    try:
        artery_payload, _, _ = fetch(
            ARTERY_DOCUMENT_INDEX,
            args.artery_cache.resolve() / "Document.json",
            args.timeout,
        )
        artery_docs = artery_document_records(artery_payload)
    except Exception as exc:  # pragma: no cover - remote availability
        artery_errors.append(f"document index: {type(exc).__name__}: {exc}")

    catalog_artery_products = {
        str(row.get("product_line") or "").upper()
        for row in devices
        if row.get("manufacturer") == "Artery"
    }
    artery_datasheets = {
        product: next((doc for doc in docs if doc.get("kind") == "datasheet"), None)
        for product, docs in artery_docs.items()
        if product in catalog_artery_products
    }
    artery_datasheets = {product: doc for product, doc in artery_datasheets.items() if doc}

    def download_artery_datasheet(product: str, doc: dict[str, str]) -> tuple[str, dict[str, Any]]:
        pdf_path = args.artery_cache.resolve() / "pdf" / f"{slug(doc.get('doc_no') or product)}.pdf"
        payload, final_url, status = fetch(
            doc["url"],
            pdf_path,
            args.timeout,
        )
        return product, {
            **extract_artery_package_mappings(extract_pdf_text_cached(payload, pdf_path)),
            "url": final_url,
            "status": status,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "document": doc,
        }

    # Artery throttles bursts of large PDF requests. Two workers keep the
    # importer reliable while still allowing a first-time cache fill.
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 2))) as executor:
        futures = {
            executor.submit(download_artery_datasheet, product, doc): product
            for product, doc in artery_datasheets.items()
        }
        for future in as_completed(futures):
            product = futures[future]
            try:
                found_product, package_map = future.result()
                artery_package_maps[found_product] = package_map
            except Exception as exc:  # pragma: no cover - remote availability
                artery_errors.append(f"{product}: {type(exc).__name__}: {exc}")

    selector_rows = espressif_selector_rows(args.espressif_cache.resolve())
    selector_by_mpn = {
        str(item.get("mpn") or item.get("name") or "").upper(): item
        for item in selector_rows
        if item.get("mpn") or item.get("name")
    }
    espressif_urls: list[str] = []
    espressif_errors: list[str] = []
    try:
        sitemap, _, _ = fetch(
            ESPRESSIF_DOCUMENT_SITEMAP,
            args.espressif_cache.resolve() / "technical-file-en-sitemap.xml",
            args.timeout,
        )
        espressif_urls = espressif_datasheet_urls(sitemap)
    except Exception as exc:  # pragma: no cover - remote availability
        espressif_errors.append(f"document sitemap: {type(exc).__name__}: {exc}")

    official_parts_by_device: dict[str, list[dict[str, str]]] = defaultdict(list)
    for part in orderable_parts:
        if (
            part.get("device_id")
            and part.get("manufacturer") != "Artery"
            and part.get("package_name")
            and part.get("source_url", "").startswith(("http://", "https://"))
            and part.get("verification_status", "").startswith("manufacturer_")
        ):
            official_parts_by_device[part["device_id"]].append(part)

    # Prefer the longest matching datasheet family.  A mapping is applied only
    # when the device suffix selects exactly one package code in that PDF.
    updated = 0
    package_updated = 0
    package_corrected = 0
    artery_document_updates = 0
    espressif_document_updates = 0
    espressif_module_updates = 0
    orderable_package_updates = 0
    artery_unresolved_packages: list[str] = []
    artery_packages_by_device: dict[str, tuple[str, str] | None] = {}
    package_evidence = pdsc_package_evidence(args.pdsc_cache.resolve())
    source_by_url = {row.get("url", ""): row for row in sources}
    added_sources: dict[str, dict[str, str]] = {}
    for row in devices:
        name = row.get("device_name", "")
        if not name:
            continue
        changed = False
        original_package = row.get("package_types", "")
        original_documents = row.get("documents_json", "")
        exact_official = EXACT_OFFICIAL_PACKAGES.get(name.upper())
        if exact_official and row.get("manufacturer") == exact_official["manufacturer"]:
            if (
                row.get("package_types") != exact_official["package"]
                or row.get("pin_counts") != exact_official["pins"]
            ):
                if row.get("package_types"):
                    package_corrected += 1
                else:
                    package_updated += 1
                row["package_types"] = exact_official["package"]
                row["pin_counts"] = exact_official["pins"]
                changed = True
            add_doc(
                row,
                exact_official["title"],
                exact_official["url"],
                kind="datasheet",
                verification_status="official_exact_device_datasheet",
            )
            source_id = exact_official["source_id"]
            row["source_id"] = ";".join(dict.fromkeys([
                value for value in [*row.get("source_id", "").split(";"), source_id] if value
            ]))
            row["source_url"] = exact_official["url"]
            row["verification_status"] = "multi_source_manufacturer_datasheet"
            added_sources[source_id] = {
                "source_id": source_id,
                "source_type": "manufacturer_datasheet",
                "publisher": exact_official["manufacturer"],
                "title": exact_official["title"],
                "url": exact_official["url"],
                "version": "live",
                "observed_at": utc_now(),
                "verification_scope": exact_official["verification_scope"],
            }
            if row.get("documents_json", "") != original_documents:
                changed = True
        # Exact cached PDSC evidence is always preferred over a heuristic name
        # parser.  It is safe because the package feature belongs to the exact
        # device node, not to a sibling family.
        exact = package_evidence.get(name.upper(), [])
        if exact and (not row.get("package_types") or not row.get("pin_counts")):
            package_values = {(clean_package(x.get("type", "")), x.get("n", "")) for x in exact}
            package_values = {(p, n) for p, n in package_values if p and n.isdigit()}
            if len(package_values) == 1:
                package, pins = next(iter(package_values))
                if not row.get("package_types"):
                    row["package_types"] = package
                if not row.get("pin_counts"):
                    row["pin_counts"] = pins
                row["verification_status"] = "multi_source_official_package_metadata"
                changed = True

        matching = [mapping for mapping in mappings.values() if name.upper().startswith(mapping["family"])]
        if matching and not row.get("package_types"):
            longest = max(len(item["family"]) for item in matching)
            matching = [item for item in matching if len(item["family"]) == longest]
            suffix = name.upper()[longest:]
            code = suffix[:1] if suffix and suffix[:1].isalpha() else ""
            selected = [item for item in matching if item["code"] == code]
            # If the datasheet publishes one package for a family without a
            # package-code split, accept that single exact family mapping.
            if not selected and len({item["package"] for item in matching}) == 1:
                selected = matching[:1]
            if len(selected) == 1:
                item = selected[0]
                row["package_types"] = item["package"]
                row["pin_counts"] = re.search(r"(\d+)$", item["package"]).group(1)
                source_id = f"gigadevice:datasheet:{slug(item['family'])}"
                row["source_id"] = ";".join(dict.fromkeys([*row.get("source_id", "").split(";"), source_id]))
                row["source_url"] = item["url"]
                row["source_version"] = item.get("version", "") or row.get("source_version", "")
                row["verification_status"] = "multi_source_manufacturer_datasheet"
                add_doc(row, item["title"], item["url"], item.get("version", ""))
                added_sources[source_id] = {
                    "source_id": source_id, "source_type": "manufacturer_datasheet", "publisher": "GigaDevice",
                    "title": item["title"], "url": item["url"], "version": item.get("version", ""),
                    "observed_at": utc_now(),
                    "verification_scope": "Exact GigaDevice datasheet package/code and pin-count table.",
                }
                package_updated += 1
                changed = True

        if row.get("manufacturer") == "Artery":
            product = str(row.get("product_line") or "").upper()
            for doc in artery_docs.get(product, []):
                add_doc(
                    row,
                    doc["title"],
                    doc["url"],
                    doc.get("version", ""),
                    kind=doc["kind"],
                    verification_status=doc["verification_status"],
                )
                doc_source_id = f"artery:{doc['kind']}:{slug(product)}"
                row["source_id"] = ";".join(dict.fromkeys([
                    value for value in [*row.get("source_id", "").split(";"), doc_source_id] if value
                ]))
                added_sources[doc_source_id] = {
                    "source_id": doc_source_id,
                    "source_type": f"manufacturer_{doc['kind']}",
                    "publisher": "ArteryTek",
                    "title": doc["title"],
                    "url": doc["url"],
                    "version": doc.get("version", ""),
                    "observed_at": utc_now(),
                    "verification_scope": "Direct PDF published by the Artery official document API.",
                }
            package_map = artery_package_maps.get(product, {})
            suffix = name.upper()[len(product):] if name.upper().startswith(product) else ""
            pin_code = next((code for code in suffix if code in package_map.get("pins", {})), "")
            package_code = next((code for code in reversed(suffix) if code in package_map.get("packages", {})), "")
            if pin_code and package_code:
                pins = package_map["pins"][pin_code]
                package = f"{package_map['packages'][package_code]}{pins}"
                artery_packages_by_device[row.get("device_id", "")] = (package, pins)
                if row.get("package_types") != package or row.get("pin_counts") != pins:
                    if row.get("package_types"):
                        package_corrected += 1
                    else:
                        package_updated += 1
                    row["package_types"] = package
                    row["pin_counts"] = pins
                    row["verification_status"] = "multi_source_manufacturer_datasheet"
                    changed = True
            elif package_map:
                artery_packages_by_device[row.get("device_id", "")] = None
                artery_unresolved_packages.append(name)
                if row.get("package_types") or row.get("pin_counts"):
                    row["package_types"] = ""
                    row["pin_counts"] = ""
                    row["verification_status"] = "official_datasheet_variant_not_listed"
                    package_corrected += 1
                    changed = True
            if row.get("documents_json", "") != original_documents:
                artery_document_updates += 1
                changed = True

        if row.get("manufacturer") == "Espressif":
            selector = selector_by_mpn.get(name.upper())
            if selector:
                if not row.get("package_types"):
                    official_package = str(selector.get("package") or "").strip()
                    module_size = str(selector.get("size") or "").strip()
                    if official_package:
                        row["package_types"] = official_package
                    elif selector.get("_product_kind") == "module" and module_size:
                        row["package_types"] = f"RF module {module_size} mm"
                    if row.get("package_types"):
                        row["verification_status"] = "manufacturer_product_selector_api"
                        espressif_module_updates += 1
                        package_updated += 1
                        changed = True
                matched_docs = match_espressif_datasheets(selector, espressif_urls)
                for doc_url in matched_docs:
                    selector_name = str(selector.get("name") or name)
                    series_name = str(selector.get("series") or "")
                    title = (
                        f"{selector_name} Datasheet"
                        if normalized_product(selector_name) in normalized_product(doc_url)
                        else f"{series_name or selector_name} Series Datasheet"
                    )
                    add_doc(
                        row,
                        title,
                        doc_url,
                        kind="datasheet",
                        verification_status="official_sitemap",
                    )
                    doc_source_id = f"espressif:datasheet:{slug(doc_url)}"
                    row["source_id"] = ";".join(dict.fromkeys([
                        value for value in [*row.get("source_id", "").split(";"), doc_source_id] if value
                    ]))
                    added_sources[doc_source_id] = {
                        "source_id": doc_source_id,
                        "source_type": "manufacturer_datasheet",
                        "publisher": "Espressif",
                        "title": title,
                        "url": doc_url,
                        "version": "live",
                        "observed_at": utc_now(),
                        "verification_scope": "Document URL listed in Espressif's official technical-file sitemap.",
                    }
                if row.get("documents_json", "") != original_documents:
                    espressif_document_updates += 1
                    changed = True

        if not row.get("package_types") and official_parts_by_device.get(row.get("device_id", "")):
            package_values = sorted({
                part["package_name"].strip()
                for part in official_parts_by_device[row["device_id"]]
                if part.get("package_name", "").strip()
            })
            if package_values:
                row["package_types"] = ";".join(package_values)
                pins = package_pin_counts(package_values)
                if pins and not row.get("pin_counts"):
                    row["pin_counts"] = ";".join(pins)
                row["verification_status"] = "manufacturer_orderable_part_package_metadata"
                orderable_package_updates += 1
                package_updated += 1
                changed = True

        if original_package and row.get("package_types") != original_package and row.get("manufacturer") != "Artery":
            package_corrected += 1
        if changed:
            updated += 1

    for part in orderable_parts:
        if part.get("manufacturer") != "Artery" or part.get("device_id") not in artery_packages_by_device:
            continue
        resolved = artery_packages_by_device[part["device_id"]]
        part["package_name"] = resolved[0] if resolved else ""
        if resolved:
            part["verification_status"] = "manufacturer_datasheet"
            part["decode_status"] = "exact_official_model;datasheet_package_code"
        else:
            part["decode_status"] = "exact_official_model;current_datasheet_variant_not_listed"

    source_map = {row.get("source_id", ""): row for row in sources}
    for key, value in added_sources.items():
        source_map[key] = value
    sources = sorted(source_map.values(), key=lambda row: row.get("source_id", ""))
    write_csv(data / "device-variants.csv", list(devices[0].keys()), devices)
    if orderable_parts:
        write_csv(data / "orderable-parts.csv", list(orderable_parts[0].keys()), orderable_parts)
    if sources:
        write_csv(data / "sources.csv", list(sources[0].keys()), sources)
    missing_packages = [row for row in devices if not row.get("package_types")]
    missing_documents = [row for row in devices if not row.get("documents_json") or row.get("documents_json") == "[]"]
    report = {
        "generated_at": utc_now(), "data_dir": str(data), "datasheet_entries": len(unique_entries),
        "downloaded_documents": sum(item.get("status") == 200 for item in documents),
        "package_mappings": len(mappings), "updated_devices": updated,
        "package_updates": package_updated,
        "package_corrections": package_corrected,
        "artery_document_updates": artery_document_updates,
        "artery_package_tables": len(artery_package_maps),
        "artery_unresolved_packages": sorted(artery_unresolved_packages),
        "espressif_selector_matches": sum(
            row.get("manufacturer") == "Espressif" and row.get("device_name", "").upper() in selector_by_mpn
            for row in devices
        ),
        "espressif_datasheets_in_sitemap": len(espressif_urls),
        "espressif_document_updates": espressif_document_updates,
        "espressif_module_package_updates": espressif_module_updates,
        "orderable_package_updates": orderable_package_updates,
        "remaining_missing_packages": len(missing_packages),
        "remaining_missing_documents": len(missing_documents),
        "remaining_missing_packages_by_manufacturer": dict(sorted(
            ((manufacturer, sum(row.get("manufacturer") == manufacturer for row in missing_packages))
             for manufacturer in {row.get("manufacturer", "") for row in missing_packages}),
            key=lambda value: (-value[1], value[0]),
        )),
        "index_errors": index_errors,
        "artery_errors": artery_errors,
        "espressif_errors": espressif_errors,
        "documents": documents,
    }
    (data / "package-augmentation-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "datasheet_entries", "downloaded_documents", "package_mappings", "updated_devices",
        "package_updates", "package_corrections", "artery_document_updates",
        "espressif_document_updates", "espressif_module_package_updates",
        "orderable_package_updates", "remaining_missing_packages", "remaining_missing_documents",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
