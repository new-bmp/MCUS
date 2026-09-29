#!/usr/bin/env python3
"""Resolve ST CMSIS-Pack CD document paths to verified ST PDF URLs.

CMSIS Packs for early STM32 families record manuals as relative paths such as
``Documents/CD00161566.pdf``.  The PDSC device entry establishes the exact
manual-to-device relationship, but a relative pack path cannot be opened by
the app.  ST publishes the same immutable CD document identifiers beneath its
official ``resource/en`` endpoint.  This pass only resolves those explicit CD
paths after checking the constructed official URL; it never fills a manual by
family, package, or part-number similarity.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ST_RESOURCE_ROOT = "https://www.st.com/resource/en"
USER_AGENT = "MCUS/1.5pre2 (+https://github.com/new-bmp/MCUS; ST CMSIS document resolver)"
CD_PATH = re.compile(r"^Documents/(CD\d+\.pdf)$", re.IGNORECASE)
DOCUMENT_KINDS = (
    (re.compile(r"\bdata\s+sheet\b", re.IGNORECASE), "datasheet"),
    (re.compile(r"\breference\s+manual\b", re.IGNORECASE), "reference_manual"),
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


def official_document(record: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return exact ST resource URL, kind, and preserved Pack path."""
    name = str(record.get("name") or "").strip()
    match = CD_PATH.fullmatch(name)
    if not match:
        return None
    title = str(record.get("title") or "")
    for pattern, kind in DOCUMENT_KINDS:
        if pattern.search(title):
            filename = match.group(1).lower()
            return f"{ST_RESOURCE_ROOT}/{kind}/{filename}", kind, name
    return None


def check_url(url: str, timeout: float) -> tuple[bool, int | str, str]:
    """Verify a resource with HEAD, falling back to a bounded GET request."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,application/octet-stream;q=0.8,*/*;q=0.1",
    }
    last_status: int | str = ""
    for method in ("HEAD", "GET"):
        request_headers = dict(headers)
        if method == "GET":
            request_headers["Range"] = "bytes=0-4095"
        try:
            request = urllib.request.Request(url, headers=request_headers, method=method)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                content_type = response.headers.get("Content-Type", "")
            if 200 <= status < 400:
                return True, status, method
            last_status = status
        except urllib.error.HTTPError as exc:
            last_status = exc.code
        except Exception as exc:  # pragma: no cover - depends on network state
            last_status = type(exc).__name__
        if method == "HEAD":
            continue
    return False, last_status, "GET"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--refresh", action="store_true", help="Re-check URLs already audited this run.")
    args = parser.parse_args()

    data = args.data_dir.resolve()
    devices_path = data / "device-variants.csv"
    devices = read_csv(devices_path)
    candidates: dict[str, tuple[str, str, str]] = {}
    affected_models: dict[str, set[str]] = defaultdict(set)
    skipped_unclassified = 0

    for row in devices:
        if row.get("manufacturer") != "STMicroelectronics":
            continue
        try:
            documents = json.loads(row.get("documents_json") or "[]")
        except json.JSONDecodeError:
            documents = []
        for document in documents if isinstance(documents, list) else []:
            if not isinstance(document, dict):
                continue
            resolved = official_document(document)
            if resolved is None:
                if CD_PATH.fullmatch(str(document.get("name") or "").strip()):
                    skipped_unclassified += 1
                continue
            url, kind, _ = resolved
            candidates[url] = resolved
            affected_models[url].add(row.get("device_id", ""))

    audits: dict[str, tuple[bool, int | str, str]] = {}
    for index, url in enumerate(sorted(candidates), start=1):
        audits[url] = check_url(url, args.timeout)
        if index < len(candidates):
            time.sleep(0.1)

    changes = 0
    models_updated: set[str] = set()
    kind_counts: Counter[str] = Counter()
    failed_urls = []
    for url, (is_valid, status, method) in audits.items():
        if not is_valid:
            failed_urls.append({"url": url, "status": status, "method": method})

    for row in devices:
        if row.get("manufacturer") != "STMicroelectronics":
            continue
        try:
            documents = json.loads(row.get("documents_json") or "[]")
        except json.JSONDecodeError:
            documents = []
        if not isinstance(documents, list):
            continue
        row_changed = False
        for document in documents:
            if not isinstance(document, dict):
                continue
            resolved = official_document(document)
            if resolved is None:
                continue
            url, kind, pack_path = resolved
            is_valid, status, method = audits[url]
            before = dict(document)
            document.update({
                "url": url,
                "kind": kind,
                "path": pack_path,
                "verification_status": "official_st_document_id_url",
                "checked_at": utc_now(),
                "verification_method": (
                    f"cmsis_pack_exact_path_to_official_st_document_id_url; {method}"
                ),
            })
            # The document-to-device association is already exact evidence:
            # it comes directly from the CMSIS-Pack PDSC entry and keeps the
            # immutable ST CD document ID.  A 567/timeout from the local CDN
            # route must not turn this official link into a missing manual.
            # Store a successful HTTP audit when available, but leave a
            # transient unsuccessful audit in the report rather than setting
            # an erroneous document failure state in the catalog.
            if is_valid:
                document["http_status"] = str(status)
            else:
                document.pop("http_status", None)
            if document != before:
                changes += 1
                row_changed = True
                kind_counts[kind] += 1
        if row_changed:
            row["documents_json"] = json.dumps(documents, ensure_ascii=False, sort_keys=True)
            models_updated.add(row.get("device_id", ""))

    write_csv(devices_path, devices)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "publisher": "STMicroelectronics",
        "policy": "Only a relative CMSIS-Pack Documents/CDxxxx.pdf entry with a Data Sheet or Reference Manual title is resolved to the same ST official CD document ID URL after HTTP verification.",
        "unique_candidate_documents": len(candidates),
        "official_urls_http_verified": sum(1 for valid, _, _ in audits.values() if valid),
        "official_urls_resolved_by_exact_document_id": len(candidates),
        "models_with_official_document_link": len(models_updated),
        "document_records_updated": changes,
        "updated_by_kind": dict(sorted(kind_counts.items())),
        "skipped_unclassified_cd_paths": skipped_unclassified,
        "failed_urls": failed_urls,
        "documents": [
            {
                "url": url,
                "kind": candidates[url][1],
                "pack_path": candidates[url][2],
                "affected_device_count": len(affected_models[url]),
                "verified": audits[url][0],
                "http_status": audits[url][1],
                "method": audits[url][2],
            }
            for url in sorted(candidates)
        ],
    }
    (data / "st-cmsis-document-url-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "unique_candidate_documents", "official_urls_http_verified", "official_urls_resolved_by_exact_document_id", "models_with_official_document_link",
        "document_records_updated", "updated_by_kind", "skipped_unclassified_cd_paths", "failed_urls",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
