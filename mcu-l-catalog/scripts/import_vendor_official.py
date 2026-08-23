#!/usr/bin/env python3
"""Run a bounded, auditable CMSIS-Pack import for one manufacturer.

CMSIS-Pack is treated as a device-metadata source only. The adapter writes an
explicit scope report and never claims that a vendor's entire catalog or all
orderable suffixes are present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


VENDORS = {
    "st": ("STMicroelectronics", "stm32"),
    "ti": ("TexasInstruments", "all-dfp"),
    "microchip": ("Microchip", "all-dfp"),
    "gigadevice": ("GigaDevice", "all-dfp"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("vendor", choices=sorted(VENDORS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    display_vendor, scope = VENDORS[args.vendor]
    default_dir = "texas-instruments" if args.vendor == "ti" else args.vendor
    output_dir = args.output_dir or root / "data" / "vendor-packs" / default_dir
    cache_dir = args.cache_dir or root / "cache" / args.vendor
    command = [
        sys.executable,
        str(root / "scripts" / "import_cmsis_packs.py"),
        "--scope", scope,
        "--output-dir", str(output_dir),
        "--cache-dir", str(cache_dir),
        "--workers", str(args.workers),
        "--timeout", str(args.timeout),
    ]
    # The public CMSIS index historically publishes STM32 packs under both
    # Keil and ST names. Scope filtering is the authoritative ST boundary;
    # exact vendor filtering would silently drop the Keil-published packs.
    if args.vendor != "st":
        command[command.index("--output-dir"):command.index("--output-dir")] = ["--vendor", display_vendor]
    allowed_hosts = {
        "st": {"developer.st.com", "www.keil.com"},
        "ti": {"software-dl.ti.com", "www.keil.com"},
        "microchip": {"packs.download.microchip.com", "www.keil.com"},
        "gigadevice": {"gd32mcu.com", "www.keil.com"},
    }[args.vendor]
    for host in sorted(allowed_hosts):
        command.extend(["--allowed-host", host])
    if args.refresh:
        command.append("--refresh")
    completed = subprocess.run(command, cwd=root, check=False)

    sources = read_csv(output_dir / "sources.csv")
    errors = read_csv(output_dir / "import-errors.csv")
    devices = read_csv(output_dir / "device-variants.csv")
    selected_sources = []
    source_errors: list[str] = []
    for source in sources:
        parsed = urlparse(source.get("url", ""))
        if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
            source_errors.append(f"unapproved_source:{source.get('url', '')}")
        else:
            selected_sources.append(source["source_id"])

    report = {
        "status": "ok" if completed.returncode == 0 and not source_errors else "partial",
        "adapter": f"import_{args.vendor}_official.py",
        "manufacturer": "STMicroelectronics" if args.vendor == "st" else (
            "Texas Instruments" if args.vendor == "ti" else display_vendor
        ),
        "scope": scope,
        "output_dir": str(output_dir.resolve()),
        "official_source_ids": sorted(set(selected_sources)),
        "official_https_hosts": sorted(allowed_hosts),
        "device_variants": len(devices),
        "sources": len({source.get("source_id", "") for source in sources if source.get("source_id")}),
        "import_errors": len(errors),
        "source_errors": source_errors,
        "subprocess_exit_code": completed.returncode,
        "snapshot_sha256": {
            str(path.relative_to(cache_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(cache_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".pdsc", ".pidx"}
        } if cache_dir.exists() else {},
        "completeness_claim": "All successfully parsed records from the selected official CMSIS-Pack scope at observation time; not a claim about unpublished products, non-Arm families, or orderable-code completeness.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official-adapter-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
