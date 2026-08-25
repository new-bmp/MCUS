#!/usr/bin/env python3
"""Build a small-file static deployment bundle from the Android web assets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ASSIGNMENT_PREFIX = "window.MCU_CATALOG="


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=300)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    catalog_path = source / "catalog.js"
    index_path = source / "index.html"
    encoded = catalog_path.read_text(encoding="utf-8").strip()
    if not encoded.startswith(ASSIGNMENT_PREFIX) or not encoded.endswith(";"):
        raise ValueError(f"Unexpected catalog wrapper: {catalog_path}")

    catalog = json.loads(encoded[len(ASSIGNMENT_PREFIX):-1])
    devices = catalog.pop("devices")
    expected = int(catalog.get("meta", {}).get("devices", len(devices)))
    if len(devices) != expected:
        raise ValueError(f"Catalog device count mismatch: {len(devices)} != {expected}")

    output.mkdir(parents=True, exist_ok=True)
    for old_part in output.glob("catalog-part-*.js"):
        old_part.unlink()

    catalog["devices"] = []
    (output / "catalog-base.js").write_text(
        ASSIGNMENT_PREFIX + compact(catalog) + ";\n", encoding="utf-8"
    )

    part_names: list[str] = []
    for offset in range(0, len(devices), args.chunk_size):
        part = devices[offset:offset + args.chunk_size]
        name = f"catalog-part-{offset // args.chunk_size + 1:03d}.js"
        (output / name).write_text(
            "window.MCU_CATALOG.devices=window.MCU_CATALOG.devices.concat("
            + compact(part)
            + ");\n",
            encoding="utf-8",
        )
        part_names.append(name)

    version = str(catalog.get("meta", {}).get("version", "current"))
    scripts = [f'  <script src="catalog-base.js?v={version}"></script>']
    scripts.extend(f'  <script src="{name}?v={version}"></script>' for name in part_names)
    scripts.append(
        "  <script>"
        f"if(!window.MCU_CATALOG||window.MCU_CATALOG.devices.length!=={expected})"
        "{document.querySelector('.splash-sub').textContent='目录分片载入不完整，请确认已上传全部文件';}"
        "</script>"
    )
    scripts.append(
        f'  <script src="quote-config.js?v={version}"></script>'
    )
    scripts.append(
        f'  <script src="app.js?v={version}" '
        "onerror=\"document.querySelector('.splash-sub').textContent='界面脚本载入失败，请确认已上传 app.js';\"></script>"
    )

    index = index_path.read_text(encoding="utf-8")
    index = re.sub(
        r'\s*<script src="catalog\.js(?:\?[^\"]*)?"></script>\s*'
        r'(?:<script src="quote-config\.js(?:\?[^\"]*)?"></script>\s*)?'
        r'<script src="app\.js(?:\?[^\"]*)?"></script>',
        "\n" + "\n".join(scripts),
        index,
        count=1,
    )
    if "catalog-base.js" not in index:
        raise ValueError("Could not replace catalog script references in index.html")
    (output / "index.html").write_text(index, encoding="utf-8")

    largest = max((path.stat().st_size for path in output.glob("catalog-*.js")), default=0)
    print(
        json.dumps(
            {
                "devices": len(devices),
                "parts": len(part_names),
                "largest_part_bytes": largest,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
