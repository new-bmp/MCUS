#!/usr/bin/env python3
"""Import a verified subset of TI C2000 real-time DSP controllers.

The C2000 family is not an Arm MCU.  Records therefore use ``C28x DSP`` as
the architecture and ``dsp_mcu`` as the product type.  Values in this file
come from the exact TI data sheet linked on each record; fields not stated by
that data sheet are deliberately left empty.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vendor_import_common import DEVICE_FIELDS, PART_FIELDS, SOURCE_FIELDS, write_csv


MANUFACTURER = "Texas Instruments"
WARE_SOURCE_ID = "ti:c2000ware-core-sdk"
WARE_URL = "https://raw.githubusercontent.com/TexasInstruments/c2000ware-core-sdk/main/README.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def feature(
    kind: str,
    name: str,
    count: int | str | None = None,
    *,
    n: int | str | None = None,
    m: int | str | None = None,
) -> dict[str, str]:
    item: dict[str, str] = {
        "type": kind,
        "name": name,
        "source_kind": "ti_datasheet",
        "verification_status": "manufacturer_datasheet",
    }
    if count is not None:
        item["count"] = str(count)
    if n is not None:
        item["n"] = str(n)
    if m is not None:
        item["m"] = str(m)
    return item


def memory_regions(flash: int, ram: int, flash_note: str, ram_note: str) -> list[dict[str, str]]:
    return [
        {"name": f"Flash ({flash_note})", "size": str(flash), "access": "rx"},
        {"name": f"SARAM ({ram_note})", "size": str(ram), "access": "rwx"},
    ]


def docs(model: str, datasheet: str) -> list[dict[str, str]]:
    return [
        {
            "title": f"TI {model} data sheet",
            "url": datasheet,
            "kind": "datasheet",
            "verification_status": "manufacturer_datasheet",
        },
        {
            "title": f"TI {model} official product page",
            "url": f"https://www.ti.com/product/{model}",
            "kind": "product_page",
            "verification_status": "manufacturer_product_page",
        },
        {
            "title": "TI C2000Ware SDK product-family index",
            "url": WARE_URL,
            "kind": "source_pack",
            "verification_status": "official_family_index",
        },
    ]


def model(
    name: str,
    series: str,
    line: str,
    clock: int,
    flash: int,
    ram: int,
    packages: str,
    pins: str,
    adc: tuple[int, int, int, str],
    pwm: int,
    timers: int,
    gpio: int,
    spi: int,
    i2c: int,
    uart: int,
    can: int,
    *,
    can_fd: bool = False,
    fpu: bool = True,
    core_count: int = 1,
    dac: int = 0,
    usb: int = 0,
    eth: int = 0,
    dma: int = 0,
    cla: int = 0,
    vcu: str = "",
    emif: int = 0,
    mcbsp: int = 0,
    flash_note: str = "ECC-protected flash",
    ram_note: str = "on-chip SARAM",
    datasheet: str,
) -> dict[str, Any]:
    adc_units, adc_channels, adc_bits, adc_rate = adc
    processors = [
        {
            "Dclock": str(clock),
            "Dcore": "C28x DSP",
            "DcoreVersion": "C28x",
            "Ddsp": "yes",
            "Dfpu": "yes" if fpu else "no",
        }
        for _ in range(core_count)
    ]
    if name == "TMS320F28388D":
        processors.append({"Dclock": "125000000", "Dcore": "Cortex-M4 connectivity manager", "Ddsp": "no", "Dfpu": "yes"})
    features = [
        feature("VCC", "Single 3.3-V supply", n="3.0", m="3.6"),
        feature("Timer", "32-bit CPU timers", timers),
        feature("PWM", f"Enhanced PWM channels ({pwm} channels)", pwm),
        feature("ADCUnits", f"ADC converter units ({adc_units})", adc_units),
        feature("ADC", f"{adc_channels} external ADC channels, {adc_bits}-bit, {adc_rate}", adc_channels),
        feature("IOs", f"General-purpose I/O ({gpio} pins)", gpio),
        feature("SPI", f"Serial Peripheral Interface ({spi} ports)", spi),
        feature("I2C", f"Inter-integrated Circuit ({i2c} interfaces)", i2c),
        feature("UART", f"SCI / UART-compatible serial interfaces ({uart})", uart),
        feature("CAN", f"CAN{' FD/MCAN' if can_fd else ''} bus ports ({can})", can),
        feature("DMA", f"6-channel DMA controllers ({dma})", dma) if dma else None,
        feature("DAC", f"Buffered DAC outputs ({dac})", dac) if dac else None,
        feature("USB", "USB 2.0 controller (MAC + PHY)", usb) if usb else None,
        feature("ETH", "Ethernet interface", eth) if eth else None,
        feature("ExtBus", f"External memory interface ({emif})", emif) if emif else None,
        feature("ComOther", f"Multichannel buffered serial ports (McBSP) ({mcbsp})", mcbsp) if mcbsp else None,
        feature("Accelerator", "IEEE 754 single-precision FPU", 1) if fpu else None,
        feature("Accelerator", "Trigonometric Math Unit (TMU)", 1),
        feature("Accelerator", "Viterbi / Complex Math Unit (VCU)", 1) if vcu else None,
        feature("Accelerator", f"Control Law Accelerator (CLA), {cla} independent core(s)", cla) if cla else None,
        feature("Security", "Dual-zone code security module", 1),
    ]
    features = [item for item in features if item is not None]
    device_id = f"texas-instruments::{name.lower()}"
    line_id = f"texas-instruments::c2000::{series.lower()}::{line.lower()}"
    source_id = f"ti:datasheet:{name.lower()}"
    return {
        "device_id": device_id,
        "product_line_id": line_id,
        "manufacturer": MANUFACTURER,
        "product_type": "dsp_mcu",
        "architecture_class": "C28x DSP",
        "family": "C2000",
        "series": series,
        "product_line": line,
        "device_name": name,
        "generic_device_name": line,
        "manufacturer_variant_code": "",
        "processor_cores": json.dumps(processors, ensure_ascii=False),
        "max_clock_hz": str(clock),
        "flash_bytes": str(flash),
        "ram_bytes": str(ram),
        "package_types": packages,
        "pin_counts": pins,
        "memory_regions_json": json.dumps(memory_regions(flash, ram, flash_note, ram_note), ensure_ascii=False),
        "features_json": json.dumps(features, ensure_ascii=False),
        "documents_json": json.dumps(docs(name, datasheet), ensure_ascii=False),
        "svd_files": "",
        "lifecycle": "active",
        "source_id": f"{WARE_SOURCE_ID};{source_id}",
        "source_url": datasheet,
        "source_version": "C2000Ware main; data-sheet",
        "observed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "verification_status": "manufacturer_datasheet",
    }


def records() -> list[dict[str, Any]]:
    # Exact values are taken from the linked TI data sheets.  The model is a
    # catalog anchor for each family; sibling suffix/package variants remain
    # unknown until their own official records are imported.
    return [
        model("TMS320F280025C", "F28002x", "TMS320F28002x", 100_000_000, 131072, 24576, "LQFP", "48;64;80", (2, 16, 12, "3.45 MSPS"), 14, 3, 43, 2, 2, 1, 1, cla=0, datasheet="https://www.ti.com/lit/ds/symlink/tms320f280025c.pdf"),
        model("TMS320F2800137", "F280013x", "TMS320F280013x", 120_000_000, 262144, 36864, "LQFP;VQFN", "", (2, 21, 12, "4 MSPS"), 14, 3, 38, 1, 2, 3, 1, cla=0, datasheet="https://www.ti.com/lit/ds/symlink/tms320f2800137.pdf"),
        model("TMS320F2800157", "F280015x", "TMS320F280015x", 120_000_000, 262144, 36864, "LQFP;VQFN", "", (2, 21, 12, "4 MSPS"), 14, 3, 52, 1, 2, 3, 2, can_fd=True, core_count=2, cla=0, datasheet="https://www.ti.com/lit/ds/symlink/tms320f2800157.pdf"),
        model("TMS320F280039C", "F28003x", "TMS320F28003x", 120_000_000, 393216, 70656, "LQFP", "48;64;80;100", (3, 21, 12, "4 MSPS"), 16, 3, 80, 2, 2, 2, 2, can_fd=True, cla=1, datasheet="https://www.ti.com/lit/ds/symlink/tms320f280039c.pdf"),
        model("TMS320F280049C", "F28004x", "TMS320F28004x", 100_000_000, 262144, 102400, "LQFP;VQFN", "56;64;100", (3, 21, 12, "3.45 MSPS"), 16, 3, 40, 2, 1, 2, 2, dac=2, cla=1, vcu="VCU", datasheet="https://www.ti.com/lit/ds/symlink/tms320f280049c.pdf", flash_note="two independent ECC-protected banks"),
        model("TMS320F28069", "F2806x", "TMS320F2806x", 90_000_000, 262144, 102400, "LQFP;HTQFP", "80;100", (1, 16, 12, "3.46 MSPS"), 16, 3, 54, 2, 1, 2, 1, usb=1, dma=1, cla=1, vcu="VCU", mcbsp=1, datasheet="https://www.ti.com/lit/ds/symlink/tms320f28069.pdf"),
        model("TMS320F28075", "F2807x", "TMS320F2807x", 120_000_000, 524288, 102400, "HTQFP", "100;176", (3, 26, 12, "3.5 MSPS"), 24, 3, 169, 3, 2, 4, 2, usb=1, dma=2, cla=1, mcbsp=2, datasheet="https://www.ti.com/lit/ds/symlink/tms320f28075.pdf"),
        model("TMS320F28335", "F2833x", "TMS320F2833x", 150_000_000, 524288, 69632, "BGA;LQFP;HLQFP", "176;179", (1, 16, 12, "12.5 MSPS"), 18, 3, 88, 1, 1, 3, 2, dma=1, mcbsp=2, emif=1, fpu=True, datasheet="https://www.ti.com/lit/ds/symlink/tms320f28335.pdf"),
        model("TMS320F28379D", "F2837xD", "TMS320F2837xD", 200_000_000, 1_048_576, 208896, "nFBGA;HLQFP", "176;337", (4, 12, 16, "1.1 MSPS (16-bit); 3.5 MSPS (12-bit)"), 24, 3, 169, 3, 2, 4, 2, core_count=2, dac=3, usb=1, dma=2, cla=2, emif=2, mcbsp=2, datasheet="https://www.ti.com/lit/ds/symlink/tms320f28379d.pdf", flash_note="ECC-protected flash", ram_note="dedicated, local shared and global shared RAM"),
        model("TMS320F28388D", "F2838x", "TMS320F2838x", 200_000_000, 1_572_864, 319488, "nFBGA;HLQFP", "176;337", (4, 12, 16, "1.1 MSPS (16-bit); 3.5 MSPS (12-bit)"), 32, 3, 169, 4, 2, 4, 2, can_fd=True, core_count=2, dac=3, usb=1, eth=1, dma=2, cla=2, emif=2, mcbsp=2, datasheet="https://www.ti.com/lit/ds/symlink/tms320f28388d.pdf", flash_note="two 512KB C28x banks plus 512KB CM flash", ram_note="44KB local RAM per C28x, 128KB shared RAM, 96KB CM RAM"),
        model("TMS320F28035", "F2803x", "TMS320F2803x", 60_000_000, 131072, 20480, "VQFN;TQFP;LQFP", "56;64;80", (1, 16, 12, "4.6 MSPS"), 14, 3, 45, 2, 1, 1, 1, cla=1, fpu=False, datasheet="https://www.ti.com/lit/ds/symlink/tms320f28035.pdf"),
        model("TMS320F28027", "F2802x", "TMS320F2802x", 60_000_000, 65536, 12288, "TSSOP;LQFP", "38;48", (1, 13, 12, "4.6 MSPS"), 8, 3, 22, 1, 1, 1, 0, fpu=False, datasheet="https://www.ti.com/lit/ds/symlink/tms320f28027.pdf", flash_note="32K x 16 flash", ram_note="6K x 16 SARAM"),
    ]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "vendor-packs" / "texas-instruments")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    existing = read_csv(out / "device-variants.csv")
    additions = records()
    by_id = {row.get("device_id", ""): row for row in existing if row.get("device_id")}
    for row in additions:
        by_id[row["device_id"]] = row
    devices = list(by_id.values())
    write_csv(out / "device-variants.csv", DEVICE_FIELDS, devices)

    old_sources = {row.get("source_id", ""): row for row in read_csv(out / "sources.csv") if row.get("source_id")}
    old_sources[WARE_SOURCE_ID] = {
        "source_id": WARE_SOURCE_ID,
        "source_type": "official_family_index",
        "publisher": MANUFACTURER,
        "title": "TI C2000Ware core SDK product-family index",
        "url": WARE_URL,
        "version": "main",
        "observed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "verification_scope": "Official TI C2000Ware README lists C2000 product lines; exact values come from each linked data sheet.",
    }
    for row in additions:
        source_id = f"ti:datasheet:{row['device_name'].lower()}"
        old_sources[source_id] = {
            "source_id": source_id,
            "source_type": "manufacturer_datasheet",
            "publisher": MANUFACTURER,
            "title": f"TI {row['device_name']} data sheet",
            "url": row["source_url"],
            "version": "live",
            "observed_at": row["observed_at"],
            "verification_scope": "Exact TI data sheet for the named C2000 device; specifications are not inferred from sibling variants.",
        }
    write_csv(out / "sources.csv", SOURCE_FIELDS, sorted(old_sources.values(), key=lambda row: row.get("source_id", "")))

    lines: dict[str, dict[str, Any]] = {}
    for row in devices:
        key = row["product_line_id"]
        item = lines.setdefault(key, {
            "product_line_id": key,
            "manufacturer": MANUFACTURER,
            "product_type": row["product_type"],
            "architecture_class": row["architecture_class"],
            "family": row["family"],
            "series": row["series"],
            "product_line": row["product_line"],
            "device_variant_count": 0,
            "source_ids": set(),
            "observed_at": row["observed_at"],
            "verification_status": "manufacturer_datasheet",
        })
        item["device_variant_count"] += 1
        item["source_ids"].update(row["source_id"].split(";"))
    line_rows = []
    for item in lines.values():
        item["source_ids"] = ";".join(sorted(item["source_ids"]))
        line_rows.append(item)
    line_fields = ["product_line_id", "manufacturer", "product_type", "architecture_class", "family", "series", "product_line", "device_variant_count", "source_ids", "observed_at", "verification_status"]
    write_csv(out / "product-lines.csv", line_fields, sorted(line_rows, key=lambda row: row["product_line"]))

    manifest = [{
        "manufacturer": MANUFACTURER,
        "source_scope": "C2000Ware plus exact TI data sheets",
        "pack_count": len({source for row in additions for source in row["source_id"].split(";")}),
        "family_count": len({row["family"] for row in additions}),
        "series_count": len({row["series"] for row in additions}),
        "product_line_count": len(lines),
        "device_variant_count": len(devices),
        "orderable_part_count": len(read_csv(out / "orderable-parts.csv")),
        "last_observed": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "device_coverage_status": "official_c2000_datasheet_anchor_models",
        "orderable_coverage_status": "partial_official_sources",
        "notes": "Twelve exact C2000 anchor models imported from TI data sheets; sibling purchasing-code variants are not fabricated.",
    }]
    write_csv(out / "coverage-manifest.csv", list(manifest[0]), manifest)
    print(json.dumps({"added": len(additions), "total_ti_devices": len(devices), "product_lines": len(lines)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
