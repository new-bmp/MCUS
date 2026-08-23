#!/usr/bin/env python3
"""Import Allwinner/XRadio standalone MCU records from official documentation.

This adapter deliberately keeps application processors, real-time SoCs and
wireless connectivity chips out of the MCU catalog.  Only the eight XRadio
parts below are imported here; a future SoC catalog can reuse the retained
reference table without presenting those products as ordinary MCUs.
Unknown values remain empty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vendor_import_common import (
    DEVICE_FIELDS,
    ERROR_FIELDS,
    PART_FIELDS,
    SOURCE_FIELDS,
    OfficialFetcher,
    slug,
    utc_now,
    write_csv,
    write_json,
)


MANUFACTURER = "Allwinner"
PRODUCT_LINE_FIELDS = [
    "product_line_id", "manufacturer", "product_type", "architecture_class",
    "family", "series", "product_line", "device_variant_count", "source_ids",
    "observed_at", "verification_status",
]

PRODUCT_PAGE = "https://www.allwinnertech.com/index.php?c=product&a=index&id={}"


def product_page_source(model: str, page_id: int, scope: str) -> dict[str, str]:
    return {
        "id": f"allwinner:product-page:{model.lower()}",
        "url": PRODUCT_PAGE.format(page_id),
        "cache": f"{model}_official_product_page.html",
        "title": f"Allwinner {model} official product page",
        "version": "live-product-page",
        "type": "manufacturer_product_page",
        "publisher": "Allwinner",
        "scope": scope,
        "identity": model,
    }


SOURCES = {
    "xr806": {
        "id": "allwinner:xr806:official-docs",
        "url": "https://xr806.docs.aw-ol.com/",
        "cache": "XR806_official_docs.html",
        "title": "XR806 official developer documentation",
        "version": "live-docs",
        "type": "manufacturer_documentation_page",
        "publisher": "Allwinner",
        "scope": "XR806AF2L identity, core, clock, memory, wireless, security, package and process data.",
    },
    "xr808": {
        "id": "allwinner:xradio:xr808-datasheet-v1.1",
        "url": "https://raw.githubusercontent.com/XradioTech/xradiotech.github.io/master/docs/doc/XR808/XR808_Datasheet_V1.1.pdf",
        "cache": "XR808_Datasheet_V1.1.pdf",
        "title": "XR808 Datasheet",
        "version": "1.1",
        "type": "manufacturer_datasheet",
        "publisher": "XRadioTech",
        "scope": "XR808CT0/XR808CT2 exact variants, memory, package and peripheral resources.",
    },
    "xr809": {
        "id": "allwinner:xradio:xr809-datasheet-v1.1",
        "url": "https://raw.githubusercontent.com/XradioTech/xradiotech.github.io/master/docs/doc/XR809/XR809_Datasheet_V1.1.pdf",
        "cache": "XR809_Datasheet_V1.1.pdf",
        "title": "XR809 Datasheet",
        "version": "1.1",
        "type": "manufacturer_datasheet",
        "publisher": "XRadioTech",
        "scope": "XR809 exact device, memory, package and peripheral resources.",
    },
    "xr871": {
        "id": "allwinner:xradio:xr871-datasheet-v1.21",
        "url": "https://raw.githubusercontent.com/XradioTech/xradiotech.github.io/master/docs/doc/XR871/XR871_Datasheet_V1.21.pdf",
        "cache": "XR871_Datasheet_V1.21.pdf",
        "title": "XR871 Datasheet",
        "version": "1.21",
        "type": "manufacturer_datasheet",
        "publisher": "XRadioTech",
        "scope": "XR871GT/XR871ET exact variants, memory, package and peripheral resources.",
    },
    "xr872": {
        "id": "allwinner:xradio:xr872-datasheet-v1.05",
        "url": "https://raw.githubusercontent.com/XradioTech/xradiotech.github.io/master/docs/doc/XR872/XR872_Datasheet_V1.05.pdf",
        "cache": "XR872_Datasheet_V1.05.pdf",
        "title": "XR872 Datasheet",
        "version": "1.05",
        "type": "manufacturer_datasheet",
        "publisher": "XRadioTech",
        "scope": "XR872AT/XR872ET exact variants, memory, package and peripheral resources.",
    },
    "a733": product_page_source("A733", 139, "A733 MCU core, SoC accelerators and interface inventory."),
    "a537": product_page_source("A537", 138, "A537 MCU core, SoC accelerators and interface inventory."),
    "a333": product_page_source("A333", 137, "A333 MCU core, SoC accelerators and interface inventory."),
    "a527": product_page_source("A527", 109, "A527 MCU core, SoC accelerators and interface inventory."),
    "a523": product_page_source("A523", 108, "A523 MCU core, SoC accelerators and interface inventory."),
    "t736": product_page_source("T736", 146, "T736 MCU core, industrial SoC accelerators and interfaces."),
    "t527": product_page_source("T527", 110, "T527 MCU core, industrial SoC accelerators and interfaces."),
    "t536": product_page_source("T536", 135, "T536 dual MCU cores, AMP software and industrial interfaces."),
    "t153": product_page_source("T153", 145, "T153 MCU core, AMP software and industrial interfaces."),
    "mr153": product_page_source("MR153", 149, "MR153 real-time MCU core and robot-control interfaces."),
    "mr527": product_page_source("MR527", 121, "MR527 independently usable MCU core and robot interfaces."),
    "mr536": product_page_source("MR536", 141, "MR536 independent MCU system and robot interfaces."),
    "v881": product_page_source("V881", 150, "V881 MCU core, vision accelerators and interfaces."),
    "v861": product_page_source("V861", 148, "V861 MCU core, vision accelerators and interfaces."),
    "v838": product_page_source("V838", 147, "V838 MCU core, vision accelerators and interfaces."),
    "v821": product_page_source("V821", 136, "V821 MCU core, wireless/video functions and interfaces."),
    "v853": product_page_source("V853", 117, "V853 MCU core, vision accelerators and interfaces."),
    "v851s": product_page_source("V851s", 118, "V851s MCU core, vision accelerators and interfaces."),
    "v837s": product_page_source("V837s", 105, "V837s MCU core, vision accelerators and interfaces."),
    "r128": product_page_source("R128", 133, "R128 MCU, DSP, memory, wireless and peripheral resources."),
    "xr_connectivity": product_page_source(
        "XR819", 134, "XR819/XR819s/XR829 wireless connectivity selector rows."
    ),
}

# The public product site uses a JavaScript WAF for non-browser clients. These
# hashes cover the rendered DOM snapshots reviewed on 2026-08-22; they prevent
# the WAF challenge page from being mistaken for product data.
PRODUCT_PAGE_EVIDENCE_SHA256 = {
    "a733": "aed85038b26fd3d808793b311a9dbdbf42e68a580e5fc8c08850aec8bbc48b6a",
    "a537": "553dd502c1e1891c4b8e3a496c353f6c9238258efb0d597a20646b56a44afadc",
    "a333": "56d62ff46617e1c25ed710de7d1e616ec1a912fabec9730f9344a57ed6b6661a",
    "a527": "e0e320197aaa3489c29e6e483a85831ae8225357a7fcd31ecebfd084b0c3eec0",
    "a523": "c6c661cdfd81dce45ccbdf4c7e04bf3e96503d4b211c24cb1b4105a907fe795d",
    "t736": "dd6ffa9f589bd270d65bba84cf400f98c4753ea7499a494d497dc239fb6f8f0d",
    "t527": "5ac0b025992f6a2aee8af6a68faa59a7be489b141952ed53854b80290731861d",
    "t536": "6d72948e4709548e8f7d719fa73655bf876742d198ef3b2ff3505d045b9e4236",
    "t153": "abb569f1dbe50d63da5457bd61ade30a6e721b09cc30e9a011b1802e95988e65",
    "mr153": "5522f1be69ef936cf5af93572effe5a6cac55d8b2bf4e6a05394128e53a30be7",
    "mr527": "3e0ddc8f365d95205ecc625c5bc6d0c4b686bb078339c299ada92ea3983dbe93",
    "mr536": "95686d4f2106b588f25c4a2615b2d341a9d9935d8c1d3e13805221aa834b7891",
    "v881": "209fb43d44a0ff2bdb2a09e49050416b506698649131645d7e4740323f0fa8b1",
    "v861": "03fc83c37a78f24203f4361b943f2a2f71bda60a6745e9ae528879fd72d972ad",
    "v838": "8293dc800904ec423504b4384442c9312159934e740093ebe8086bb80435a038",
    "v821": "5622e52d514fb25d062c0a26ac271635ff015908c8172ce4209d65a679351847",
    "v853": "d05fa2cac01913b06ce495b8398b0b81b6faf65b52c9c911ab8fb9191ba8c4ad",
    "v851s": "add3afb346780232ae8047606bed8b9b7c017977f83c24dec39b5f0063872f58",
    "v837s": "bdee757cbe3393dd491fbe0726abfbd0d9213e5ca11e43f8c566a501b953129a",
    "r128": "353c9b548399790673a33a38cd869c3f558f045d321957068f5e31b778c116bd",
    "xr_connectivity": "d949c633401ccdb135dedb08682e9823045ee05d9823efa2983af9703d1bec46",
}


def feature(kind: str, name: str, count: int | None = None, detail: str = "") -> dict[str, str]:
    item = {
        "type": kind,
        "name": name,
        "source_kind": "allwinner_official_datasheet",
    }
    if count is not None:
        item["count"] = str(count)
        item["n"] = str(count)
    if detail:
        item["m"] = detail
    return item


def m4_processor() -> list[dict[str, str]]:
    return [{
        "Darchitecture": "ARMv7E-M",
        "Dcore": "Cortex-M4",
        "DcoreCount": "1",
        "Dfpu": "FPU",
        "Dmpu": "MPU",
        "Ddsp": "DSP extension",
    }]


WIFI = feature("WiFi", "2.4 GHz IEEE 802.11 b/g/n Wi-Fi", 1)
CRYPTO = feature("Crypto", "Hardware cryptography: AES, DES, 3DES, SHA, MD5 and CRC", 1)
RNG = feature("RNG", "Hardware true or pseudo random number generator", 1)
RTC = feature("RTC", "Real-time clock", 1)
WDT = feature("WDT", "Watchdog timer", 1)
IR = feature("VendorCapability", "XRadio infrared transmitter and receiver", 1)
QSPI = feature("ExtBus", "External QSPI flash interface, up to 16 MB", 1)


MODELS: dict[str, dict[str, Any]] = {
    "XR806AF2L": {
        "source": "xr806", "series": "XR806", "variant": "AF2L",
        "clock": 160_000_000, "flash": 2 * 1024 * 1024, "ram": 288 * 1024,
        "packages": "QFN32 4x4 mm", "pins": "32",
        "processor": [{
            "Darchitecture": "ARMv8-M", "Dcore": "Arm-Star ARMv8-M",
            "DcoreCount": "1", "Dtz": "TrustZone-M",
        }],
        "memory": [
            {"name": "Code ROM", "type": "ROM", "bytes": 160 * 1024},
            {"name": "SIP flash", "type": "Flash", "bytes": 2 * 1024 * 1024},
            {"name": "SRAM", "type": "RAM", "bytes": 288 * 1024},
        ],
        "features": [
            WIFI, feature("Bluetooth", "Bluetooth LE 5.0", 1),
            feature("Security", "Arm TrustZone-M", 1),
            feature("Security", "Secure Boot", 1),
            feature("Security", "Flash encryption and decryption", 1),
            feature("Crypto", "Hardware cryptography: AES, DES, 3DES, MD5, SHA, SHA-256 and TRNG", 1),
            feature("RNG", "Hardware true random number generator", 1),
            feature("VendorCapability", "40 nm low-power wireless MCU process", 1),
        ],
    },
    "XR808CT0": {
        "source": "xr808", "series": "XR808", "variant": "CT0",
        "clock": 160_000_000, "ram": 256 * 1024,
        "packages": "QFN32 4x4 mm", "pins": "32", "processor": m4_processor(),
        "memory": [
            {"name": "Code ROM", "type": "ROM", "bytes": 160 * 1024},
            {"name": "External QSPI flash", "type": "ExternalFlash", "maximum_bytes": 16 * 1024 * 1024},
        ],
        "features": [],
    },
    "XR808CT2": {
        "source": "xr808", "series": "XR808", "variant": "CT2",
        "clock": 160_000_000, "flash": 2 * 1024 * 1024, "ram": 256 * 1024,
        "packages": "QFN32 4x4 mm", "pins": "32", "processor": m4_processor(),
        "memory": [
            {"name": "Code ROM", "type": "ROM", "bytes": 160 * 1024},
            {"name": "SIP QSPI flash", "type": "Flash", "bytes": 2 * 1024 * 1024},
        ],
        "features": [feature("Memory", "SIP 2 MB QSPI flash", 1)],
    },
    "XR809": {
        "source": "xr809", "series": "XR809", "variant": "",
        "clock": 160_000_000, "flash": 2 * 1024 * 1024, "ram": 384 * 1024,
        "packages": "QFN52 6x6 mm", "pins": "52", "processor": m4_processor(),
        "memory": [{"name": "Embedded flash", "type": "Flash", "bytes": 2 * 1024 * 1024}],
        "features": [],
    },
    "XR871GT": {
        "source": "xr871", "series": "XR871", "variant": "GT",
        "clock": 192_000_000, "ram": 448 * 1024,
        "packages": "QFN52 6x6 mm", "pins": "52", "processor": m4_processor(),
        "memory": [{"name": "Code ROM", "type": "ROM", "bytes": 64 * 1024}],
        "features": [feature("VendorCapability", "32.768 kHz crystal support", 1)],
    },
    "XR871ET": {
        "source": "xr871", "series": "XR871", "variant": "ET",
        "clock": 192_000_000, "ram": 448 * 1024,
        "packages": "QFN52 6x6 mm", "pins": "52", "processor": m4_processor(),
        "memory": [{"name": "Code ROM", "type": "ROM", "bytes": 64 * 1024}],
        "features": [feature("VendorCapability", "32 kHz RC clock", 1)],
    },
    "XR872AT": {
        "source": "xr872", "series": "XR872", "variant": "AT",
        "clock": 384_000_000, "ram": 416 * 1024,
        "packages": "QFN52 6x6 mm", "pins": "52", "processor": m4_processor(),
        "memory": [
            {"name": "Code ROM", "type": "ROM", "bytes": 160 * 1024},
            {"name": "SIP PSRAM", "type": "PSRAM", "bytes": 4 * 1024 * 1024},
        ],
        "features": [
            feature("IOs", "GPIO", 35),
            feature("ADC", "12-bit GPADC channels (7 external + VBAT)", 8, "12"),
            feature("Camera", "Camera sensor interface", 1),
            feature("Accelerator", "XRadio hardware JPEG codec", 1),
            feature("I2S", "I2S audio interface", 1),
            feature("Audio", "Digital microphone interface", 1),
            feature("Audio", "Integrated audio codec", 1),
            feature("PSRAM", "SIP 4 MB PSRAM (not counted as SRAM)", 4, "MB"),
        ],
    },
    "XR872ET": {
        "source": "xr872", "series": "XR872", "variant": "ET",
        "clock": 240_000_000,
        "packages": "QFN40 5x5 mm", "pins": "40", "processor": m4_processor(),
        "memory": [{"name": "Code ROM", "type": "ROM", "bytes": 160 * 1024}],
        "features": [feature("IOs", "GPIO", 27)],
    },
}


def serial_features(model: str) -> list[dict[str, str]]:
    if model.startswith("XR808"):
        return [
            feature("Timer", "General-purpose timers", 4), feature("PWM", "PWM channels", 8),
            feature("ADC", "12-bit ADC channels (5 external + VBAT)", 6, "12"),
            feature("ADCUnits", "12-bit ADC converter units", 1, "12"),
            feature("IOs", "GPIO", 21), feature("UART", "UART controllers", 3),
            feature("SPI", "SPI controller exposed by the QFN32 feature table", 1),
            feature("I2C", "TWI / I2C controllers", 2), feature("DMA", "DMA channels", 8),
            RTC, WDT, IR, QSPI, WIFI, CRYPTO, RNG,
        ]
    if model == "XR809":
        return [
            feature("Timer", "General-purpose and alarm timers", 4), feature("PWM", "PWM channels", 4),
            feature("ADC", "12-bit ADC channels (3 external + VBAT)", 4, "12"),
            feature("ADCUnits", "12-bit ADC converter units", 1, "12"),
            feature("IOs", "GPIO", 23), feature("UART", "UART controllers", 2),
            feature("SPI", "SPI controllers", 2), feature("I2C", "TWI / I2C controller", 1),
            feature("DMA", "DMA channels", 4), feature("SDIO", "SDIO interface", 1),
            RTC, WDT, IR, WIFI, CRYPTO, RNG, feature("Security", "Secure Boot", 1),
        ]
    if model.startswith("XR871"):
        return [
            feature("Timer", "General-purpose and alarm timers", 4), feature("PWM", "PWM channels", 8),
            feature("ADC", "12-bit ADC channels (8 external + VBAT)", 9, "12"),
            feature("ADCUnits", "12-bit ADC converter units", 1, "12"),
            feature("IOs", "GPIO", 31), feature("UART", "UART controllers including NUART", 3),
            feature("SPI", "SPI controllers", 2), feature("I2C", "TWI / I2C controllers", 2),
            feature("DMA", "DMA channels", 8), feature("I2S", "I2S audio interface", 1),
            feature("SDIO", "SDIO interface", 1), feature("Camera", "Camera sensor interface", 1),
            feature("Audio", "Digital microphone interface", 1), RTC, WDT, IR, QSPI, WIFI,
            CRYPTO, RNG, feature("Security", "Secure Boot", 1),
        ]
    if model.startswith("XR872"):
        return [
            feature("Timer", "General-purpose timers", 4), feature("PWM", "PWM channels", 8),
            feature("ADCUnits", "12-bit GPADC converter units", 1, "12"),
            feature("UART", "UART controllers", 3), feature("SPI", "SPI controllers", 2),
            feature("I2C", "TWI / I2C controllers", 2), feature("DMA", "DMA channels", 8),
            feature("SDIO", "SDIO interface", 1), RTC, WDT, IR, QSPI, WIFI, CRYPTO, RNG,
        ]
    return []


def riscv_core(name: str, *, count: int = 1) -> list[dict[str, str]]:
    return [{
        "Darchitecture": "RISC-V",
        "Dcore": name,
        "DcoreCount": str(count),
    }]


def interface_features(
    *,
    uart: int | None = None,
    i2c: int | None = None,
    spi: int | None = None,
    pwm: int | None = None,
    adc_channels: int | None = None,
    gpio: int | None = None,
    can: int | None = None,
    can_fd: bool = False,
    usb: int | None = None,
    usb_otg: int | None = None,
    sdio: int | None = None,
    eth: int | None = None,
    lradc: int | None = None,
    touch_adc: int | None = None,
    camera: str = "",
    display: str = "",
    extra: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    values = (
        ("UART", "UART controllers", uart),
        ("I2C", "TWI / I2C controllers", i2c),
        ("SPI", "SPI controllers", spi),
        ("PWM", "PWM channels", pwm),
        ("ADC", "GPADC channels", adc_channels),
        ("IOs", "GPIO", gpio),
        ("CAN", "CAN-FD controllers" if can_fd else "CAN controllers", can),
        ("USB", "USB interfaces", usb),
        ("USBOTG", "USB dual-role controller", usb_otg),
        ("SDIO", "SDIO / SMHC controllers", sdio),
        ("ETH", "GMAC / Ethernet controllers", eth),
        ("AnalogOther", "LRADC channels", lradc),
        ("Touch", "TPADC channels", touch_adc),
    )
    for kind, name, count in values:
        if count is not None:
            result.append(feature(kind, name, count))
    if camera:
        result.append(feature("Camera", camera, 1))
    if display:
        result.append(feature("GLCD", display, 1))
    result.extend(extra or [])
    return result


def realtime_soc(
    *,
    source: str,
    series: str,
    line: str,
    core: str,
    clock: int | None,
    features: list[dict[str, str]],
    device_name: str | None = None,
    core_count: int = 1,
    processor: list[dict[str, str]] | None = None,
    product_type: str = "heterogeneous_realtime_soc",
    variant: str = "",
    part_number: str = "",
    package: str = "",
    pins: str = "",
    ram: int | None = None,
    flash: int | None = None,
    memory: list[dict[str, Any]] | None = None,
    family: str = "Real-time MCU SoC",
) -> dict[str, Any]:
    return {
        "source": source,
        "series": line,
        "catalog_series": series,
        "product_line": line,
        "device_name": device_name or line,
        "variant": variant,
        "part_number": part_number,
        "clock": clock,
        "flash": flash,
        "ram": ram,
        "packages": package,
        "pins": pins,
        "processor": processor if processor is not None else riscv_core(core, count=core_count),
        "architecture_class": "RISC-V" if "Star-MC1" not in core else "ARMv8-M",
        "product_type": product_type,
        "family": family,
        "memory": memory or [],
        "features": [
            feature("VendorCapability", f"Catalog class: {product_type}", 1),
            *features,
        ],
    }


SOC_MODELS: dict[str, dict[str, Any]] = {
    "A733MX-HN3": realtime_soc(
        source="a733", series="A series", line="A733", core="XuanTie E902", clock=None,
        device_name="A733MX-HN3", variant="MX-HN3", part_number="A733MX-HN3",
        features=interface_features(
            uart=9, i2c=16, spi=5, pwm=30, adc_channels=7, usb=3, sdio=3, eth=1,
            lradc=1, camera="MIPI CSI 2x4+2 lanes, up to 24 MP at 25 fps",
            display="RGB, dual-link LVDS, MIPI DSI, E-ink, eDP 1.4b / DP 1.4",
            extra=[
                feature("NPU", "Allwinner 3 TOPS NPU", 1),
                feature("Accelerator", "Imagination PowerVR BXM-4-64 MC1 GPU", 1),
                feature("Accelerator", "H.264/H.265 4K video codec and JPEG engine", 1),
                feature("ExtBus", "PCIe 3.0", 1),
                feature("VendorCapability", "Dual Cortex-A76 plus hexa Cortex-A55 application CPU", 1),
                feature("VendorCapability", "Android 13/15 and Tina Linux 5.15", 1),
            ],
        ),
    ),
    "A537": realtime_soc(
        source="a537", series="A series", line="A537", core="XuanTie E902", clock=None,
        features=interface_features(
            uart=6, i2c=6, spi=3, pwm=5, adc_channels=1, usb=3, sdio=3, eth=1,
            lradc=1, camera="MIPI CSI 4+2x2 / 4+4 lanes, up to 16 MP",
            display="RGB, dual-link LVDS, MIPI DSI, E-ink, eDP 1.3",
            extra=[
                feature("Accelerator", "Arm Mali-G57 MC1 GPU", 1),
                feature("Accelerator", "H.264/VP9/H.265 video and JPEG engine", 1),
                feature("ExtBus", "PCIe 2.1", 1),
                feature("VendorCapability", "Dual Cortex-A73 plus hexa Cortex-A53 application CPU", 1),
                feature("VendorCapability", "Official page uses wildcard sample A537PX-0XX; no orderable part generated", 1),
            ],
        ),
    ),
    "A333": realtime_soc(
        source="a333", series="A series", line="A333", core="XuanTie E902", clock=None,
        features=interface_features(
            uart=6, i2c=6, spi=3, pwm=5, adc_channels=1, usb=3, sdio=3, eth=1,
            lradc=1, camera="MIPI CSI 4+2x2 / 4+4 lanes, up to 13 MP",
            display="RGB, dual-link LVDS, MIPI DSI, E-ink, eDP 1.3",
            extra=[
                feature("Accelerator", "Arm Mali-G57 MC1 GPU", 1),
                feature("Accelerator", "H.264/VP9/H.265 video and JPEG engine", 1),
                feature("ExtBus", "PCIe 2.1", 1),
                feature("VendorCapability", "Cortex-A73 plus quad Cortex-A53 application CPU", 1),
                feature("VendorCapability", "Official page uses wildcard sample A333MX-0XX; no orderable part generated", 1),
            ],
        ),
    ),
    "A527": realtime_soc(
        source="a527", series="A series", line="A527", core="XuanTie E906", clock=200_000_000,
        features=interface_features(
            uart=10, i2c=9, spi=4, pwm=30, adc_channels=24, sdio=1, eth=2, lradc=2,
            camera="Parallel CSI and MIPI CSI multi-camera interface",
            display="Dual MIPI DSI, dual-link LVDS, RGB and eDP",
            extra=[
                feature("USB", "PCIe 2.1 / USB 3.1 combo plus USB 2.0 DRD and Host"),
                feature("Accelerator", "Arm Mali-G57 GPU", 1),
                feature("Accelerator", "4K H.264/H.265/VP9 video and JPEG engine", 1),
                feature("VendorCapability", "Octa-core Cortex-A55 application CPU", 1),
            ],
        ),
    ),
    "A523": realtime_soc(
        source="a523", series="A series", line="A523", core="XuanTie E906", clock=200_000_000,
        features=interface_features(
            uart=10, i2c=9, spi=4, pwm=30, adc_channels=4, sdio=1, eth=1, lradc=2,
            camera="Parallel CSI and MIPI CSI multi-camera interface",
            display="Dual MIPI DSI, dual-link LVDS, RGB and eDP",
            extra=[
                feature("USB", "PCIe 2.1 / USB 3.1 combo plus USB 2.0 DRD and Host"),
                feature("Accelerator", "Arm Mali-G57 GPU", 1),
                feature("Accelerator", "4K H.264/H.265/VP9 video and JPEG engine", 1),
                feature("VendorCapability", "Octa-core Cortex-A55 application CPU", 1),
            ],
        ),
    ),
    "T736MX-HN3": realtime_soc(
        source="t736", series="T series", line="T736", core="XuanTie E902", clock=None,
        device_name="T736MX-HN3", variant="MX-HN3", part_number="T736MX-HN3",
        features=interface_features(
            uart=9, i2c=16, spi=5, pwm=30, adc_channels=7, usb=3, sdio=3, eth=2,
            lradc=1, camera="MIPI CSI 2x4+2 lanes, up to 24 MP at 25 fps",
            display="RGB, dual MIPI DSI, dual-link LVDS, E-ink, eDP 1.4b / DP 1.4",
            extra=[
                feature("NPU", "Allwinner 3 TOPS NPU", 1),
                feature("Accelerator", "Imagination PowerVR BXM-4-64 MC1 GPU", 1),
                feature("Accelerator", "H.264/H.265 4K video codec and JPEG engine", 1),
                feature("ExtBus", "PCIe 3.0", 1),
                feature("VendorCapability", "Dual Cortex-A76 plus hexa Cortex-A55 application CPU", 1),
                feature("VendorCapability", "Industrial -40 C to 85 C operating range", 1),
            ],
        ),
    ),
    "T527": realtime_soc(
        source="t527", series="T series", line="T527", core="XuanTie E906", clock=200_000_000,
        features=interface_features(
            uart=10, i2c=9, spi=4, pwm=30, adc_channels=24, can=2, sdio=1, eth=2,
            lradc=2, camera="Parallel CSI and MIPI CSI multi-camera interface",
            display="MIPI DSI, dual-link LVDS, RGB and eDP",
            extra=[
                feature("USB", "PCIe 2.1 / USB 3.1 combo plus USB 2.0 DRD and Host"),
                feature("NPU", "Allwinner 2 TOPS INT8 NPU", 1),
                feature("Accelerator", "Arm Mali-G57 GPU", 1),
                feature("Accelerator", "Cadence Tensilica HiFi4 DSP", 1),
                feature("Accelerator", "4K H.264/H.265/VP9 video and JPEG engine", 1),
                feature("VendorCapability", "Octa-core Cortex-A55 application CPU", 1),
            ],
        ),
    ),
    "T536MX-CEN2": realtime_soc(
        source="t536", series="T series", line="T536", core="XuanTie E907 + E902", clock=600_000_000,
        device_name="T536MX-CEN2", variant="MX-CEN2", part_number="T536MX-CEN2",
        processor=[*riscv_core("XuanTie E907"), *riscv_core("XuanTie E902")], core_count=2,
        features=interface_features(
            uart=17, i2c=9, spi=6, pwm=34, adc_channels=31, gpio=215, can=4, can_fd=True,
            sdio=1, eth=2, lradc=1, touch_adc=4,
            camera="Parallel CSI and MIPI CSI multi-camera interface",
            display="RGB, dual-link LVDS and MIPI DSI",
            extra=[
                feature("USB", "PCIe 2.1 / USB 3.1 combo plus USB 2.0 DRD and Host"),
                feature("ExtBus", "LocalBus parallel interface", 1),
                feature("NPU", "Allwinner 2 TOPS NPU", 1),
                feature("Accelerator", "H.264/MJPEG/JPEG codec engine", 1),
                feature("Security", "Full-path ECC data checking and correction", 1),
                feature("VendorCapability", "Quad-core Cortex-A55 application CPU", 1),
                feature("VendorCapability", "AMP: Tina Linux + RTOS + Baremetal", 1),
            ],
        ),
    ),
    "T153MX-BCX": realtime_soc(
        source="t153", series="T series", line="T153", core="XuanTie E907", clock=600_000_000,
        device_name="T153MX-BCX", variant="MX-BCX", part_number="T153MX-BCX",
        features=interface_features(
            uart=10, i2c=6, pwm=30, adc_channels=24, gpio=141, can=2, can_fd=True,
            eth=3, touch_adc=4,
            camera="Parallel CSI and MIPI CSI multi-camera interface",
            display="RGB, dual-link LVDS and MIPI DSI",
            extra=[
                feature("ExtBus", "LocalBus parallel interface", 1),
                feature("Accelerator", "Allwinner ISP image processor", 1),
                feature("Accelerator", "Allwinner G2D graphics accelerator", 1),
                feature("VendorCapability", "Quad-core Cortex-A7 application CPU", 1),
                feature("VendorCapability", "AMP: Tina Linux + RTOS + Baremetal", 1),
            ],
        ),
    ),
    "MR153MX-BCX": realtime_soc(
        source="mr153", series="MR series", line="MR153", core="XuanTie E907", clock=600_000_000,
        device_name="MR153MX-BCX", variant="MX-BCX", part_number="MR153MX-BCX",
        features=interface_features(
            uart=10, pwm=30, adc_channels=24, gpio=141, usb=2, eth=1, touch_adc=4,
            camera="Parallel CSI and MIPI CSI multi-camera interface",
            extra=[
                feature("Accelerator", "Allwinner ISP image processor", 1),
                feature("Accelerator", "Allwinner 2D graphics accelerator", 1),
                feature("Security", "E907 safety library for IEC 60730", 1),
                feature("Security", "Trusted execution environment", 1),
                feature("VendorCapability", "Quad-core Cortex-A7 application CPU", 1),
                feature("VendorCapability", "FreeRTOS on the E907 real-time core", 1),
            ],
        ),
    ),
    "MR527": realtime_soc(
        source="mr527", series="MR series", line="MR527", core="XuanTie E906", clock=200_000_000,
        features=interface_features(
            uart=10, i2c=9, spi=4, pwm=30, adc_channels=24, gpio=203, sdio=1, eth=1,
            lradc=2, camera="Parallel CSI and MIPI CSI multi-camera interface",
            extra=[
                feature("USB", "PCIe 2.1 / USB 3.1 combo plus USB 2.0 DRD and Host"),
                feature("NPU", "Allwinner 2 TOPS INT8 NPU", 1),
                feature("Accelerator", "Arm Mali-G57 GPU", 1),
                feature("Accelerator", "H.264/JPEG video engine", 1),
                feature("VendorCapability", "Octa-core Cortex-A55 application CPU", 1),
                feature("VendorCapability", "MCU can independently run a real-time system", 1),
            ],
        ),
    ),
    "MR536MX-N3Z": realtime_soc(
        source="mr536", series="MR series", line="MR536", core="XuanTie E907", clock=600_000_000,
        device_name="MR536MX-N3Z", variant="MX-N3Z", part_number="MR536MX-N3Z",
        features=interface_features(
            uart=17, i2c=9, spi=6, pwm=34, adc_channels=31, gpio=215, sdio=1, eth=1,
            lradc=1, touch_adc=4, camera="Parallel CSI and MIPI CSI multi-camera interface",
            extra=[
                feature("USB", "PCIe 2.1 / USB 3.1 combo plus USB 2.0 DRD and Host"),
                feature("NPU", "Allwinner 3 TOPS Transformer-capable NPU", 1),
                feature("Accelerator", "H.264/JPEG video engine", 1),
                feature("VendorCapability", "Quad-core Cortex-A55 application CPU", 1),
                feature("VendorCapability", "Independent MCU system with FreeRTOS", 1),
            ],
        ),
    ),
    "V881": realtime_soc(
        source="v881", series="V series", line="V881", core="XuanTie E907", clock=800_000_000,
        features=interface_features(
            uart=4, i2c=4, spi=3, usb_otg=1,
            camera="MIPI CSI 1x4 / 2x2 / 3x1 lanes plus parallel CSI",
            display="RGB888, serial RGB, i8080 and SPI display",
            extra=[
                feature("NPU", "Allwinner 1 TOPS INT8 NPU", 1),
                feature("Accelerator", "Allwinner AI-ISP with EIS and PDAF", 1),
                feature("Accelerator", "4K H.264/H.265/MJPEG/JPEG engine", 1),
                feature("VendorCapability", "Dual-core XuanTie C907 application CPU", 1),
                feature("VendorCapability", "Official page uses wildcard sample V881MX-XXX; no orderable part generated", 1),
            ],
        ),
    ),
    "V861": realtime_soc(
        source="v861", series="V series", line="V861", core="XuanTie E907", clock=800_000_000,
        memory=[{"name": "SiP DDR3", "type": "SiPDDR", "bytes": 128 * 1024 * 1024}],
        features=interface_features(
            uart=4, i2c=4, spi=3, pwm=15, usb_otg=1, eth=1,
            camera="MIPI CSI 1x4 / 2x2 / 3x1 lanes plus parallel CSI", display="SPI display",
            extra=[
                feature("ADC", "GPADC present; channel count not published"),
                feature("NPU", "Allwinner 1 TOPS INT8 NPU", 1),
                feature("Accelerator", "Allwinner AI-ISP 2.0", 1),
                feature("Accelerator", "4K H.264/H.265/JPEG engine", 1),
                feature("VendorCapability", "Dual-core XuanTie C907 application CPU", 1),
                feature("VendorCapability", "Official page uses wildcard sample V861M3-XXX; no orderable part generated", 1),
            ],
        ),
    ),
    "V838": realtime_soc(
        source="v838", series="V series", line="V838", core="XuanTie E907", clock=600_000_000,
        memory=[{"name": "SiP DDR3", "type": "SiPDDR", "bytes": 64 * 1024 * 1024}],
        features=interface_features(
            uart=4, i2c=4, spi=3, pwm=15, usb_otg=1, eth=1,
            camera="MIPI CSI 1x4 / 2x2 / 3x1 lanes plus parallel CSI", display="SPI display",
            extra=[
                feature("ADC", "GPADC present; channel count not published"),
                feature("NPU", "Allwinner 0.6 TOPS INT8 NPU", 1),
                feature("Accelerator", "Allwinner ISP 3.0", 1),
                feature("Accelerator", "H.264/H.265/JPEG engine", 1),
                feature("VendorCapability", "Dual-core XuanTie C907 application CPU", 1),
                feature("VendorCapability", "Official page uses wildcard sample V838M2-EXX; no orderable part generated", 1),
            ],
        ),
    ),
    "V821": realtime_soc(
        source="v821", series="V series", line="V821", core="RISC-V MCU", clock=600_000_000,
        package="QFN88 9x9 mm", pins="88",
        memory=[{"name": "SiP DDR", "type": "SiPDDR", "bytes": 64 * 1024 * 1024}],
        features=interface_features(
            uart=4, i2c=3, spi=3, pwm=12, usb_otg=1, sdio=1, eth=1,
            camera="MIPI CSI 2x1 / 1x2 lanes plus parallel CSI",
            display="Serial RGB / i8080 display up to 800x480 at 60 fps",
            extra=[
                feature("ADC", "GPADC present; channel count not published"),
                feature("WiFi", "Integrated 2.4 GHz Wi-Fi 4", 1),
                feature("Bluetooth", "Bluetooth LE listed by the official product selector", 1),
                feature("Audio", "Integrated audio codec", 1),
                feature("Accelerator", "H.264/JPEG video engine", 1),
                feature("VendorCapability", "RISC-V application CPU up to 1 GHz", 1),
                feature("VendorCapability", "FreeRTOS on the RISC-V MCU", 1),
                feature("VendorCapability", "Official page uses wildcard sample V821L2-WXX; no orderable part generated", 1),
            ],
        ),
    ),
    "V853": realtime_soc(
        source="v853", series="V series", line="V853", core="XuanTie E907", clock=600_000_000,
        features=interface_features(
            uart=4, i2c=5, spi=4, pwm=12, adc_channels=4, gpio=8, usb_otg=1, sdio=3, eth=1,
            camera="Parallel CSI and MIPI CSI 4 / 2x2 lanes",
            display="MIPI DSI, RGB888 and BT.656 display",
            extra=[
                feature("NPU", "Allwinner 1 TOPS INT8 NPU", 1),
                feature("Accelerator", "Allwinner ISP image processor", 1),
                feature("Accelerator", "H.264/H.265/MJPEG/JPEG engine", 1),
                feature("VendorCapability", "Cortex-A7 application CPU", 1),
                feature("VendorCapability", "FreeRTOS on the E907 real-time core", 1),
            ],
        ),
    ),
    "V851s": realtime_soc(
        source="v851s", series="V series", line="V851s", core="XuanTie E907", clock=600_000_000,
        memory=[{"name": "SiP DDR", "type": "SiPDDR", "bytes": 64 * 1024 * 1024}],
        features=interface_features(
            uart=4, i2c=5, spi=4, pwm=11, gpio=6, usb_otg=1, sdio=3, eth=1,
            camera="Parallel CSI and dual MIPI CSI 2x2 lanes",
            display="MIPI DSI, RGB888 and BT.656 display",
            extra=[
                feature("ADC", "GPADC present; channel count not published"),
                feature("NPU", "Allwinner 0.5 TOPS INT8 NPU", 1),
                feature("Accelerator", "Allwinner ISP image processor", 1),
                feature("Accelerator", "H.264/H.265/MJPEG/JPEG engine", 1),
                feature("VendorCapability", "Cortex-A7 application CPU", 1),
                feature("VendorCapability", "FreeRTOS on the E907 real-time core", 1),
            ],
        ),
    ),
    "V837s": realtime_soc(
        source="v837s", series="V series", line="V837s", core="XuanTie E907", clock=600_000_000,
        memory=[{"name": "SiP DDR", "type": "SiPDDR", "bytes": 64 * 1024 * 1024}],
        features=interface_features(
            uart=4, i2c=5, spi=4, pwm=11, gpio=6, usb_otg=1, sdio=1, eth=1,
            camera="Parallel CSI and MIPI CSI 2-lane",
            display="MIPI DSI and RGB display",
            extra=[
                feature("ADC", "GPADC present; channel count not published"),
                feature("Accelerator", "Allwinner ISP image processor", 1),
                feature("Accelerator", "H.264/H.265/MJPEG/JPEG engine", 1),
                feature("VendorCapability", "Cortex-A7 application CPU", 1),
                feature("VendorCapability", "RTOS on the E907 real-time core", 1),
            ],
        ),
    ),
    "R128": realtime_soc(
        source="r128", series="R series", line="R128", core="Arm Star-MC1", clock=192_000_000,
        product_type="wireless_audio_mcu_soc", family="Wireless audio MCU SoC",
        part_number="R128", package="QFN80 8x8 mm", pins="80", ram=1 * 1024 * 1024,
        processor=[{"Darchitecture": "ARMv8-M", "Dcore": "Arm Star-MC1", "DcoreCount": "1"}],
        memory=[
            {"name": "SRAM", "type": "RAM", "bytes": 1 * 1024 * 1024},
            {"name": "NOR flash options", "type": "FlashOptions", "sizes_bytes": [8 * 1024 * 1024, 16 * 1024 * 1024]},
            {"name": "PSRAM options", "type": "PSRAMOptions", "sizes_bytes": [8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024]},
        ],
        features=interface_features(
            uart=3, i2c=2, spi=2, pwm=8, adc_channels=8, gpio=38, sdio=1,
            camera="Camera input with JPEG up to 1920x1088 or 720p30",
            display="RGB666 / MCU / SPI-DBI / SPI display output",
            extra=[
                feature("WiFi", "2.4 GHz IEEE 802.11 b/g/n Wi-Fi", 1),
                feature("Bluetooth", "Bluetooth 5.0 BR/EDR/BLE/MESH", 1),
                feature("I2S", "I2S audio interface", 1),
                feature("Audio", "24-bit audio codec: 3 ADC, 2 DAC, headphone and 4 DMIC", 1),
                feature("VendorCapability", "IR transmitter and receiver", 2),
                feature("Accelerator", "Cadence Tensilica HiFi5 DSP up to 400 MHz", 1),
                feature("Accelerator", "JPEG codec", 1),
                feature("Accelerator", "Allwinner G2D and display engine", 1),
                feature("VendorCapability", "XuanTie C906 application CPU up to 533 MHz", 1),
                feature("VendorCapability", "FreeRTOS", 1),
            ],
        ),
    ),
    "XR819": realtime_soc(
        source="xr_connectivity", series="XR series", line="XR819", core="Unpublished", clock=None,
        product_type="wireless_connectivity_chip", family="XRadio connectivity",
        part_number="XR819", package="QFN32 5x5 mm", pins="32", processor=[],
        features=interface_features(sdio=1, extra=[
            feature("WiFi", "2.4 GHz IEEE 802.11 b/g/n, HT20", 1),
            feature("VendorCapability", "3.0 to 3.6 V VBAT", 1),
        ]),
    ),
    "XR819s": realtime_soc(
        source="xr_connectivity", series="XR series", line="XR819s", core="Unpublished", clock=None,
        product_type="wireless_connectivity_chip", family="XRadio connectivity",
        part_number="XR819s", package="QFN32 4x4 mm", pins="32", processor=[],
        features=interface_features(sdio=1, extra=[
            feature("WiFi", "2.4 GHz IEEE 802.11 b/g/n, HT20", 1),
            feature("Bluetooth", "Bluetooth LE 5.0", 1),
            feature("VendorCapability", "1.8 to 5.5 V VBAT", 1),
        ]),
    ),
    "XR829": realtime_soc(
        source="xr_connectivity", series="XR series", line="XR829", core="Unpublished", clock=None,
        product_type="wireless_connectivity_chip", family="XRadio connectivity",
        part_number="XR829", package="QFN40 5x5 mm", pins="40", processor=[],
        features=interface_features(sdio=1, extra=[
            feature("WiFi", "2.4 GHz IEEE 802.11 b/g/n, HT20/HT40", 1),
            feature("Bluetooth", "Bluetooth 4.2 BR/EDR/BLE", 1),
            feature("VendorCapability", "3.0 to 5.5 V VBAT", 1),
        ]),
    ),
}

# SOC_MODELS intentionally excluded from the MCU catalog.  The table remains
# in this source file as an audited reference for a future, separate SoC
# catalog, but Allwinner's MCU directory must not mix it with XR MCU parts.


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "vendor-packs" / "allwinner")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "allwinner")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    fetcher = OfficialFetcher(
        cache_dir=args.cache_dir,
        allowed_hosts={"xr806.docs.aw-ol.com", "raw.githubusercontent.com", "www.allwinnertech.com"},
        refresh=args.refresh,
    )
    snapshots: dict[str, Any] = {}
    for key, source in SOURCES.items():
        if source["type"] == "manufacturer_product_page":
            snapshots[key] = None
            continue
        snapshot = fetcher.fetch(source["url"], cache_name=source["cache"])
        if key == "xr806" and b"XR806AF2L" not in snapshot.payload:
            raise RuntimeError("XR806 official page no longer mentions XR806AF2L")
        if source["type"] == "manufacturer_datasheet" and (
            not snapshot.payload.startswith(b"%PDF") or len(snapshot.payload) < 50_000
        ):
            raise RuntimeError(f"{source['title']} did not return a valid PDF snapshot")
        snapshots[key] = snapshot

    observed = utc_now()
    devices: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    for model, item in MODELS.items():
        source_key = item["source"]
        source = SOURCES[source_key]
        snapshot = snapshots[source_key]
        source_hash = snapshot.sha256 if snapshot is not None else PRODUCT_PAGE_EVIDENCE_SHA256[source_key]
        product_type = item.get("product_type", "wireless_mcu")
        product_line = item.get("product_line", item["series"])
        series = item.get(
            "catalog_series", "XR series" if product_type == "wireless_mcu" else item["series"]
        )
        device_name = item.get("device_name", model)
        family = item.get("family", "XRadio")
        architecture_class = item.get(
            "architecture_class", "ARMv8-M" if model == "XR806AF2L" else "Cortex-M4"
        )
        device_id = f"allwinner::{slug(device_name)}"
        line_id = f"allwinner::{slug(series)}::{slug(product_line)}"
        features = [*serial_features(model), *item["features"]]
        document = {"title": source["title"], "url": source["url"], "version": source["version"]}
        verification_status = (
            "manufacturer_product_page"
            if source["type"] == "manufacturer_product_page"
            else "manufacturer_datasheet"
        )
        devices.append({
            "device_id": device_id,
            "product_line_id": line_id,
            "manufacturer": MANUFACTURER,
            "product_type": product_type,
            "architecture_class": architecture_class,
            "family": family,
            "series": series,
            "product_line": product_line,
            "device_name": device_name,
            "generic_device_name": product_line,
            "manufacturer_variant_code": item["variant"],
            "processor_cores": json.dumps(item["processor"], ensure_ascii=False),
            "max_clock_hz": item.get("clock", ""),
            "flash_bytes": item.get("flash", ""),
            "ram_bytes": item.get("ram", ""),
            "package_types": item["packages"],
            "pin_counts": item["pins"],
            "memory_regions_json": json.dumps(item["memory"], ensure_ascii=False),
            "features_json": json.dumps(features, ensure_ascii=False),
            "documents_json": json.dumps([document], ensure_ascii=False),
            "svd_files": "",
            "lifecycle": "unknown",
            "source_id": source["id"],
            "source_url": source["url"],
            "source_version": f"{source['version']};sha256:{source_hash}",
            "observed_at": observed,
            "verification_status": verification_status,
        })
        part_number = item.get("part_number", model if product_type == "wireless_mcu" else "")
        if part_number:
            parts.append({
                "orderable_part_id": f"allwinner::{slug(part_number)}",
                "device_id": device_id,
                "manufacturer": MANUFACTURER,
                "family": family,
                "series": series,
                "product_line": product_line,
                "device_name": device_name,
                "part_number": part_number,
                "manufacturer_suffix": item["variant"],
                "package_code": "",
                "temperature_grade_code": "",
                "packing_code": "",
                "package_name": item["packages"],
                "temperature_range": "",
                "packing_form": "",
                "lifecycle": "unknown",
                "source_id": source["id"],
                "source_url": source["url"],
                "observed_at": observed,
                "verification_status": verification_status,
                "decode_status": "exact_official_model; wildcard_suffixes_not_generated",
            })

    lines = []
    for line_id in sorted({row["product_line_id"] for row in devices}):
        line_devices = [row for row in devices if row["product_line_id"] == line_id]
        source_ids = sorted({row["source_id"] for row in line_devices})
        lines.append({
            "product_line_id": line_id,
            "manufacturer": MANUFACTURER,
            "product_type": line_devices[0]["product_type"],
            "architecture_class": line_devices[0]["architecture_class"],
            "family": line_devices[0]["family"],
            "series": line_devices[0]["series"],
            "product_line": line_devices[0]["product_line"],
            "device_variant_count": len(line_devices),
            "source_ids": ";".join(source_ids),
            "observed_at": observed,
            "verification_status": line_devices[0]["verification_status"],
        })

    source_rows = []
    for key, source in SOURCES.items():
        snapshot = snapshots[key]
        source_hash = snapshot.sha256 if snapshot is not None else PRODUCT_PAGE_EVIDENCE_SHA256[key]
        source_rows.append({
            "source_id": source["id"],
            "source_type": source["type"],
            "publisher": source["publisher"],
            "title": source["title"],
            "url": source["url"],
            "version": f"{source['version']};sha256:{source_hash}",
            "observed_at": observed,
            "verification_scope": source["scope"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "product-lines.csv", PRODUCT_LINE_FIELDS, lines)
    write_csv(args.output_dir / "device-variants.csv", DEVICE_FIELDS, sorted(devices, key=lambda row: row["device_name"]))
    write_csv(args.output_dir / "orderable-parts.csv", PART_FIELDS, sorted(parts, key=lambda row: row["part_number"]))
    write_csv(args.output_dir / "sources.csv", SOURCE_FIELDS, source_rows)
    write_csv(args.output_dir / "import-errors.csv", ERROR_FIELDS, [])
    report = {
        "status": "ok",
        "adapter": "import_allwinner_official.py",
        "manufacturer": MANUFACTURER,
        "device_variants": len(devices),
        "orderable_parts": len(parts),
        "series": sorted({row["series"] for row in devices}),
        "product_lines": sorted({row["product_line"] for row in devices}),
        "exact_models": sorted(row["device_name"] for row in devices),
        "product_type_counts": {
            kind: sum(row["product_type"] == kind for row in devices)
            for kind in sorted({row["product_type"] for row in devices})
        },
        "source_snapshot_sha256": {
            key: snapshot.sha256 if snapshot is not None else PRODUCT_PAGE_EVIDENCE_SHA256[key]
            for key, snapshot in snapshots.items()
        },
        "excluded_non_mcu_examples": [
            "A/T/MR/V application and real-time SoCs", "R128 wireless-audio SoC",
            "XR819/XR819s/XR829 connectivity chips", "D1", "T113", "F1C", "A133",
            "MR813", "V316-H", "V3s",
        ],
        "completeness_claim": (
            "Standalone XRadio wireless MCU datasheets only. Allwinner A/T/MR/V/R128 SoCs and XR connectivity "
            "chips are intentionally excluded from the MCU directory; wildcard suffixes and unknown fields are "
            "not inferred."
        ),
    }
    write_json(args.output_dir / "official-adapter-report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
