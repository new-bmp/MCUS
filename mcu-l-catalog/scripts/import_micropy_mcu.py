#!/usr/bin/env python3
"""Import MCU-class chips with an official MicroPython port.

The directory is intentionally separate from silicon vendors: ``MicroPy MCU``
is a software/ecosystem entry for chips that have a maintained MicroPython
port, while the ``family`` and ``series`` fields retain the silicon maker's
identity.  Values below come from official datasheets, SDK headers or the
official MicroPython port documentation.  Package alternatives are represented
as separate exact device variants where their GPIO/ADC resources differ.
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


MANUFACTURER = "MicroPy MCU"

SOURCES: dict[str, dict[str, str]] = {
    "micropy:rp2-port": {
        "source_type": "official_software_port",
        "publisher": "MicroPython",
        "title": "MicroPython official rp2 port documentation",
        "url": "https://raw.githubusercontent.com/micropython/micropython/master/ports/rp2/README.md",
        "version": "main",
        "cache": "micropython-rp2-README.md",
        "scope": "RP2040 and RP2350 MicroPython rp2 port, machine peripherals and PIO support.",
    },
    "raspberrypi:rp2040-datasheet": {
        "source_type": "manufacturer_datasheet",
        "publisher": "Raspberry Pi",
        "title": "RP2040 Datasheet",
        "url": "https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf",
        "version": "official",
        "cache": "rp2040-datasheet.pdf",
        "scope": "RP2040 core, memory, package, GPIO, ADC and peripheral resources.",
    },
    "raspberrypi:rp2350-datasheet": {
        "source_type": "manufacturer_datasheet",
        "publisher": "Raspberry Pi",
        "title": "RP2350 Datasheet",
        "url": "https://datasheets.raspberrypi.com/rp2350/rp2350-datasheet.pdf",
        "version": "official",
        "cache": "rp2350-datasheet.pdf",
        "scope": "RP2350A/RP2350B selectable cores, memory, package, GPIO, ADC and peripheral resources.",
    },
    "kendryte:k210-datasheet": {
        "source_type": "manufacturer_datasheet",
        "publisher": "Canaan / Kendryte",
        "title": "Kendryte K210 Functional Description",
        "url": "https://raw.githubusercontent.com/kendryte/kendryte-doc-datasheet/master/en/003.md",
        "version": "official repository",
        "cache": "k210-functional-description.md",
        "scope": "K210 CPU, SRAM, GPIO/FPIOA, UART, SPI, I2C, I2S, timers, DMA and accelerators.",
    },
    "kendryte:k210-package": {
        "source_type": "manufacturer_datasheet",
        "publisher": "Canaan / Kendryte",
        "title": "Kendryte K210 Package Information",
        "url": "https://raw.githubusercontent.com/kendryte/kendryte-doc-datasheet/master/en/005.md",
        "version": "official repository",
        "cache": "k210-package.md",
        "scope": "K210 BGA144 package dimensions and ball configuration.",
    },
    "kendryte:k210-micropython": {
        "source_type": "official_software_manual",
        "publisher": "Kendryte / OpenMV",
        "title": "K210 MicroPython and OpenMV Manual",
        "url": "https://raw.githubusercontent.com/kendryte/K210-Micropython-OpenMV/master/doc/K210-MicroPython%26OpenMV%20Manual%20v1.1-en(20210420).pdf",
        "version": "v1.1",
        "cache": "k210-micropython-openmv-manual.pdf",
        "scope": "Official K210 MicroPython/OpenMV runtime and peripheral API manual.",
    },
    "kendryte:k230-datasheet": {
        "source_type": "manufacturer_datasheet",
        "publisher": "Canaan / Kendryte",
        "title": "Kendryte K230 Full Datasheet",
        "url": "https://raw.githubusercontent.com/kendryte/k230_docs/main/en/00_hardware/K230_datasheet.md",
        "version": "V1.0",
        "cache": "k230-datasheet.md",
        "scope": "K230 CPU, KPU, SRAM, memory, audio, UART/I2C/SPI/PWM/GPIO, USB/SD and package variants.",
    },
    "kendryte:k230-canmv": {
        "source_type": "official_software_manual",
        "publisher": "Kendryte / Canaan",
        "title": "CanMV K230 Tutorial",
        "url": "https://raw.githubusercontent.com/kendryte/k230_docs/main/en/CanMV_K230_Tutorial.md",
        "version": "main",
        "cache": "k230-canmv-tutorial.md",
        "scope": "Official CanMV K230 workflow and MicroPython API support statement.",
    },
    "kendryte:k510-docs": {
        "source_type": "manufacturer_documentation",
        "publisher": "Canaan / Kendryte",
        "title": "Kendryte K510 official documentation index",
        "url": "https://github.com/kendryte/k510_docs",
        "version": "dev",
        "cache": "k510-docs-readme.md",
        "scope": "K510 official CRB hardware guide and SDK application documentation; chip-level fields remain unknown where the public source is board-only.",
    },
    "raspberrypi:rp2354-hardware": {
        "source_type": "manufacturer_datasheet",
        "publisher": "Raspberry Pi",
        "title": "RP2350 Datasheet used for RP2354 stacked-flash silicon",
        "url": "https://datasheets.raspberrypi.com/rp2350/rp2350-datasheet.pdf",
        "version": "official RP2350 family reference",
        "cache": "rp2354-rp2350-family-datasheet.pdf",
        "scope": "RP2354A/B core, memory and peripheral resources inherited from the RP2350 family reference.",
    },
    "raspberrypi:rp2354-micropython": {
        "source_type": "official_software_issue",
        "publisher": "MicroPython",
        "title": "RP2354B stacked-flash support discussion",
        "url": "https://github.com/micropython/micropython/issues/19108",
        "version": "open issue",
        "cache": "micropython-rp2354-issue-19108.json",
        "scope": "Documents current rp2 UF2 boot compatibility issue for RP2354B stacked flash; support is under review.",
    },
}


def feature(kind: str, name: str, count: int | None = None, detail: str = "") -> dict[str, str]:
    item: dict[str, str] = {"type": kind, "name": name, "source_kind": "official_datasheet"}
    if count is not None:
        item["count"] = str(count)
        item["n"] = str(count)
    if detail:
        item["m"] = detail
    return item


def processor(
    core: str,
    *,
    architecture: str,
    count: int = 1,
    fpu: str = "unknown",
    dsp: str = "unknown",
    description: str = "",
) -> list[dict[str, str]]:
    item = {
        "Darchitecture": architecture,
        "Dcore": core,
        "DcoreCount": str(count),
        "Dfpu": fpu,
        "Ddsp": dsp,
    }
    if description:
        item["DsourceDescription"] = description
    return [item]


def rp_features(*, adc_channels: int, gpio: int, dma: int | None, pio_blocks: int, pio_sms: int, pwm_slices: int) -> list[dict[str, str]]:
    features = [
        feature("Timer", "hardware timer peripheral", 1, "one timer peripheral with four alarm channels"),
        feature("TimerOther", "timer alarm channels", 4),
        feature("ADCUnits", "12-bit SAR ADC converter unit", 1, "500 kS/s; ADC pins are not counted"),
        feature("ADC", "ADC channels including internal temperature sensor", adc_channels, "external inputs plus internal temperature sensor"),
        feature("IOs", "GPIO", gpio),
        feature("UART", "UART interfaces", 2),
        feature("SPI", "SPI interfaces", 2),
        feature("I2C", "I2C interfaces", 2),
        feature("PWM", "PWM channels", pwm_slices * 2, f"{pwm_slices} PWM slices, two outputs per slice"),
        feature("USB", "USB 1.1 controller", 1, "USB device and host support"),
        feature("USBD", "USB Device", 1),
        feature("USBH", "USB Host", 1),
        feature("PIO", "Programmable I/O (PIO)", pio_sms, f"{pio_blocks} PIO blocks; {pio_sms} state machines"),
    ]
    if dma is not None:
        features.append(feature("DMA", "DMA channels", dma))
    else:
        features.append(feature("DMA", "DMA controller", detail="channel count not stated in the imported source"))
    features.extend([
        feature("VendorCapability", "MicroPython official rp2 port", 1),
        feature("VendorCapability", "Raspberry Pi programmable I/O (PIO)", pio_sms),
    ])
    return features


def k210_features() -> list[dict[str, str]]:
    return [
        feature("Timer", "32-bit timer modules", 3),
        feature("WDT", "Watchdog timers", 2),
        feature("RTC", "Real-time clock", 1),
        feature("IOs", "GPIOHS 32 + GPIO 8", 40, "40 GPIO controllers mapped through 48 FPIOA IO pads"),
        feature("UART", "UARTHS/UART0 plus UART1-3", 4, "high-speed UART plus three general-purpose UARTs"),
        feature("SPI", "SPI interfaces", 4),
        feature("I2C", "I2C interfaces", 3),
        feature("I2S", "I2S interfaces", 3),
        feature("DMA", "DMAC channels", 8),
        feature("Camera", "DVP camera interface", 1),
        feature("NPU", "KPU neural network processor", 1, "fixed-point convolution, pooling and activation engine"),
        feature("Accelerator", "APU audio processor", 1, "up to eight audio input channels; integrated FFT"),
        feature("Accelerator", "FFT accelerator", 1, "512-point audio FFT"),
        feature("Crypto", "AES-128-CBC hardware accelerator", 1),
        feature("Crypto", "SHA-256 hardware accelerator", 1),
        feature("VendorCapability", "FPIOA function multiplexer", 1, "255 internal functions to 48 IO pads"),
        feature("VendorCapability", "Kendryte K210 official MicroPython/OpenMV manual", 1),
        feature("VendorCapability", "External SPI NOR/NAND flash boot; flash capacity depends on fitted memory", 1),
    ]


def k230_features(*, gpio: int, package_note: str) -> list[dict[str, str]]:
    return [
        feature("Timer", "timer subsystem", detail="official datasheet confirms WDT / RTC / Timer; timer instance count not published"),
        feature("WDT", "watchdog timer", 1),
        feature("RTC", "RTC", 1, "32 kHz RTC function"),
        feature("ADCUnits", "audio ADC converter units", 2, "8-192 kHz audio ADC; not GPIO ADC"),
        feature("DAC", "audio DAC converter units", 2, "8-192 kHz audio DAC"),
        feature("IOs", "GPIO", gpio, package_note),
        feature("UART", "UART interfaces", 5),
        feature("I2C", "I2C interfaces", 5),
        feature("PWM", "PWM interfaces", 6),
        feature("SPI", "SPI interfaces", 3, "1 OSPI + 2 QSPI"),
        feature("USBOTG", "USB 2.0 OTG", 2),
        feature("SDIO", "SD/eMMC controllers", 2, "SD3.01 / eMMC 5.0"),
        feature("I2S", "I2S audio interface", 1),
        feature("Camera", "MIPI CSI interfaces", 3),
        feature("Display", "MIPI DSI interface", 1),
        feature("DMA", "PDMA / SDMA", detail="DMA engines present; channel totals not published in the imported summary"),
        feature("NPU", "KPU knowledge process unit", 1, "INT8 and INT16 AI acceleration"),
        feature("Accelerator", "AI 2D engine", 1),
        feature("Accelerator", "2D GDMA engine", 1),
        feature("Accelerator", "DPU 3D depth processing unit", 1),
        feature("Accelerator", "2.5D GPU / D GPU", 1),
        feature("Accelerator", "4096-point FFT/IFFT", 1),
        feature("Crypto", "PUF / OTP / TRNG / AES / SHA / RSA / SM2/3/4 / ECC", 1),
        feature("VendorCapability", "CanMV K230 MicroPython API", 1, "official tutorial states CanMV provides C++ and MicroPython APIs"),
    ]


def k510_features() -> list[dict[str, str]]:
    return [
        feature("VendorCapability", "Kendryte K510 official Linux/SDK platform", 1),
        feature("UART", "UART resources observed by the official SDK demo", 2, "UART0 and UART1 application examples; chip-wide count not published in the imported public source"),
        feature("PWM", "PWM resources observed by the official SDK demo", 2, "pwmchip0 and pwmchip3 driver nodes; not treated as total chip channel count"),
        feature("IOs", "GPIO", detail="GPIO/IOMUX are exposed on the official CRB; chip-wide count not published in the imported public source"),
        feature("I2C", "I2C expansion interface", detail="present on official K510 CRB; chip-wide count not published"),
        feature("SPI", "SPI expansion interface", detail="present on official K510 CRB; chip-wide count not published"),
        feature("Camera", "MIPI CSI / DVP camera interfaces", 1, "official CRB routes camera interfaces; exact chip total not published"),
        feature("Display", "MIPI DSI display interface", 1, "official CRB routes MIPI DSI"),
        feature("VendorCapability", "MicroPython / CanMV support not verified in official K510 sources", 1),
        feature("VendorCapability", "External DDR / eMMC / NAND are board configuration, not on-chip SRAM", 1),
    ]


def rp2354_features(*, gpio: int, adc_channels: int, variant: str) -> list[dict[str, str]]:
    return rp_features(adc_channels=adc_channels, gpio=gpio, dma=None, pio_blocks=3, pio_sms=12, pwm_slices=12) + [
        feature("Memory", "Integrated stacked 2 MiB flash", 2, "RP2354 stacked-flash variant; RP2350 reference has external flash"),
        feature("VendorCapability", f"RP2354{variant} stacked-flash hardware", 1),
        feature("VendorCapability", "MicroPython rp2 support under review for stacked flash", 1, "mainline issue #19108 tracks RP2354B boot compatibility"),
    ]


DEVICES: list[dict[str, Any]] = [
    {
        "name": "RP2040",
        "source_ids": ["raspberrypi:rp2040-datasheet", "micropy:rp2-port"],
        "family": "Raspberry Pi",
        "series": "Raspberry Pi RP",
        "line": "RP2040",
        "variant": "",
        "part": "RP2040",
        "core": processor("Cortex-M0+", architecture="ARMv6-M", count=2, fpu="0", description="dual-core Cortex-M0+"),
        "clock": 133_000_000,
        "ram": 264 * 1024,
        "package": "QFN56 7x7 mm",
        "pins": "56",
        "features": rp_features(adc_channels=5, gpio=30, dma=12, pio_blocks=2, pio_sms=8, pwm_slices=8)
        + [feature("VendorCapability", "External QSPI flash interface", 1, "flash is external to the RP2040 die")],
        "memory": [{"name": "SRAM", "type": "RAM", "bytes": 264 * 1024}],
        "documents": [
            {"title": "RP2040 Datasheet", "url": SOURCES["raspberrypi:rp2040-datasheet"]["url"]},
            {"title": "MicroPython rp2 port", "url": SOURCES["micropy:rp2-port"]["url"]},
        ],
    },
    {
        "name": "RP2350A",
        "source_ids": ["raspberrypi:rp2350-datasheet", "micropy:rp2-port"],
        "family": "Raspberry Pi",
        "series": "Raspberry Pi RP",
        "line": "RP2350",
        "variant": "A",
        "part": "RP2350A",
        "core": processor(
            "Cortex-M33 / Hazard3 RISC-V", architecture="ARMv8-M / RISC-V", count=2,
            fpu="unknown", description="selectable dual Arm Cortex-M33 or dual Hazard3 RISC-V cores",
        ),
        "clock": 150_000_000,
        "ram": 520 * 1024,
        "package": "QFN60 7x7 mm",
        "pins": "60",
        "features": rp_features(adc_channels=5, gpio=30, dma=None, pio_blocks=3, pio_sms=12, pwm_slices=12)
        + [feature("VendorCapability", "External QSPI flash interface", 1, "flash is external to the RP2350 die")],
        "memory": [{"name": "SRAM", "type": "RAM", "bytes": 520 * 1024}],
        "documents": [
            {"title": "RP2350 Datasheet", "url": SOURCES["raspberrypi:rp2350-datasheet"]["url"]},
            {"title": "MicroPython rp2 port", "url": SOURCES["micropy:rp2-port"]["url"]},
        ],
    },
    {
        "name": "RP2350B",
        "source_ids": ["raspberrypi:rp2350-datasheet", "micropy:rp2-port"],
        "family": "Raspberry Pi",
        "series": "Raspberry Pi RP",
        "line": "RP2350",
        "variant": "B",
        "part": "RP2350B",
        "core": processor(
            "Cortex-M33 / Hazard3 RISC-V", architecture="ARMv8-M / RISC-V", count=2,
            fpu="unknown", description="selectable dual Arm Cortex-M33 or dual Hazard3 RISC-V cores",
        ),
        "clock": 150_000_000,
        "ram": 520 * 1024,
        "package": "QFN80 10x10 mm",
        "pins": "80",
        "features": rp_features(adc_channels=9, gpio=48, dma=None, pio_blocks=3, pio_sms=12, pwm_slices=12)
        + [feature("VendorCapability", "External QSPI flash interface", 1, "flash is external to the RP2350 die")],
        "memory": [{"name": "SRAM", "type": "RAM", "bytes": 520 * 1024}],
        "documents": [
            {"title": "RP2350 Datasheet", "url": SOURCES["raspberrypi:rp2350-datasheet"]["url"]},
            {"title": "MicroPython rp2 port", "url": SOURCES["micropy:rp2-port"]["url"]},
        ],
    },
    {
        "name": "K210",
        "source_ids": ["kendryte:k210-datasheet", "kendryte:k210-package", "kendryte:k210-micropython"],
        "family": "Kendryte",
        "series": "Kendryte K",
        "line": "K210",
        "variant": "",
        "part": "K210",
        "core": processor(
            "dual RV64GC RISC-V", architecture="RISC-V RV64GC", count=2, fpu="yes", dsp="unknown",
            description="two 64-bit RISC-V cores with independent double-precision FPUs; 400 MHz",
        ),
        "clock": 400_000_000,
        "ram": 8 * 1024 * 1024,
        "package": "BGA144 8x8 mm",
        "pins": "144",
        "features": k210_features(),
        "memory": [
            {"name": "General-purpose SRAM", "type": "RAM", "bytes": 6 * 1024 * 1024},
            {"name": "AI SRAM", "type": "RAM", "bytes": 2 * 1024 * 1024},
            {"name": "External SPI NOR/NAND flash", "type": "ExternalFlash", "sizes": "8/16/32 MiB NOR; 64/128/256 MiB NAND"},
        ],
        "documents": [
            {"title": "K210 Functional Description", "url": SOURCES["kendryte:k210-datasheet"]["url"]},
            {"title": "K210 Package Information", "url": SOURCES["kendryte:k210-package"]["url"]},
            {"title": "K210 MicroPython/OpenMV Manual", "url": SOURCES["kendryte:k210-micropython"]["url"]},
        ],
    },
    {
        "name": "K230",
        "source_ids": ["kendryte:k230-datasheet", "kendryte:k230-canmv"],
        "family": "Kendryte",
        "series": "Kendryte K",
        "line": "K230",
        "variant": "",
        "part": "K230",
        "core": processor(
            "dual C908 RISC-V", architecture="RISC-V RV64", count=2, fpu="yes", dsp="unknown",
            description="CPU0 up to 800 MHz Linux; CPU1 up to 1.6 GHz RISC-V Vector RTOS core; both have FPU",
        ),
        "clock": 1_600_000_000,
        "ram": 4 * 1024 * 1024,
        "package": "BGA 13x13 mm",
        "pins": "",
        "features": k230_features(gpio=72, package_note="64 GPIO + 8 PMU GPIO; pins are multiplexed with function interfaces"),
        "memory": [
            {"name": "System SRAM", "type": "RAM", "bytes": 2 * 1024 * 1024},
            {"name": "KPU SRAM", "type": "RAM", "bytes": 2 * 1024 * 1024},
            {"name": "External DDR", "type": "ExternalRAM", "sizes": "DDR3L/LPDDR3/LPDDR4; capacity depends on fitted memory"},
        ],
        "documents": [
            {"title": "K230 Full Datasheet", "url": SOURCES["kendryte:k230-datasheet"]["url"]},
            {"title": "CanMV K230 Tutorial", "url": SOURCES["kendryte:k230-canmv"]["url"]},
        ],
    },
    {
        "name": "K230D",
        "source_ids": ["kendryte:k230-datasheet", "kendryte:k230-canmv"],
        "family": "Kendryte",
        "series": "Kendryte K",
        "line": "K230",
        "variant": "D",
        "part": "K230D",
        "core": processor(
            "dual C908 RISC-V", architecture="RISC-V RV64", count=2, fpu="yes", dsp="unknown",
            description="K230D uses the K230 compute die with integrated 1 Gb LPDDR4; CPU0 up to 800 MHz and CPU1 up to 1.6 GHz",
        ),
        "clock": 1_600_000_000,
        "ram": 4 * 1024 * 1024,
        "package": "BGA 11x11 mm SIP with 1 Gb LPDDR4",
        "pins": "",
        "features": k230_features(gpio=56, package_note="50 GPIO + 6 PMU GPIO for SIP package; pins are multiplexed with function interfaces")
        + [feature("Memory", "Integrated 1 Gb LPDDR4", 1, "K230D package-specific SIP memory; not counted as on-chip SRAM")],
        "memory": [
            {"name": "System SRAM", "type": "RAM", "bytes": 2 * 1024 * 1024},
            {"name": "KPU SRAM", "type": "RAM", "bytes": 2 * 1024 * 1024},
            {"name": "Integrated LPDDR4", "type": "SiPDDR", "bytes": 128 * 1024 * 1024},
        ],
        "documents": [
            {"title": "K230 Full Datasheet (K230/K230D package section)", "url": SOURCES["kendryte:k230-datasheet"]["url"]},
            {"title": "CanMV K230 Tutorial", "url": SOURCES["kendryte:k230-canmv"]["url"]},
        ],
    },
    {
        "name": "K510",
        "source_ids": ["kendryte:k510-docs"],
        "family": "Kendryte",
        "series": "Kendryte K",
        "line": "K510",
        "variant": "",
        "part": "K510",
        "core": [{
            "Darchitecture": "RISC-V",
            "Dcore": "Unpublished K510 heterogeneous application processor",
            "DcoreCount": "",
            "Dfpu": "unknown",
            "Ddsp": "unknown",
            "DsourceDescription": "exact core count not stated in the imported official public source",
        }],
        "clock": "",
        "ram": "",
        "package": "",
        "pins": "",
        "features": k510_features(),
        "memory": [],
        "documents": [
            {"title": "K510 official documentation index", "url": SOURCES["kendryte:k510-docs"]["url"]},
            {"title": "K510 CRB V1.2 Hardware Guide", "url": "https://raw.githubusercontent.com/kendryte/k510_docs/dev/en/K510_CRB_V1.2_Hardware_Guide.md"},
        ],
    },
    {
        "name": "RP2354A",
        "source_ids": ["raspberrypi:rp2354-hardware", "raspberrypi:rp2354-micropython"],
        "family": "Raspberry Pi",
        "series": "Raspberry Pi RP",
        "line": "RP2354",
        "variant": "A",
        "part": "RP2354A",
        "core": processor(
            "Cortex-M33 / Hazard3 RISC-V", architecture="ARMv8-M / RISC-V", count=2,
            fpu="unknown", description="RP2350-family dual Arm Cortex-M33 or dual Hazard3 RISC-V with stacked flash",
        ),
        "clock": 150_000_000,
        "ram": 520 * 1024,
        "package": "QFN60 7x7 mm",
        "pins": "60",
        "flash": 2 * 1024 * 1024,
        "features": rp2354_features(gpio=30, adc_channels=5, variant="A"),
        "memory": [{"name": "SRAM", "type": "RAM", "bytes": 520 * 1024}, {"name": "Stacked flash", "type": "Flash", "bytes": 2 * 1024 * 1024}],
        "documents": [
            {"title": "RP2350 family Datasheet", "url": SOURCES["raspberrypi:rp2354-hardware"]["url"]},
            {"title": "MicroPython RP2354B support issue", "url": SOURCES["raspberrypi:rp2354-micropython"]["url"]},
        ],
    },
    {
        "name": "RP2354B",
        "source_ids": ["raspberrypi:rp2354-hardware", "raspberrypi:rp2354-micropython"],
        "family": "Raspberry Pi",
        "series": "Raspberry Pi RP",
        "line": "RP2354",
        "variant": "B",
        "part": "RP2354B",
        "core": processor(
            "Cortex-M33 / Hazard3 RISC-V", architecture="ARMv8-M / RISC-V", count=2,
            fpu="unknown", description="RP2350-family dual Arm Cortex-M33 or dual Hazard3 RISC-V with stacked flash",
        ),
        "clock": 150_000_000,
        "ram": 520 * 1024,
        "package": "QFN80 10x10 mm",
        "pins": "80",
        "flash": 2 * 1024 * 1024,
        "features": rp2354_features(gpio=48, adc_channels=9, variant="B"),
        "memory": [{"name": "SRAM", "type": "RAM", "bytes": 520 * 1024}, {"name": "Stacked flash", "type": "Flash", "bytes": 2 * 1024 * 1024}],
        "documents": [
            {"title": "RP2350 family Datasheet", "url": SOURCES["raspberrypi:rp2354-hardware"]["url"]},
            {"title": "MicroPython RP2354B support issue", "url": SOURCES["raspberrypi:rp2354-micropython"]["url"]},
        ],
    },
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "vendor-packs" / "micropy-mcu")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "micropy-mcu")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true", help="write records using the pinned official URLs without downloading sources")
    args = parser.parse_args()

    snapshots: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    if not args.skip_fetch:
        fetcher = OfficialFetcher(
            cache_dir=args.cache_dir,
            allowed_hosts={"datasheets.raspberrypi.com", "raw.githubusercontent.com", "github.com", "api.github.com", "pip.raspberrypi.com", "pip-assets.raspberrypi.com"},
            refresh=args.refresh,
            timeout=60,
            retries=3,
        )
        for source_id, source in SOURCES.items():
            try:
                snapshots[source_id] = fetcher.fetch(source["url"], cache_name=source["cache"])
            except Exception as exc:  # retain the record and make the fetch gap explicit in the report
                errors.append({"scope": source_id, "source_url": source["url"], "item": source["title"], "error": str(exc)})

    observed = utc_now()
    devices: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    for item in DEVICES:
        source_ids = item["source_ids"]
        source_versions = []
        for source_id in source_ids:
            source = SOURCES[source_id]
            snapshot = snapshots.get(source_id)
            source_versions.append(f"{source['version']};sha256:{snapshot.sha256}" if snapshot else source["version"])
        device_id = f"micropy-mcu::{slug(item['name'])}"
        line_id = f"micropy-mcu::{slug(item['series'])}::{slug(item['line'])}"
        devices.append({
            "device_id": device_id,
            "product_line_id": line_id,
            "manufacturer": MANUFACTURER,
            "product_type": "micropython_mcu",
            "architecture_class": item["core"][0]["Darchitecture"],
            "family": item["family"],
            "series": item["series"],
            "product_line": item["line"],
            "device_name": item["name"],
            "generic_device_name": item["line"],
            "manufacturer_variant_code": item["variant"],
            "processor_cores": json.dumps(item["core"], ensure_ascii=False),
            "max_clock_hz": item["clock"],
            "flash_bytes": item.get("flash", ""),
            "ram_bytes": item["ram"],
            "package_types": item["package"],
            "pin_counts": item["pins"],
            "memory_regions_json": json.dumps(item["memory"], ensure_ascii=False),
            "features_json": json.dumps(item["features"], ensure_ascii=False),
            "documents_json": json.dumps(item["documents"], ensure_ascii=False),
            "svd_files": "",
            "lifecycle": "active",
            "source_id": ";".join(source_ids),
            "source_url": item["documents"][0]["url"],
            "source_version": ";".join(source_versions),
            "observed_at": observed,
            "verification_status": "manufacturer_datasheet",
        })
        parts.append({
            "orderable_part_id": f"micropy-mcu::{slug(item['part'])}",
            "device_id": device_id,
            "manufacturer": MANUFACTURER,
            "family": item["family"],
            "series": item["series"],
            "product_line": item["line"],
            "device_name": item["name"],
            "part_number": item["part"],
            "manufacturer_suffix": item["variant"],
            "package_code": "",
            "temperature_grade_code": "",
            "packing_code": "",
            "package_name": item["package"],
            "temperature_range": "",
            "packing_form": "",
            "lifecycle": "active",
            "source_id": ";".join(source_ids),
            "source_url": item["documents"][0]["url"],
            "observed_at": observed,
            "verification_status": "manufacturer_datasheet",
            "decode_status": "exact_official_model;package_variants_kept_separate",
        })

    lines = []
    for line_id in sorted({row["product_line_id"] for row in devices}):
        line_devices = [row for row in devices if row["product_line_id"] == line_id]
        source_ids = sorted({source_id for row in line_devices for source_id in row["source_id"].split(";")})
        lines.append({
            "product_line_id": line_id,
            "manufacturer": MANUFACTURER,
            "product_type": "micropython_mcu",
            "architecture_class": line_devices[0]["architecture_class"],
            "family": line_devices[0]["family"],
            "series": line_devices[0]["series"],
            "product_line": line_devices[0]["product_line"],
            "device_variant_count": len(line_devices),
            "source_ids": ";".join(source_ids),
            "observed_at": observed,
            "verification_status": "manufacturer_datasheet",
        })

    source_rows = []
    for source_id, source in SOURCES.items():
        snapshot = snapshots.get(source_id)
        source_rows.append({
            "source_id": source_id,
            "source_type": source["source_type"],
            "publisher": source["publisher"],
            "title": source["title"],
            "url": source["url"],
            "version": f"{source['version']};sha256:{snapshot.sha256}" if snapshot else source["version"],
            "observed_at": snapshot.observed_at if snapshot else observed,
            "verification_scope": source["scope"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "product-lines.csv", [
        "product_line_id", "manufacturer", "product_type", "architecture_class", "family", "series",
        "product_line", "device_variant_count", "source_ids", "observed_at", "verification_status",
    ], lines)
    write_csv(args.output_dir / "device-variants.csv", DEVICE_FIELDS, sorted(devices, key=lambda row: row["device_name"]))
    write_csv(args.output_dir / "orderable-parts.csv", PART_FIELDS, sorted(parts, key=lambda row: row["part_number"]))
    write_csv(args.output_dir / "sources.csv", SOURCE_FIELDS, source_rows)
    write_csv(args.output_dir / "import-errors.csv", ERROR_FIELDS, errors)
    report = {
        "status": "ok" if not errors else "ok_with_source_fetch_errors",
        "adapter": "import_micropy_mcu.py",
        "manufacturer": MANUFACTURER,
        "device_variants": len(devices),
        "orderable_parts": len(parts),
        "families": sorted({row["family"] for row in devices}),
        "product_lines": sorted({row["product_line"] for row in devices}),
        "exact_models": sorted(row["device_name"] for row in devices),
        "source_fetch_errors": len(errors),
        "completeness_claim": "MicroPython MCU ecosystem scope: Raspberry Pi RP2040/RP2350 and stacked-flash RP2354 variants, Kendryte K210, and CanMV K230/K230D. K510 is included as a requested ecosystem candidate, but its public official sources do not verify a MicroPython port; unknown chip-level fields remain blank and board memory is not attributed to the silicon.",
    }
    write_json(args.output_dir / "official-adapter-report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
