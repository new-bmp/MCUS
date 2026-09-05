# MCU-L catalog workspace

This directory turns the MCU-L coverage policy into a reproducible catalog.
Generated CSV files are suitable for import into SQLite/Room or another search
index used by the Android app.

## What is represented

The catalog keeps three identifiers separate:

1. **Product line** — for example `STM32F103`.
2. **Device variant** — for example `STM32F103C8`; `C8` is preserved as the
   manufacturer variant code.
3. **Orderable part** — for example `STM32F103C8T6` or
   `STM32F103C8T6TR`.

An orderable part is never produced as a Cartesian product of code positions.
It is added only when the complete string is found on a manufacturer page or
another explicitly recorded source.

## Generated files

- `data/cmsis-pack-index.csv`: all Device Family Packs discovered in the
  current CMSIS-Pack index, including deprecated/replacement metadata.
- `data/product-lines.csv`: normalized manufacturer → family → series →
  product-line rows.
- `data/device-variants.csv`: actual device/variant names and the specifications
  present in the source PDSC files.
- `data/orderable-parts.csv`: complete purchasing codes confirmed on official
  manufacturer product pages.
- `data/sources.csv`: source/version/observation provenance.
- `data/coverage-manifest.csv`: counts and explicit coverage status by vendor.
- `data/import-errors.csv`: failed PDSC downloads or parse failures; an empty
  file means the selected import completed without known fetch errors.

## Verification states

- `cmsis_pack_metadata`: a name or specification is present in a CMSIS Device
  Family Pack. This is strong evidence for device metadata, but not a claim
  that all purchasing suffixes are covered.
- `manufacturer_product_page`: the complete purchasing code was present on the
  manufacturer's own product page.
- `unknown`: no source was allowed to fill the value by inference.

## Rebuild

```powershell
python scripts/import_cmsis_packs.py --scope stm32
python scripts/import_st_orderable_parts.py --series-prefix STM32F1
python scripts/validate_catalog.py
```

Use `--refresh` when a fresh network snapshot is required. Cached files are
ignored by version control and only reduce repeated downloads.

Vendor data packs can be generated independently, which is also the intended
mobile-update unit:

```powershell
python scripts/import_cmsis_packs.py --scope all-dfp --vendor Microchip `
  --output-dir data/vendor-packs/microchip
python scripts/merge_catalog.py
python scripts/enrich_catalog.py --data-dir data/combined
python scripts/validate_catalog.py --data-dir data/combined
```

Manufacturer-scoped official adapters are available for the current Arm
catalog boundary:

```powershell
python scripts/import_st_official.py
python scripts/import_ti_official.py
python scripts/import_ti_c2000_official.py
python scripts/import_microchip_official.py
python scripts/import_gigadevice_official.py
python scripts/import_qinheng_official.py
python scripts/import_stc32_official.py
python scripts/import_espressif_products.py
python scripts/import_hpmicro_official.py
python scripts/import_allwinner_official.py
python scripts/import_renesas_official.py
python scripts/augment_infineon_from_device_db.py
python scripts/augment_espressif_from_idf_soc_caps.py
```

Each adapter writes an `official-adapter-report.json` or `import-report.json`
with the selected official scope, failed items, source hosts, snapshot hashes,
and a non-completeness claim. CMSIS-Pack rows are device metadata; complete
orderable suffixes are only added by a source-specific part-number importer.
The Espressif adapter uses the official Product Selector API and keeps SoCs and
modules separate, including Xtensa/RISC-V, wireless, RMT/MCPWM/LEDPWM, PSRAM,
RTC SRAM, and other vendor-specific capabilities.

The TI C2000 adapter imports exact, manufacturer-documented C28x real-time
controllers from TI C2000Ware and their individual data sheets. These records
are classified as `dsp_mcu` / `C28x DSP`, not Arm MCU records. FPU, TMU, VCU,
CLA, ADC converter/channel counts, PWM, serial buses, memory structure and
the optional Cortex-M4 connectivity manager are retained only where the exact
TI data sheet states them.

The Infineon augmenter uses the official ModusToolbox `device-db` snapshot. It
supports a local cache, retries, `--proxy`, and an optional pre-downloaded
`--archive`. Exact leaf MCU records provide package, clock, flash, SRAM,
availability, parameterized peripherals, vendor capabilities, and published
part numbers. Lowercase-`x` wildcard device patterns are never counted as
orderable parts.

The HPMicro adapter discovers every current official MCU selection table from
the HPMicro sitemap. It imports exact selector model rows separately from exact
purchase-card strings, and supplements only matching product lines with their
own HPM SDK SoC IP header. Vendor blocks such as PLA, PLB, SEI, MMC, PSEC,
Pixelmux, ESC, FFA, QEIv2/QEO and MTG retain their manufacturer names. The
official 12-bit and 16-bit ADC columns are treated as converter units; they are
not converted into ADC pins or guessed channel counts.

The Allwinner adapter is MCU-only: it imports the eight standalone XRadio
wireless MCU parts (XR806/XR808/XR809/XR871/XR872). A/T/MR/V application and
real-time SoCs, R128, and XR819/XR829 connectivity chips remain in the source
file as an audited reference but are excluded from the MCU directory. ADC
converter units remain separate from ADC channels, and wildcard page examples
never become orderable parts.

The Renesas adapter reads the live official product selectors for RA, RX,
RL78, RH850, Synergy and Renesas RISC-V MCUs. It keeps exact base devices and
complete orderable codes separate, uses product-group values only when the
official selector has one unambiguous value, and supplements matching RA
devices with the official RA CMSIS pack. RZ/Five and all RZ MPU/SoC families
are explicitly outside the MCU catalog scope.

The `import_micropy_mcu.py` adapter provides an independent `MicroPy MCU`
ecosystem entry for chips with an official MicroPython/CanMV route or an
explicitly requested ecosystem candidate. It contains Raspberry Pi RP2040,
RP2350A/B, stacked-flash RP2354A/B, Kendryte K210 and CanMV K230/K230D. The
K510 record is retained with only the resources confirmed by its public
official SDK/CRB documentation; the importer does not claim an official K510
MicroPython port. Package variants remain separate when GPIO, ADC or memory
resources differ, and unknown chip-level values stay blank.

The Espressif IDF augmenter reads each supported target's official
`soc_caps.h`. It fills per-SoC capability/count macros and maps modules through
their Product Selector series. General-purpose timers, LEDC and MCPWM remain
separate; a support macro is not converted into an invented timer count.

`device-capabilities.csv` contains flattened core, clock, timer, analog,
connectivity, accelerator, vendor-specific feature, and explicit operating
voltage range fields (`operating_voltage_min_v` / `operating_voltage_max_v`).
It also keeps a
source-backed `peripheral_inventory_json` so less common resources such as I²S,
LIN, SDIO, watchdogs, RTC, comparators, op-amps, touch, display, camera,
external buses and security blocks are not discarded.
The mobile catalog additionally extracts explicitly named external-memory
interfaces into `eb`, including FMC/FSMC, OCTOSPI/OSPI, QSPI, XSPI, FlexSPI,
HyperBus, EMIF and EBI; this is a source-name index, not an inferred bus count.

Explicit source-backed current and power figures are stored separately in
`power_measurements_json`. Every record keeps its operating mode, value, unit,
typical/maximum status, disclosed test conditions and source document ID.
Unitless `Consumption` values and low-power mode counts are never interpreted
as electrical measurements. Devices without an explicit A/W unit remain
`not_found`, and measurements with incomplete voltage, temperature or clock
conditions are shown as such instead of being silently compared.

The cached official PDF text snapshots can be rechecked with
`augment_power_from_cached_datasheets.py`. It currently covers the Artery and
GigaDevice datasheets present in `cache/`, applies suffix-scoped tables such as
GD32F103x4/6/8/B versus GD32F103xC/D/E/F/G/I/K to the matching devices, and
writes an audit report to `data/combined/power-augmentation-report.json`.

ADC converter units, ADC channels and the raw CMSIS ADC quantity parameter are
separate fields. ADC-capable pin counts are deliberately not derived or used as
an MCUS selection metric; raw pinout evidence remains only in source records.
`device-scores.csv` keeps the MCU-L derived selection index separate from empty
benchmark fields. See `SCORING_MODEL.md` for the formula and limitations.

## Current boundary

CMSIS-Pack is a broad source for Arm-based MCUs. It does not prove complete
coverage of PIC, AVR, 8051, RL78, RX, or other proprietary/non-Arm catalogs.
Those require manufacturer-specific import adapters; TI C2000 is covered by
the dedicated official adapter above. Likewise, a
device row such as `STM32F103C8` is not counted as complete orderable-part
coverage until every official package/temperature/packing code is collected.

The STC32 adapter reads the official STC Micro STC32G family and product pages.
It imports the exact models named by the public selection tables, including
STC32G12K64/G12K128, STC32G8K48/G8K64, STC32CL8K48/CL8K64 and STC32G144K246.
The catalog preserves the 8051-compatible 1T core, MDU32, DSP32/TFPU, UART /
USART, timer, ADC channel and ADC-unit distinctions, and records STC32G18K64
and STC32G96K246 only as official page mentions because their individual
public specification table is not present on the linked page.

## Release metadata

`release-metadata.csv` stores optional per-device release observations. The
`collect_release_metadata.py` adapter currently imports exact Espressif MPNs
from the official Product Selector snapshot. Its `releaseTime` values are
normalized to `YYYY-MM-DD` and `YYYY/Qx`. Selector prices are stored with the
status `official_product_selector_listed_price`; the UI presents these as
“official reference price” because the source does not state that they are
launch prices. A value is shown as “launch price” only when a future source
explicitly marks it as a launch/release price.
