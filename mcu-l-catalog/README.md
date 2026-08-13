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
python scripts/import_microchip_official.py
python scripts/import_gigadevice_official.py
python scripts/import_espressif_products.py
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

The Infineon augmenter uses the official ModusToolbox `device-db` snapshot. It
supports a local cache, retries, `--proxy`, and an optional pre-downloaded
`--archive`. Exact leaf MCU records provide package, clock, flash, SRAM,
availability, parameterized peripherals, vendor capabilities, and published
part numbers. Lowercase-`x` wildcard device patterns are never counted as
orderable parts.

The Espressif IDF augmenter reads each supported target's official
`soc_caps.h`. It fills per-SoC capability/count macros and maps modules through
their Product Selector series. General-purpose timers, LEDC and MCPWM remain
separate; a support macro is not converted into an invented timer count.

`device-capabilities.csv` contains flattened core, clock, timer, analog,
connectivity, accelerator, and vendor-specific feature fields. It also keeps a
source-backed `peripheral_inventory_json` so less common resources such as I²S,
LIN, SDIO, watchdogs, RTC, comparators, op-amps, touch, display, camera,
external buses and security blocks are not discarded.

ADC converter units, ADC channels and the raw CMSIS ADC quantity parameter are
separate fields. ADC-capable pin counts are deliberately not derived or used as
an MCUS selection metric; raw pinout evidence remains only in source records.
`device-scores.csv` keeps the MCU-L derived selection index separate from empty
benchmark fields. See `SCORING_MODEL.md` for the formula and limitations.

## Current boundary

CMSIS-Pack is a broad source for Arm-based MCUs. It does not prove complete
coverage of PIC, AVR, 8051, RL78, RX, C2000, or other proprietary/non-Arm
catalogs. Those require manufacturer-specific import adapters. Likewise, a
device row such as `STM32F103C8` is not counted as complete orderable-part
coverage until every official package/temperature/packing code is collected.
