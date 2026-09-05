#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def number(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def text(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def first_text(*values):
    for value in values:
        normalized = text(value)
        if normalized:
            return normalized
    return None


def decimal(value):
    if value is None or value == "":
        return None
    try:
        number_value = float(value)
    except (TypeError, ValueError):
        return None
    return number_value if number_value >= 0 else None


def json_value(value):
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def memory_regions(value):
    """Keep compact memory-map evidence for selection and detail views."""
    result = []
    seen = set()
    for item in json_value(value):
        if not isinstance(item, dict):
            continue
        name = text(item.get("name"))
        if not name:
            continue
        size_raw = text(item.get("size"))
        try:
            size = int(size_raw, 0) if size_raw else None
        except (TypeError, ValueError):
            size = None
        access = text(item.get("access"))
        key = (name, size, access)
        if key in seen:
            continue
        seen.add(key)
        record = {"n": name}
        if size is not None and size > 0:
            record["s"] = size
        if access:
            record["a"] = access
        if text(item.get("alias")):
            record["alias"] = text(item.get("alias"))
        result.append(record)
    return result


def feature_labels(value):
    result = []
    for item in json_value(value):
        if isinstance(item, str):
            label = item.strip()
        elif isinstance(item, dict):
            source_feature = item.get("source_feature")
            source_name = source_feature.get("name") if isinstance(source_feature, dict) else ""
            label = str(source_name or item.get("name") or item.get("feature_id") or item.get("category") or "").strip()
        else:
            label = ""
        if label and label not in result:
            result.append(label)
    return result


def document_records(value):
    """Normalize manuals, product pages, and pack references without conflating them."""
    result = []
    seen = set()
    for item in json_value(value):
        if not isinstance(item, dict):
            continue
        raw_name = text(item.get("name"))
        title = text(item.get("title")) or raw_name or text(item.get("Dname"))
        url = text(item.get("url")) or text(item.get("href"))
        if not url and raw_name and raw_name.startswith(("http://", "https://")):
            url = raw_name
        if url:
            url = re.sub(r"(?i)(?:\.pdf){2,}(?=\?|$)", ".pdf", url)
        path = text(item.get("file")) or (raw_name if raw_name and not raw_name.startswith(("http://", "https://")) else None)
        if not title and not url and not path:
            continue
        # Relative CMSIS-Pack book paths are useful evidence, but are not
        # converted to fake web links. The UI presents them as pack paths.
        if url and not url.startswith(("http://", "https://")):
            path = path or url
            url = None
        explicit_kind = text(item.get("kind"))
        evidence = " ".join(filter(None, (title, url, path))).lower()
        product_page_url = bool(url) and (
            "/product/" in url.lower()
            or "#documentation" in url.lower()
            or "product-selector" in url.lower()
        )
        if product_page_url:
            kind = "product_page"
        elif explicit_kind:
            kind = explicit_kind
        elif "datasheet" in evidence or "data sheet" in evidence or "数据手册" in evidence:
            kind = "datasheet"
        elif "reference manual" in evidence or "参考手册" in evidence or "technical reference" in evidence:
            kind = "reference_manual"
        elif "user manual" in evidence or "user guide" in evidence or "用户手册" in evidence:
            kind = "user_manual"
        elif path or any(token in evidence for token in (".pdsc", ".pack", ".atpack", "cmsis driver", "github.com")):
            kind = "source_pack"
        elif "product page" in evidence or "/product/" in evidence or "product-selector" in evidence:
            kind = "product_page"
        else:
            kind = "source"
        record = {}
        if title:
            record["title"] = title
        if url:
            record["url"] = url
        if path and path != url:
            record["path"] = path
        if text(item.get("version")):
            record["version"] = text(item.get("version"))
        record["kind"] = kind
        for source_field, target_field in (
            ("verification_status", "status"),
            ("http_status", "http"),
            ("checked_at", "checked"),
        ):
            source_value = item.get(source_field)
            if source_value is not None and source_value != "":
                record[target_field] = source_value
        key = tuple(record.get(field, "") for field in ("title", "url", "path", "version", "kind"))
        if key != ("", "", "", "", "") and key not in seen:
            seen.add(key)
            result.append(record)
    return result


def inventory_quantity(value, feature_type, preferred_source_kind=None):
    matches = [
        item for item in json_value(value)
        if isinstance(item, dict) and item.get("type") == feature_type
    ]
    preferred = [
        item for item in matches
        if preferred_source_kind and item.get("source_kind") == preferred_source_kind
    ]
    values = [
        number(item.get("count") if item.get("count") not in (None, "") else item.get("n"))
        for item in (preferred or matches)
    ]
    values = [value for value in values if value is not None and value > 0]
    return max(values) if values else None


EXTERNAL_BUS_TOKENS = (
    "OCTOSPI", "OCTOSPIM", "OSPI", "XSPI", "XSPIM", "QUADSPI", "QSPI",
    "FLEXSPI", "HYPERBUS", "EMIF", "EBI", "FMC", "FSMC", "SMC", "SQI",
)


def external_bus_types(value):
    """Extract only explicitly named external-memory interfaces from source text."""
    result = []
    seen = set()
    token_pattern = re.compile(
        r"(?<![A-Z0-9])(?:OCTOSPI\d*|OCTOSPIM|OSPI\d*|XSPI\d*|XSPIM|"
        r"QUADSPI\d*|QSPI\d*|FLEXSPI\d*|HYPERBUS|EMIF\d*|EBI\d*|FMC\d*|FSMC\d*|SMC\d*|SQI\d*)(?![A-Z0-9])",
        re.IGNORECASE,
    )
    for item in json_value(value):
        if not isinstance(item, dict) or str(item.get("type") or "").lower() not in {"extbus", "comother", "other", "vendorcapability"}:
            continue
        source = " ".join(str(item.get(key) or "") for key in ("name", "n", "description"))
        for match in token_pattern.finditer(source.upper()):
            token = match.group(0).upper()
            if token in EXTERNAL_BUS_TOKENS or any(token.startswith(prefix) for prefix in EXTERNAL_BUS_TOKENS):
                if token not in seen:
                    seen.add(token)
                    result.append(token)
    return sorted(result)


INVENTORY_LABELS = {
    "Timer": "定时器", "TimerOther": "其他定时资源", "PWM": "PWM", "WDT": "看门狗",
    "RTC": "RTC", "ADC": "ADC", "ADC12": "12-bit ADC", "ADC16": "16-bit ADC",
    "A/D": "ADC", "ADCUnits": "ADC 转换器单元", "DAC": "DAC", "D/A": "DAC", "COMP": "比较器",
    "AnalogOther": "模拟外设", "TempSens": "温度传感器", "TS": "温度传感器",
    "Touch": "触摸感应", "IOs": "GPIO", "I/O": "GPIO", "ExtInt": "外部中断",
    "GPIOPort": "GPIO 端口", "GPIO": "GPIO 控制器",
    "SPI": "SPI", "I2C": "I²C", "I2S": "I²S", "USART": "USART", "UART": "UART",
    "LIN": "LIN", "CAN": "CAN", "USBD": "USB Device", "USBH": "USB Host",
    "USBOTG": "USB OTG", "USB": "USB", "ETH": "Ethernet", "ETHPhy": "Ethernet PHY", "SDIO": "SDIO / SDMMC",
    "SDIOHost": "SDIO Host", "SDIOSlave": "SDIO Slave", "I3C": "I³C", "MIPI": "MIPI",
    "MPSerial": "多功能串行模块", "Com": "通信接口", "ComOther": "其他通信接口",
    "DMA": "DMA", "ExtBus": "外部存储总线", "Camera": "摄像头接口",
    "LCD": "段码 LCD", "GLCD": "图形 LCD", "RNG": "随机数发生器",
    "Crypto": "加密单元", "NPU": "NPU", "CoreOther": "计算 / 校验单元",
    "NVIC": "NVIC", "PLL": "PLL", "XTAL": "外部晶振", "IntRC": "内部 RC",
    "PWR": "电源控制器", "RCC": "复位与时钟控制", "SYS": "系统控制", "DEBUG": "调试接口", "JTAG": "JTAG / SWD",
    "DWT": "数据观察点", "FPU": "浮点单元", "MPU": "存储保护单元",
    "ClockOther": "其他时钟", "PowerMode": "低功耗模式", "PowerOther": "电源管理",
    "Consumption": "功耗参数", "RMT": "RMT 遥控收发器", "LEDPWM": "LED PWM",
    "MCPWM": "电机控制 PWM", "WiFi": "Wi-Fi", "WiFi6": "Wi-Fi 6",
    "Bluetooth": "Bluetooth", "IEEE802154": "IEEE 802.15.4 / Thread / Zigbee",
    "PSRAM": "PSRAM", "RTC_RAM": "RTC SRAM", "Hall": "霍尔传感器", "TOF": "ToF 接口",
    "Audio": "数字音频", "Accelerator": "硬件加速器",
    "VendorCapability": "厂商专用能力",
    "DMA2D": "DMA2D / Chrom-ART", "CORDIC": "CORDIC 数学加速", "FMAC": "FMAC 滤波乘加",
    "MDF": "MDF 数字滤波器", "DFSDM": "DFSDM 数字滤波器", "JPEG": "JPEG 编解码器",
    "SAI": "SAI 音频接口", "PDMIC": "PDM 麦克风接口", "SPDIFRX": "S/PDIF 接收器",
    "CANFD": "CAN FD", "FlexRay": "FlexRay", "UCPD": "USB-C PD / UCPD", "USBPD": "USB Power Delivery",
    "USBSS": "USB SuperSpeed", "USBHS": "USB High-Speed", "USB_OTG_HS": "USB OTG HS",
    "DMAMUX": "DMAMUX", "DMAChannels": "DMA 通道", "LINKEDLIST": "DMA 链表队列",
    "ICACHE": "指令缓存", "DCACHE": "数据缓存", "RAMECC": "RAM ECC", "FLEXRAMECC": "FlexRAM ECC",
    "OTFDEC": "在线 Flash 解密", "GTZC": "TrustZone 资源控制", "SAU": "安全属性单元", "IDAU": "实现定义安全属性",
    "AES": "AES 加密引擎", "AESB": "AES 加密引擎", "PKA": "公钥加速器", "CRCCU": "CRC 校验单元", "CRCSCAN": "Flash CRC 扫描",
    "BSEC": "安全配置 / 密钥控制", "HSM": "硬件安全模块",
    "HSEM": "硬件信号量", "EVSYS": "事件系统", "CCL": "可配置逻辑", "PDEC": "位置解码器",
    "PIO": "PIO 可编程 IO", "HSP_Engine": "HSP 专用处理引擎", "HSP1": "HSP1 专用处理模块",
    "DIVAS": "除法 / 三角数学加速", "GFX": "图形加速引擎", "VREFBUF": "内部参考电压缓冲器",
    "SDADC": "Sigma-Delta ADC", "BANDGAP": "带隙参考", "ZCD": "过零检测",
    "SUBGHZ": "Sub-GHz 无线", "BLE": "Bluetooth Low Energy", "ZIGBEE": "Zigbee",
    "HDMI_CEC": "HDMI-CEC", "SWPMI": "单线协议接口", "LIN_UART": "LIN UART",
}


def human_frequency(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value or "")
    if numeric >= 1_000_000:
        return f"{numeric / 1_000_000:g} MHz"
    if numeric >= 1_000:
        return f"{numeric / 1_000:g} kHz"
    return f"{numeric:g} Hz"


def inventory_records(value):
    result = []
    seen = set()
    frequency_types = {"SPI", "UART", "USART", "XTAL", "IntRC"}
    bit_width_types = {"Timer", "ADC", "DAC", "Camera"}
    for item in json_value(value):
        if not isinstance(item, dict):
            continue
        feature_type = str(item.get("type") or "").strip()
        if feature_type.lower() == "adcexternalpins":
            continue
        raw_n = item.get("count") if item.get("count") not in (None, "") else item.get("n", "")
        raw_m = item.get("m", "")
        source_name = str(item.get("name") or "").strip()
        display = source_name or INVENTORY_LABELS.get(feature_type, feature_type or "其他外设")
        if feature_type == "NPU" and raw_n and not str(raw_n).replace(".", "", 1).isdigit():
            display = str(raw_n)
            raw_n = ""
        detail_parts = []
        if raw_n not in (None, ""):
            if feature_type == "RTC":
                detail_parts.append("时钟 " + human_frequency(raw_n))
            elif feature_type == "ADCUnits":
                detail_parts.append(f"转换器单元 {raw_n}")
            elif feature_type in {"ADC", "ADC12", "ADC16", "A/D"} and "channel" in source_name.lower():
                detail_parts.append(f"通道 {raw_n}")
            elif feature_type in {"ADC", "ADC12", "ADC16", "A/D", "DAC", "D/A", "Touch"}:
                detail_parts.append(f"来源数量参数 {raw_n}")
            elif feature_type == "NVIC":
                detail_parts.append(f"{raw_n} 路")
            elif feature_type not in {"Crypto", "Consumption", "XTAL", "IntRC"}:
                detail_parts.append(f"数量 {raw_n}")
        if raw_m not in (None, ""):
            if feature_type in frequency_types:
                detail_parts.append("最高 " + human_frequency(raw_m))
            elif feature_type in bit_width_types and str(raw_m).replace(".", "", 1).isdigit():
                detail_parts.append(f"参数 {raw_m} bit")
            else:
                detail_parts.append(f"参数 {raw_m}")
        detail = " · ".join(detail_parts) or "来源列出，数量未说明"
        record = {"g": item.get("category") or "other", "n": display, "d": detail, "t": feature_type}
        key = (record["g"], record["n"], record["d"])
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def power_records(value):
    """Compact explicit power measurements for the offline detail view."""
    result = []
    seen = set()
    for item in json_value(value):
        if not isinstance(item, dict):
            continue
        mode = text(item.get("mode")) or "other"
        label = text(item.get("label")) or mode
        unit = text(item.get("unit"))
        measurement = decimal(item.get("value"))
        if not unit or measurement is None:
            continue
        record = {"m": mode, "l": label, "v": measurement, "u": unit}
        quality = text(item.get("typical_or_max"))
        if quality and quality != "unknown":
            record["q"] = quality
        conditions = item.get("conditions")
        if isinstance(conditions, dict):
            compact_conditions = {}
            for source_key, target_key in (("clock_hz", "hz"), ("voltage_v", "v"), ("temperature_c", "t"), ("note", "n")):
                value = conditions.get(source_key)
                if value not in (None, ""):
                    compact_conditions[target_key] = value
            if compact_conditions:
                record["c"] = compact_conditions
        source = text(item.get("source_document_id"))
        if source:
            record["s"] = source
        key = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def put(record, key, value):
    if value is not None and value != "" and value != []:
        record[key] = value


def main():
    parser = argparse.ArgumentParser(description="Build the compact offline MCUS web catalog.")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    catalog = args.catalog
    board_data_path = Path(__file__).resolve().parents[1] / "data" / "arduino-boards.json"
    board_data = json.loads(board_data_path.read_text(encoding="utf-8"))
    boards_by_device = board_data.get("boards", {})
    capabilities = {r["device_id"]: r for r in rows(catalog / "device-capabilities.csv")}
    scores = {r["device_id"]: r for r in rows(catalog / "device-scores.csv")}
    parts = defaultdict(list)
    for row in rows(catalog / "orderable-parts.csv"):
        parts[row["device_id"]].append({
            "n": row["part_number"],
            "s": row.get("manufacturer_suffix") or "",
            "p": row.get("package_code") or "",
            "t": row.get("temperature_grade_code") or "",
            "k": row.get("packing_code") or "",
            "u": row.get("source_url") or "",
            "v": row.get("verification_status") or "",
        })
    release_metadata = {}
    release_path = catalog / "release-metadata.csv"
    if release_path.exists():
        release_metadata = {
            row["device_id"]: row
            for row in rows(release_path)
            if row.get("device_id")
        }

    devices = []
    manufacturer_counts = defaultdict(int)
    series = set()
    product_lines = set()
    for variant in rows(catalog / "device-variants.csv"):
        device_id = variant["device_id"]
        cap = capabilities.get(device_id, {})
        score = scores.get(device_id, {})
        release = release_metadata.get(device_id, {})
        record = {
            "id": device_id,
            "m": variant["manufacturer"],
            "f": variant["family"],
            "s": variant["series"],
            "l": variant["product_line"],
            "n": variant["device_name"],
            "a": variant["architecture_class"],
            "pt": variant.get("product_type") or "mcu",
        }
        release_date = first_text(
            variant.get("release_date"),
            cap.get("release_date"),
            release.get("release_date"),
        )
        release_year = number(
            first_text(
                variant.get("release_year"),
                cap.get("release_year"),
                release.get("release_year"),
            )
        )
        release_quarter = first_text(
            variant.get("release_quarter"),
            cap.get("release_quarter"),
            release.get("release_quarter"),
        )
        if release_date and not release_year:
            date_match = re.match(r"^(20\d{2})[-/]", release_date)
            if date_match:
                release_year = int(date_match.group(1))
        if release_quarter:
            release_quarter = release_quarter.upper()
            if release_quarter.isdigit() and 1 <= int(release_quarter) <= 4:
                release_quarter = "Q" + release_quarter
        put(record, "rd", release_date)
        put(record, "ry", release_year)
        put(record, "rq", release_quarter)
        put(record, "rds", first_text(variant.get("release_date_source"), cap.get("release_date_source"), release.get("release_date_source")))
        launch_price = decimal(first_text(variant.get("launch_price"), cap.get("launch_price"), release.get("launch_price")))
        put(record, "lp", launch_price)
        put(record, "lc", first_text(variant.get("launch_price_currency"), cap.get("launch_price_currency"), release.get("launch_price_currency")))
        put(record, "lps", first_text(variant.get("launch_price_status"), cap.get("launch_price_status"), release.get("launch_price_status")))
        put(record, "lpsrc", first_text(variant.get("launch_price_source"), cap.get("launch_price_source"), release.get("launch_price_source")))
        put(record, "v", text(variant.get("manufacturer_variant_code")))
        put(record, "c", text(cap.get("primary_core")) or text(cap.get("core_names")))
        put(record, "cc", number(cap.get("core_count")))
        put(record, "hz", number(cap.get("max_clock_hz")) or number(variant.get("max_clock_hz")))
        put(record, "vmin", decimal(cap.get("operating_voltage_min_v")))
        put(record, "vmax", decimal(cap.get("operating_voltage_max_v")))
        put(record, "fl", number(variant.get("flash_bytes")))
        put(record, "ra", number(variant.get("ram_bytes")))
        put(record, "mem", memory_regions(variant.get("memory_regions_json")))
        put(record, "pkg", text(variant.get("package_types")))
        put(record, "pin", text(variant.get("pin_counts")))
        put(record, "docs", document_records(variant.get("documents_json")))
        put(record, "tim", number(cap.get("timer_count")))
        put(record, "tw", number(cap.get("timer_width_bits")))
        put(record, "adc", number(cap.get("adc_source_quantity")))
        put(record, "adcs", text(cap.get("adc_quantity_semantics")))
        put(record, "adcu", number(cap.get("adc_unit_count")))
        put(record, "adch", number(cap.get("adc_channel_count")))
        put(record, "adr", text(cap.get("adc_resolution_bits")))
        put(record, "adcr", number(cap.get("adc_sample_rate_hz")))
        put(record, "dac", number(cap.get("dac_source_quantity")))
        put(record, "dacr", number(cap.get("dac_sample_rate_hz")))
        put(record, "iospeed", number(cap.get("io_speed_hz")))
        put(record, "fw", number(cap.get("flash_wait_states")))
        put(record, "fb", number(cap.get("flash_bank_count")))
        put(record, "fecc", text(cap.get("flash_ecc_present")))
        put(record, "cache", text(cap.get("cache_present")))
        put(record, "recc", text(cap.get("ram_ecc_present")))
        put(record, "ramarch", text(cap.get("ram_architecture")))
        put(record, "ramex", text(cap.get("ram_exclusive_present")))
        put(record, "gpio", number(cap.get("gpio_count")))
        put(record, "spi", number(cap.get("spi_count")))
        put(record, "i2c", number(cap.get("i2c_count")))
        put(record, "usart", number(cap.get("usart_count")))
        uart_count = number(cap.get("uart_count"))
        if variant["manufacturer"] == "Espressif" and not uart_count:
            uart_count = inventory_quantity(
                cap.get("peripheral_inventory_json"),
                "UART",
                preferred_source_kind="espressif_idf_soc_caps",
            )
        if variant["manufacturer"] == "Espressif" and not uart_count:
            raise ValueError(f"Espressif device has no verified UART count: {device_id}")
        put(record, "uart", uart_count)
        put(record, "can", number(cap.get("can_count")))
        put(record, "cfd", text(cap.get("can_fd_present")))
        put(record, "usb", number(cap.get("usb_count")))
        put(record, "usbd", number(cap.get("usb_device_count")))
        put(record, "usbh", number(cap.get("usb_host_count")))
        put(record, "eth", number(cap.get("ethernet_count")))
        put(record, "dma", number(cap.get("dma_source_quantity")))
        put(record, "pwm", number(cap.get("pwm_source_quantity")))
        put(record, "i2s", number(cap.get("i2s_count")))
        put(record, "lin", number(cap.get("lin_count")))
        put(record, "sercom", number(cap.get("configurable_serial_count")))
        put(record, "otg", number(cap.get("usb_otg_count")))
        put(record, "sdio", number(cap.get("sdio_count")))
        put(record, "wdt", number(cap.get("watchdog_count")))
        put(record, "rtc", text(cap.get("rtc_present")))
        put(record, "comp", number(cap.get("comparator_count")))
        put(record, "opamp", number(cap.get("opamp_count")))
        put(record, "touch", number(cap.get("touch_source_quantity")))
        put(record, "cam", number(cap.get("camera_interface_count")))
        put(record, "display", number(cap.get("display_controller_count")))
        put(record, "extbus", number(cap.get("external_bus_count")))
        put(record, "eb", external_bus_types(cap.get("peripheral_inventory_json")))
        put(record, "exti", number(cap.get("external_interrupt_count")))
        put(record, "tempsens", number(cap.get("temperature_sensor_count")))
        put(record, "crypto", text(cap.get("crypto_accelerator_present")))
        put(record, "pwr", power_records(cap.get("power_measurements_json")))
        put(record, "pws", text(cap.get("power_data_status")))
        put(record, "pi", inventory_records(cap.get("peripheral_inventory_json")))
        put(record, "fpu", text(cap.get("fpu_present")))
        put(record, "mpu", text(cap.get("mpu_present")))
        put(record, "dsp", text(cap.get("dsp_extension_present")))
        put(record, "tz", text(cap.get("trustzone_present")))
        put(record, "acc", feature_labels(cap.get("accelerators_json")))
        put(record, "feat", feature_labels(cap.get("special_features_json")))
        put(record, "pending", feature_labels(cap.get("pending_feature_candidates_json")))
        put(record, "missing", text(cap.get("missing_key_fields")))
        put(record, "idx", number(score.get("selection_index")))
        put(record, "cs", number(score.get("compute_score")))
        put(record, "ms", number(score.get("memory_score")))
        put(record, "ps", number(score.get("peripheral_score")))
        put(record, "acs", number(score.get("accelerator_score")))
        put(record, "cov", number(score.get("score_coverage_percent")))
        put(record, "cm", number(score.get("benchmark_coremark")))
        put(record, "dm", number(score.get("benchmark_dmips")))
        put(record, "src", text(variant.get("source_url")))
        put(record, "verify", text(variant.get("verification_status")))
        put(record, "boards", boards_by_device.get(variant["device_name"], []))
        if device_id in parts:
            record["parts"] = sorted(parts[device_id], key=lambda p: p["n"])
        devices.append(record)
        manufacturer_counts[variant["manufacturer"]] += 1
        series.add((variant["manufacturer"], variant["series"]))
        product_lines.add((variant["manufacturer"], variant["product_line"]))

    esp_devices = [record for record in devices if record["m"] == "Espressif"]
    esp_uart_missing = [record["n"] for record in esp_devices if not record.get("uart")]
    if esp_uart_missing:
        raise ValueError(
            "Espressif UART coverage check failed: " + ", ".join(esp_uart_missing[:20])
        )

    coverage = []
    for row in rows(catalog / "coverage-manifest.csv"):
        coverage.append({
            "m": row["manufacturer"],
            "packs": number(row.get("pack_count")) or 0,
            "families": number(row.get("family_count")) or 0,
            "series": number(row.get("series_count")) or 0,
            "lines": number(row.get("product_line_count")) or 0,
            "devices": number(row.get("device_variant_count")) or 0,
            "parts": number(row.get("orderable_part_count")) or 0,
            "deviceStatus": row.get("device_coverage_status") or "",
            "partStatus": row.get("orderable_coverage_status") or "",
        })

    score_values = [number(row.get("score_coverage_percent")) for row in scores.values()]
    score_values = [value for value in score_values if value is not None]
    fpu_verified = sum(
        row.get("fpu_present") in {"yes", "no"} for row in capabilities.values()
    )
    payload = {
        "meta": {
            "version": "1.4x",
            "name": "MCUS",
            "author": "new.bmp",
            "repository": "https://github.com/new-bmp/MCUS",
            "snapshot": datetime.now(timezone.utc).date().isoformat(),
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manufacturers": len(manufacturer_counts),
            "series": len(series),
            "productLines": sum(item["lines"] for item in coverage),
            "devices": len(devices),
            "parts": sum(len(value) for value in parts.values()),
            "averageCoverage": round(sum(score_values) / len(score_values), 1) if score_values else 0,
            "devicesAt90": sum(value >= 90 for value in score_values),
            "fpuVerified": fpu_verified,
            "fpuCoverage": round(100 * fpu_verified / len(capabilities), 1) if capabilities else 0,
        },
        "coverage": coverage,
        "boards": {
            "label": "Arduino",
            "kind": "development_board_ecosystem",
            "source": board_data.get("source", ""),
            "sourceUrls": board_data.get("sourceUrls", []),
        },
        "devices": devices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    args.output.write_text("window.MCU_CATALOG=" + encoded + ";\n", encoding="utf-8")
    print(json.dumps(payload["meta"], ensure_ascii=False))
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
