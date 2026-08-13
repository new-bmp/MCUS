# MCUS

面向工程师的离线 MCU 选型工具，按厂商、系列大类、产品线、器件变体和完整订货号组织目录，并支持参数筛选、四器件横向比较与突出项标记。

## 当前版本

- Android：`0.6.0`
- 包名：`com.newbmp.mcus`
- 数据快照：7,504 个器件、4,427 个官方订货号
- 覆盖率 ≥ 90%：7,113 个器件；FPU 已核验：7,364 个
- 支持 Arduino / Atmel AVR、SAM，以及 Espressif ESP 系列

APK：[`release/MCUS-0.6.0-debug.apk`](release/MCUS-0.6.0-debug.apk)

## 目录结构

- `mcu-l-android/`：Android 应用源码、离线资源与构建脚本
- `mcu-l-catalog/`：数据目录、厂商采集脚本、来源与校验报告
- `release/`：当前可安装 APK

## 作者

`new.bmp`

项目主页：[github.com/new-bmp/MCUS](https://github.com/new-bmp/MCUS)

## 数据说明

目录中的器件能力优先采用厂商产品选择器、CMSIS-Pack、官方设备数据库和官方工具链元数据。缺失字段保留为未知，不根据同系列型号臆造；ADC 转换器单元、ADC 通道和 ADC 外部输入引脚分别记录。

本项目遵循仓库中的 Apache License 2.0。各 MCU 厂商名称、型号和标识属于其各自权利人。
