# MCUS

面向工程师的离线 MCU 选型工具，按厂商、系列大类、产品线、器件变体和完整订货号组织目录，并支持参数筛选、四器件横向比较与突出项标记。

## 当前版本

- Android：`1.5`
- Web：可直接上传 `mcu-l-web/staticfiles/`
- 包名：`com.newbmp.mcus`
- 数据快照：13,791 个器件、20,853 个订货号
- 平均数据覆盖率：97.9%；覆盖率 ≥ 90%：12,967 个器件；FPU 已核验：13,509 个
- 支持 STM32、ESP、CH32、HPM、AT32、MM32、国民技术、瑞萨、Microchip/Atmel、TI DSP 等系列
- 详情页提供 Flash / RAM 架构、ECC、Bank、RWW、Swap、等待周期、擦写寿命、保护及外部 XIP 等工程信息
- 内置固件升级适配评估，辅助判断 A/B 升级、外部 Flash 暂存和单区 Bootloader 条件

APK 与 Web 静态包请从 [GitHub Releases](https://github.com/new-bmp/MCUS/releases/latest) 下载。

## 目录结构

- `mcu-l-android/`：Android 应用源码、离线资源与构建脚本
- `mcu-l-catalog/`：数据目录、厂商采集脚本、来源与校验报告
- `mcu-l-web/staticfiles/`：可直接上传部署的纯静态 Web 成品
- `release/`：当前可安装 APK

## 作者

`new.bmp`

项目主页：[github.com/new-bmp/MCUS](https://github.com/new-bmp/MCUS)

## 数据说明

目录中的器件能力优先采用厂商产品选择器、CMSIS-Pack、官方设备数据库和官方工具链元数据。缺失字段保留为未知，不根据同系列型号臆造；ADC 转换器单元与 ADC 通道分别记录，不把 ADC 引脚数作为选型指标。

本项目遵循仓库中的 Apache License 2.0。各 MCU 厂商名称、型号和标识属于其各自权利人。
