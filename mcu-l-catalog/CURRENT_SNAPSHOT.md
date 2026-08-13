# MCUS 当前目录快照

## 2026-08-12 官方逐器件数据库更新

- 10 家厂商、236 个系列、820 条产品线、7,187 个器件变体。
- 3,379 个官方订货号，315 条来源记录。
- Infineon ModusToolbox device-db 4.39.0.10988：1,514 个现有型号精确匹配，新增 140 个官方 MCU 记录和 1,487 个订货号。
- Espressif ESP-IDF：接入 13 个 SoC 目标、增强 291 条产品记录；319 条 Espressif 记录均有 UART 数据，287 条已有精确通用定时器数量。
- Microchip DSU 保留为调试/系统单元，不再误标为加密加速器。
- 全库校验为 0 错误、0 警告；9 条历史上游包获取错误仍保留在审计文件中，未算作成功覆盖。

快照日期：2026-08-11

## 已落盘规模

| 指标 | 数量 |
|---|---:|
| 厂商 | 9 |
| 家族 | 146 |
| 系列 | 197 |
| 产品线 | 696 |
| 器件变体 | 6,728 |
| 已核验完整订货号 | 303 |
| 来源记录 | 261 |
| 结构化能力记录 | 6,728 |
| 结构化评分记录 | 6,728 |

当前合并包位于 `data/combined/`，校验结果为 `ok`，没有重复主键、断裂父级、负数规格或无来源订货号。

能力表已经增加核心型号、核心数、最大核心频率、TIM 数量、定时器位宽、ADC 来源数量及语义、GPIO、SPI、I²C、USART/UART、CAN、USB、DMA、加速器和厂商特性字段。当前 2,780 个器件具有可直接提取的 TIM 数量，723 个器件具有来源中可识别的加速器或硬件计算特性。

## 当前各厂商器件层覆盖

| 厂商 | 系列 | 产品线 | 器件变体 | 完整订货号 |
|---|---:|---:|---:|---:|
| STMicroelectronics | 27 | 231 | 2,778 | 303 |
| Infineon | 56 | 139 | 1,594 | 0 |
| Nuvoton | 11 | 111 | 1,058 | 0 |
| Microchip | 43 | 43 | 450 | 0 |
| GigaDevice | 12 | 33 | 388 | 0 |
| MindMotion | 18 | 44 | 157 | 0 |
| Geehy | 9 | 28 | 153 | 0 |
| Puya | 6 | 41 | 87 | 0 |
| Texas Instruments | 15 | 26 | 63 | 0 |

这里的“器件层覆盖”只表示已成功索引当前可用 CMSIS Device Family Pack，不等于厂商全目录已经完成。例如 Microchip 当前数据主要覆盖可由 CMSIS-Pack 描述的 SAM/PIC32 等系列，不能代表 PIC8、PIC16、PIC18、AVR 和 dsPIC 已全部收录。

## STM32F103 变体与订货号

`STM32F103` 已收录 29 个实际器件变体和 133 个由 ST 官方产品页确认的完整订货号。

器件变体包括：

- `C4/C6/C8/CB`
- `R4/R6/R8/RB/RC/RD/RE/RF/RG`
- `T4/T6/T8/TB`
- `V8/VB/VC/VD/VE/VF/VG`
- `ZC/ZD/ZE/ZF/ZG`

例如 `STM32F103C8` 在目录中拆分为：

```text
产品线：STM32F103
器件变体：STM32F103C8
厂商变体码：C8
Flash：65536 bytes
RAM：20480 bytes
引脚数：48

官方完整订货号：
STM32F103C8T6
STM32F103C8T6TR
STM32F103C8T7
STM32F103C8T7TR
```

订货号记录继续拆出：

- `T`：封装代码字段；
- `6` / `7`：温度等级代码字段；
- `TR`：卷带包装代码；
- 具体封装名称和温度范围暂不靠代码表猜测，等待对应器件数据手册再次核验。

以 `STM32F103C8` 为例，当前来源还给出：

```text
Core：Arm Cortex-M3 · 1 core
最大核心频率：72 MHz
TIM：4 · 16-bit
ADC 来源数量：10 · 源参数 12 · 数量语义未说明
GPIO：36
SPI / I2C / USART：2 / 2 / 3
CAN / USB Device：1 / 1
MCUS 选型指数：42 / 100
评分数据覆盖率：85%
CoreMark / DMIPS：未知（未导入可引用基准）
```

MCUS 选型指数用于候选排序，不是实测性能分数。CoreMark、DMIPS、ULPMark 等必须有独立来源，不能由核心和主频推算。

厂商专有能力使用独立特性字段，保留原厂名称，例如 `Chrom-ART (DMA2D)`、`Neural-ART`、`CORDIC`、`FMAC`、`TMU`、密码加速器和安全单元。只有已在器件资料中确认的特性才进入已核验字段；仅按系列推测的项目进入待核验候选区。

## 已知导入缺口

当前有 9 个上游包未成功读取：Geehy 7 个、Infineon 1 个、Puya 1 个。原因已写入 `data/combined/import-errors.csv`，包括厂商服务器返回 404/405、连接超时，以及地址返回非有效 XML。它们没有被计入“已覆盖”。

## 下一步扩充顺序

1. NXP、Renesas、Silicon Labs、Nordic、Espressif 等厂商的器件层适配。
2. Microchip 的 PIC8/PIC16/PIC18、AVR、dsPIC 独立官方目录适配。
3. TI 的 MSP430、C2000、TM4C、Hercules 与完整订货号适配。
4. STM32 F0/F2/F3/F4/F7/G/H/L/U/W 等其余系列的官方订货号抓取。
5. 为每家厂商建立产品页/选型表订货号适配器，逐步把 `orderable_part_count=0` 清零。
6. 继续保留停产历史型号、厂商并购前品牌和上游失效链接的回填队列。

