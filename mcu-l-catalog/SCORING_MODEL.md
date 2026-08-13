# MCUS 参数与评分模型

## 两类分数必须分开

1. **实测/公布基准分数**：CoreMark、DMIPS、ULPMark 等。只有存在可引用的厂商或基准组织结果时才填写；不根据核心和主频推算。
2. **MCUS 选型指数**：为了排序候选器件而计算的 0–100 分启发式指标，不是 benchmark，也不代表同频 IPC、实时性或最终应用性能。

界面必须同时显示评分模型版本和数据覆盖率。例如 `选型指数 46 · 覆盖率 85%` 表示仍有 15% 权重因专用加速器资料缺失而没有参与计算，而不是按零分处罚。

## 当前 V1 维度

- 计算能力 35%：最大核心频率 60%，核心代际能力表 40%。
- 存储能力 25%：Flash 55%，RAM 45%，使用对数区间归一化。
- 外设能力 25%：通用定时器、ADC/DAC、串行接口、CAN、USB、Ethernet 和 GPIO。
- 加速器与专用能力 15%：NPU、数学/控制加速器、图形/编解码加速器、密码加速器等。

缺少某一维度时，对剩余已知维度重新归一化，同时降低 `score_coverage_percent`。评分不能用来掩盖缺失字段。

## 参数语义

- `timer_count`：排除 watchdog 和 RTC 后，可由来源明确计数的通用/高级定时器数量。
- `adc_source_quantity`：保留 CMSIS 特性中的原始数量；`adc_quantity_semantics` 标记它代表转换器、通道还是来源未说明。
- `primary_core/core_names/core_count/max_clock_hz`：直接来自处理器描述。
- `accelerators_json`：只包含来源已记录或人工核验通过的厂商专用能力。
- `pending_feature_candidates_json`：例如尚待逐器件官方资料确认的 Chrom-ART 候选项；不能作为已确认特性展示。


