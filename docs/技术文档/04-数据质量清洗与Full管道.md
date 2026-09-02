# 04 数据质量清洗与 Full 管道

## 1. 本章范围

本章说明视频平滑之后，系统如何把统一时间轴、数值流、视频质量、手部可见性和 VLM 行为语义合并为最终的坏帧、待复核帧和有效帧状态，并在 Full 模式下生成可复核、可追溯的输出数据集。界面可以用红、黄、绿显示三种状态，但技术文档统一使用质量语义名称。

当前清洗实现的主入口是 app/curation_pipeline.py。它不是一个单独的模型，而是一个作业编排器：每个阶段只负责一种证据，最后再按统一的帧状态规则合并。

当前版本号为：

    Paper Curation schema: alice/paper-curation/v1
    Curation pipeline version: 19
    Full run schema: alice/full-run/v1
    Full export pipeline version: 8

本章只描述现有代码已经执行的功能。S4、S5 尚未实现；Nexus/OpenXR 的 C3 仅在严格的外部标定契约完整时启用。

## 2. 两种运行入口

系统共用同一套清洗核心，但提供两个入口。

### 2.1 数据质量清洗

数据质量清洗不主动生成平滑视频，也不导出训练数据。它按用户选定的 RGB 流执行：

    媒体与格式门禁
    → T0 统一时间轴
    → P1、S1、S2、S3
    → S4、S5 明确跳过
    → C3
    → 排除坏帧后的候选片段 VLM
    → C1、C2
    → 写入 .alicePD 清洗报告

报告和 S1 修复补丁先暂存，源数据文件在运行期间保持只读。

### 2.2 Full Pipeline

Full 在相同清洗核心外增加视频平滑、可选 Action、时间轴冻结和导出：

    媒体与格式门禁
    → T0 验证所选源视频及手部数值流对齐
    → EgoDex：MediaPipe 手部二维投影归正（统一调整率 65%）
       Nexus / OpenXR / 其他模式：跳过该步骤
    → 视频平滑
       ↘ 可选 Action 生成与 S2 校验
    → 冻结 Full analysis timeline
    → P1、S1、S2、S3、S4、S5、C3 初筛
    → 对所有未判定为坏帧的候选片段运行或复用 VLM
    → C1、C2
    → Full 导出
    → 发布该 Episode 的 latest Full run

视频平滑与用户明确启用的 Action 生成可在同一个 Episode 内并行执行。一个 Full 作业中的多个 Episode 仍按顺序处理；多个独立作业可由后台工作线程并行运行。

## 3. 输入门禁和作业预留

### 3.1 媒体门禁

清洗和 Full 只能读取满足以下条件的 RGB 媒体：

- modality 为 rgb；
- analysis_eligible 为 true；
- 媒体类型是可解码的视频或图像序列；
- Full 还要求该媒体没有被明确标记为 vlm_eligible=false 或 smoothing_eligible=false。

原始 Depth、IR、触觉文件和未知二进制流不能作为清洗参考视频。旧 manifest 如果尚无 modality 字段，只对已经识别为 video 或 images 的非深度媒体保留兼容性回退。

多视频 Episode 必须锁定具体 media_file_id。T0、平滑、C3、VLM、报告读取和 Full run 发布都沿用同一个媒体身份，不能把头部视频的映射用于腕部视频。

### 3.2 预检不是最终判定

curation_preflight() 只读取元数据和能力状态，返回 ready、pending、skipped 或 warning。它用于告诉界面某个阶段是否有先决条件，不产生坏帧、待复核帧或有效帧结论，也不表示该阶段运行后一定通过。

特别需要区分：

- 阶段 status=completed 表示阶段成功执行，不表示没有异常帧；
- status=warning 可能表示检测到复核项，也可能表示某项能力不可用；
- status=skipped 表示没有执行，不能当作通过；
- C3 的整手投影不可用时，视频画质检查仍然继续。

### 3.3 Episode 预留

同一数据集、同一 Episode 同一时间只能被一个清洗或 Full 作业占用。任务提交时先预留全部目标 Episode；任务完成、失败或取消后释放预留，避免两个后台任务同时写同一份报告。

后台工作线程数由 ALICE_FULL_WORKERS 控制，默认 2。一个作业内部按 Episode 顺序推进，因此跨 Episode 的结果顺序与请求中的 episode_ids 一致。

## 4. 数据集模式严格分流

所有空间和关节处理都读取 app/dataset_modes.py 生成的显式模式契约，不通过“看到 20、21 或 26 个点”直接猜测坐标空间。

| 模式 | 清洗数值流 | Full 自动手部归正 | 压力/触觉 | C3 整手投影 | 外参边界 |
|---|---|---|---|---|---|
| EgoDex | EgoDex 完整骨架按自身语义读取；C3 内部重建 MANO21 | 启用 MediaPipe，统一调整率 65% | 不运行 Nexus P1/触觉 S1 | 当前可运行 | 使用 Episode HDF5 内置 camera pose 和 intrinsic，不需要 Nexus 外参 |
| Nexus | Nexus/DexWeaveG1 20 点经专用适配器形成 MANO21 数值列，可进入 S1/S3 | 完全禁用，不加载 MediaPipe、不生成归正产物 | 运行双侧压力 P1 和触觉孤立突变 S1 | 标定完整时运行 | 只接受 Mocap Tracking → RGB 的专用 4×4 外参，不能借用 EgoDex 投影器或 OAK RGB/Depth 预设 |
| OpenXR | OpenXR 26 点经专用适配器形成 MANO21 数值列，可进入 S1/S3 | 禁用 | 不运行 Nexus P1 | 标定完整时运行 | 只接受 OpenXR baseSpace → RGB 的专用 4×4 外参 |
| LeRobot / Alice Full | 按已声明字段读取 | 禁用 | 只在显式 Nexus 模式启用 | 当前无通用投影后端 | 不从数组形状推断相机关系 |
| unknown / conflict | 只读取能够确认的通用数值流 | 禁用 | 不启用 Nexus 专用逻辑 | 禁用 | 进入保守模式 |

Nexus 和 OpenXR 的 MANO21 适配解决的是点序和数值列语义，不等于已经获得 RGB 相机坐标。规范化数值可以直接参与时间序列检查；C3 则必须再通过各自模式的外部标定验证，绝不会送入 EgoDex 的内置相机投影器。

如果同一传感器同时存在 recorder 同步流和 raw 流，清洗优先使用同步流。raw 流仍保留在清单中用于审计，但不会与同步流重复拼接，避免把同一只手以两种采样率重复计入 S1/S3。

## 5. Full 的真实作业生命周期

### 5.1 创建 run

Full 使用作业 ID 作为 run_id，并创建不可覆盖的运行目录：

    <dataset>/.alicePD/full-runs/<run_id>/
      run.alice
      episodes/<episode_id>/

run.alice 保存原始请求、Episode 顺序、每个 Episode 的状态、时间轴、产物索引、摘要和错误。相同 run_id 已存在时不会覆盖。

训练数据导出到：

    <dataset>/output/<run_id>/

运行产物和训练输出分开保存：前者用于审计和回看，后者用于交付。

### 5.2 EgoDex Full 自动 MediaPipe 手部归正

Full 在显式 EgoDex 模式下默认先执行手部二维投影归正。固定配置为：

- 检测后端：MediaPipe Hands，整幅 RGB 图像输入，同时检测最多两只手；
- 模型采样率：15 FPS；
- 调整策略：uniform；
- 调整率：0.65，即 65%；
- 0 点来源：EgoDex/MANO 估算腕根；MediaPipe landmark 0 不直接替换腕根；
- 骨长和刚性掌部约束保持启用，MediaPipe 只提供二维观测，不允许无约束覆盖三维关节。

该步骤在源视频 T0 对齐验证通过后执行，并早于视频平滑、清洗和导出。归正后的 HDF5 在本次 Full run 内立即成为有效数值源；如果归正引入新的 S1 突变，归正模块可以插入受约束的中间帧，并生成同步重定时视频。随后的视频平滑、C3、Action、VLM 和导出都沿用该运行级时间轴。

产物保存在不可覆盖的运行目录：

    <dataset>/.alicePD/full-runs/<run_id>/episodes/<episode_id>/projection/
      <episode_id>.projection.alice
      <episode_id>.projection.hdf5
      <episode_id>.projection.mp4    # 仅发生重定时时生成

原始 HDF5 和视频始终只读。Full 自动归正属于 `activation_scope=full_run`，不会伪装成用户已经审核并应用的全局修改，也不会覆盖独立“手部二维投影归正”功能中的待审核结果。运行记录保存后端、65% 调整率、采样率、腕根策略、应用帧数、拒绝帧数、重投影残差和重定时信息。

模式边界是硬约束：只有 `projection_correction_backend=egodex_mano_prior_v1` 才启用该步骤。Nexus Full 不加载 MediaPipe，不调用 EgoDex 归正器，不读取已有 EgoDex 归正快照，也不生成 projection 目录；Nexus 仍只走自身的 20 点适配、触觉处理和外部标定路径。OpenXR、LeRobot、Alice Full、unknown/conflict 同样不会自动进入 EgoDex 归正器。

如果 MediaPipe 加载、二维检测、三维约束求解或运行级产物校验失败，对应 EgoDex Episode 明确失败，不会静默回退到未归正轨迹后继续导出。

### 5.3 构建 S3 cohort

当一次任务包含多个 Episode 时，管道在逐 Episode 处理前尝试构建跨 EP 的 S3 分位参考。只有以下条件同时满足才会进入同一 cohort：

- 至少两个 Episode；
- embodiment_id 已明确；
- Joint/Action 总维度一致；
- dimension_names 顺序一致；
- 数值流语义完整；
- 构建参考前后源文件签名未变化。

S3 对每个目标 Episode 使用 leave-one-episode-out 参考：只拼接同 cohort 的其他 Episode，不把目标 Episode 自身纳入 q01/q99。这样异常 Episode 不能用自身极值扩大自己的判定区间。每个参考 Episode 最多抽取 20,000 行，以限制内存；报告记录实际使用的 reference_episode_ids 和这些参考源的签名。不能可靠分组的 Episode 回退为 episode_limited，S3 只生成待复核帧，不自动判为坏帧。

### 5.4 非 Full 清洗对已应用结果的复用

独立“数据质量清洗”不会自动运行 MediaPipe 或 AlicePose。它可以复用此前由用户审核并应用的 EgoDex 投影归正快照。未应用的待审核结果不会进入清洗数值流。

清洗数值流读取已应用快照时仍保留原始 relative_path 作为源身份，并同时锁定原文件和派生文件，避免派生结果脱离原始数据版本。

### 5.5 初始 T0 和视频平滑

管道先对用户选中的原始媒体运行 T0 验证，EgoDex 自动归正才能据此取得严格的视频帧到手部数据行映射。归正完成后生成平滑 analysis_media。投影归正的受约束插帧和高帧率 EIS 都可能改变逻辑帧数或 FPS，因此清洗核心会针对最终 analysis_media 再执行一次 T0 retime，把各传感器重新映射到 Full 分析帧空间。

可选 Action 文件目前仍按源帧率生成。Full 不另写一份 30 Hz Action 文件，只把 Action 的 S2 invalid_mask 映射到 analysis_media 的 source_frame_positions。

### 5.6 冻结时间轴

平滑完成后，write_full_timeline_lock() 写入 timeline.alice。冻结内容包括：

- 输入视频和分析视频的媒体身份；
- 文件大小、修改时间和摘要身份；
- 逻辑帧数、FPS、宽高和时长；
- source_frame_positions 的数量和哈希；
- 平滑产物路径；
- 投影归正是否启用、检测后端、调整率、激活范围、application_id（仅人工应用结果）和是否重定时。

上述核心字段序列化后计算 SHA-256，得到 timeline_id。后续 smoothing、curation 和 behavior 产物都写入相同的 full_run_id 和 timeline_id。

回看 Full run 时，系统至少要求平滑 sidecar、平滑视频、最终清洗报告存在，并校验它们的 run_id 与 timeline_id。任一必需产物缺失或时间轴不一致，整个 Full bundle 都不会被当作可读结果。

## 6. 帧质量状态模型

管道内部使用两个逐帧布尔掩码：

- invalid：坏帧，不能进入最终普通训练片段；
- review：待复核帧，需要人工或后续检查确认；
- 两者都为 false：有效帧。

合并优先级固定为：

    bad frame > review frame > valid frame

任一阶段把某帧写入 invalid 后，后续 review 不能覆盖它。所有阶段结束后执行 review &= ~invalid，再把连续同状态帧合并为 segments。

默认会把相隔小于 0.3 秒的相邻坏帧区间连接为一个坏帧区间，也会对待复核区间执行同样的连接；连接待复核区间时仍然排除已经判为坏帧的帧。判断条件是严格小于阈值，不包含刚好等于 0.3 秒的间隔。

每个 finding 保存 stage、severity、帧范围、时间范围、reason 和 confidence。最终 segments 的原因是所有与该区间相交 finding 原因的并集。

### 6.1 统一质量证据视图

清洗报告在保留原有字段的同时，新增 `alice/quality-evidence/v1` 视图。它把每个阶段的结果统一为 `checks`，每个 check 包含：

- `measurements`：阶段原始测量值和统计量；
- `intervals`：带帧号、时间和原因的坏帧/待复核区间；
- `tags`：阶段和质量类型；
- `verdict`：`pass`、`review`、`fail` 或 `skipped`。

顶层 `aggregate` 使用最终合并后的 segments 计算有效、待复核和坏帧数量。`skipped` 不会被转换为通过。`pipeline` 保存清洗版本、`full_run_id` 和 `timeline_id`；`provenance` 保存配置指纹、源文件签名和产物引用。这样下游查询可以直接使用统一结构，不需要解析每个阶段的私有 metrics。

完整 Full Episode 导出时，如果清洗报告包含该视图，Episode 目录会额外写入 `quality_evidence.json`，并由 `subtasks.json` 的 `quality_evidence.file` 引用。原有 `.alice` 报告和 `.invalid.bin` 不变。

## 7. 各阶段的当前实现

### 7.1 T0：统一时间轴

T0 调用 scan_episode_sensor_alignment()、retime_sensor_alignment() 和 validate_episode_time_sync()，把所选视频帧映射到各 Joint、Action、压力和其他动态流。

映射越界、源 partial 行或无效行会进入 bundle.valid_mask；后续 S1 将这些帧作为 source_invalid 坏帧。T0 本身不是质量分数，它是所有后续数值判断的帧空间门禁。

### 7.2 P1：Nexus 压力完整性

P1 只在 nexus_multimodal 模式运行，并要求左右两侧同步压力流。

以下情况判为坏帧：

- 缺少左侧或右侧压力文件；
- 压力文件不可读；
- 空数据行；
- partial 行；
- 无效 source_index；
- 非有限时间戳；
- T0 无法把压力行映射到视频帧。

某一侧文件完全缺失时，该侧的整个 Episode 掩码为红。左右任一侧为空，合并结果即为红。

压力数值等于 0 始终视为有效。P1 判断“是否存在有效记录”，不把“没有接触”误判为“没有传感器数据”。

### 7.3 S1：突变、加速度和 Jerk

通用 S1 对已经映射到视频时间轴的 Joint 和 Action 数值列执行：

    缺值插值
    → 3/5 点中值滤波
    → Savitzky-Golay 平滑
    → 原始值与平滑值残差
    → 二阶差分加速度
    → 三阶差分 Jerk
    → median + sigma × robust scale 阈值

只有残差异常并同时命中加速度或 Jerk 条件的维度才产生坏帧。默认 sudden_change_sigma 为 6.0，接口允许 3.0–12.0。

rot6d 另行检查 6D 基向量有效性和相邻相对旋转突变，不把旋转列当普通欧氏标量直接处理。

Nexus 模式还检查左右触觉的孤立突变。检测只使用 pressure_sum 和 pressure_max 作为主要证据，最多接受 2 帧的短脉冲候选，并要求突变后回到局部基线。持续接触阶跃和数值 0 不判错。

T0 映射越界、partial 或无效的数值源帧也在 S1 判为坏帧。

已应用投影归正时，S1 同时读取不可变原始轨迹做保护比较。只在归正结果中新增、而原始轨迹未命中的突变不会直接判为坏帧，而是保留为待复核帧；这些帧也禁止进入自动修复，避免生成错误的源级补丁。若同一批突变点跨越持续运动区间，系统把它们从硬性异常降为待复核，防止把真实快速动作拆成大量孤立坏帧。

#### S1 自动修复

repair_s1_spikes 默认开启，最多修复 5 帧的孤立尖峰：

- 普通数值列使用两侧可靠锚点的有界三次插值；
- rot6d 使用两侧旋转的 SLERP；
- 夹爪列和 rot6d 列不进入普通三次插值；
- 修复后重新运行 S1，仍在邻域内产生异常的修复会被撤销；
- 源文件不在清洗运行中直接改写，修复以 s1-repair.alice 补丁保存并在 C2/导出读取时应用。

Nexus20 和 OpenXR26 适配器会重排、丢弃或合成节点，规范化列无法安全映射回单个源单元格，因此当前不会对这些适配结果生成源级 S1 写回补丁。

### 7.4 S2：State-Action 对齐与导出一致性

S2 有两条路径。

第一条路径校验已生成的 Action：

- 源轨迹身份；
- 预测目标帧；
- Action 文件索引；
- 输出帧数。

不一致帧逐帧判为坏帧。Full 选择生成 Action 时使用此路径，并可把 invalid_mask 重映射到 EIS 目标时间轴。

第二条路径用于数据集中已经同时存在 Joint/State 和 Action 的情况。系统要求两侧都提供可唯一匹配的 dimension_names，按规范化语义名称建立维度对应关系，因此允许 State/Action 列顺序不同，但不会再按“前 N 列”猜测。系统先按 absolute、delta 或 velocity 解释 Action，再基于运动差分搜索允许范围内的最佳相关 lag，最后在局部滑窗中计算逐帧方向一致率。

默认 directional_agreement_threshold 为 0.65：

- 局部一致率大于等于 0.65：该窗口通过；
- 局部一致率低于 0.65、但未达到硬性失败阈值：对应帧判为待复核；
- 局部一致率达到硬性失败条件：只把对应区间判为坏帧；
- Action 表示类型未知、dimension_names 不能唯一匹配或运动变化不足：S2 跳过，不自行猜测。

全局 directional_agreement 仍写入报告作为摘要，但最终质量掩码来自局部窗口，不再因为一个局部错位把整个 Episode 全部判废。

### 7.5 S3：分位极值

S3 对 Joint 和 Action 拼接矩阵计算 q01、q99，并使用：

    lower = q01 - alpha × (q99 - q01)
    upper = q99 + alpha × (q99 - q01)

默认 alpha 为 0.1。已明确的 gripper 列不参与极值判定。

只有 reference_scope=cohort 且维度语义完整时，超界帧才判为坏帧。以下情况只生成待复核帧：

- 只有当前 Episode 自身统计；
- embodiment 未知；
- dimension_names 不完整；
- 维度结构无法与其他 Episode 安全比较。

该限制用于避免把一个 Episode 自己的正常端点，或不同机器人/不同点序的数值范围，误当成跨数据集异常。

### 7.6 S4：FK 一致性

当前固定为 skipped。代码没有执行 URDF、Pinocchio 或其他 FK 一致性计算，即使预检发现相关文件也不会宣称可运行。

### 7.7 S5：基座与方向统一

当前固定为 skipped。系统只允许其他模块记录坐标修正建议，不在清洗管道中自动旋转、平移或覆盖 Joint/Action 坐标。

S5 跳过不代表坐标已经统一，只表示本版本没有执行该检查。

### 7.8 C3：视频质量

C3 视频质量部分适用于所有通过 RGB 门禁的模式。请求默认值和允许下限均为 15 FPS；源视频低于 15 FPS 时逐帧检查，源视频高于 15 FPS 时选择整数步长并保证实际采样率不低于 15 FPS。报告同时记录 requested_sample_fps、effective_sample_fps 和 sample_step_frames。每个抽样结果扩展到该样本附近的源帧区间。

当前检测包括：

- 解码失败或损坏帧；
- 平均灰度低于 black_level_threshold 的黑帧；
- Laplacian 方差低于 blur_laplacian_threshold 的持续模糊；
- 帧差低于 static_difference_threshold 且持续达到 static_duration_seconds 的长静止。

模糊必须在相邻三个抽样点中至少出现两个才成立。若 Action 在对应邻域存在明显运动，该样本被 protected，不因模糊或静止直接判为坏帧。

长静止的颜色取决于动作证据：

- 有 Action 且对应区间没有关键运动：判为坏帧；
- 有 Action 且存在关键运动：保护，不因静止判废；
- 没有 Action：判为待复核帧，要求人工复核。

### 7.9 C3：整手可见性

EgoDex 从不可变的 Episode HDF5 读取：

- transforms/camera；
- 相机 intrinsic；
- EgoDex 完整手部骨架或已重定向的 MANO21 快照；
- T0 对齐位置；
- Full EIS 每帧像素变换。

EgoDex 先用自身 camera pose 把世界点转换到相机坐标，再用内参投影。它不读取 Nexus/OpenXR 外参。

Nexus 和 OpenXR 使用独立的外部标定投影器。启用前必须同时满足：

- camera_calibration.source_extrinsics_applied=true；
- hand_projection.applied=true；
- Nexus 的 source_space=mocap_tracking 且提供 T_rgb__mocap_tracking；
- OpenXR 的 source_space=openxr_base_space 且提供 T_rgb__openxr_base；
- transform_direction=source_to_rgb_camera，平移和关节坐标单位明确为米；
- 提供有效的 RGB 3×3 内参；
- 通过 target_media_file_id 或 target_stream_name 绑定当前分析视频；
- T0 已把对应模式的 MANO21 数值流对齐到当前视频时间轴，且手侧明确。

OAK-D Pro W9 的左右 Depth 到 RGB ±3.75 cm 预设只允许 RGB/Depth 配准，不能充当 Mocap Tracking/OpenXR baseSpace 到 RGB 的手部外参。任何字段、方向、单位或媒体绑定不完整时，整手投影安全跳过并保持空掩码，视频质量 C3 仍继续执行。

C3 不会因为没有 Action 就默认要求左右手。required_sides 只从已生成 Action、明确的流 side、Episode 元数据、文件名提示或 HDF5 小型属性中获取；手侧不明确时跳过整手可见性，不制造全视频坏帧。

对每一只明确要求的手独立判定：

- 21 个 MANO 点全部可见：该手有效；
- 有点可见但不是全部可见，且不可见比例不超过 60%：待复核；
- 没有任何点可见，或超过 60% 点不可见：坏帧。

多手任务中，任一要求手达到坏帧条件，该帧即为坏帧；只有所有要求手的全部点均可见，该帧才是有效帧。某只手完全可见不能稀释另一只缺失手的判定。

Nexus/OpenXR 的外部投影与 EgoDex 使用同一套 MANO21 三档可见性分类，但关节适配器、坐标空间和标定入口严格分开。

## 8. VLM、C1 和 C2

### 8.1 VLM 输入范围

P1、S1、S2、S3、S4、S5 和 C3 完成后，管道生成 pre_vlm_segments。

VLM 输入规则是先排除坏帧，而不是只使用已经确认的有效帧：

- 坏帧片段不送入 Qwen；
- 待复核片段和有效片段都送入 Qwen；
- 相邻的候选片段合并为连续 allowed_ranges。

如果排除坏帧后不存在候选片段，VLM、C1 自动跳过。若存在候选片段但没有可复用标注，并且 Qwen-VLM 未配置，该 Episode 在 VLM 阶段失败；已经写出的 pre_vlm 清洗报告仍可用于诊断，但不能作为完整 post_vlm 结果发布。

### 8.2 VLM 复用条件

已有行为标注只有同时满足以下条件才复用：

- force_vlm=false；
- allowed_ranges 与本次排除坏帧后的候选区间完全一致；
- analysis video 的 file_id 一致；
- frame_count 一致；
- 媒体 fingerprint 一致。

Full 复用时会把行为标注复制到当前 run 目录，并重新写入本次 run_id 和 timeline_id，不直接引用另一个 run 的可变文件。

### 8.3 C1：指令一致性

C1 比较 VLM task_label 与 Episode 的 name、task、instruction 和 description：

- task_label 为 other、unknown 或空值；
- 两侧有效词集合完全不相交；
- VLM confidence 低于 0.35；

任一条件成立时，把所有初筛候选帧判为待复核。C1 是轻量词级检查，不进行完整自然语言蕴含推理。

### 8.4 C2：视频-State 一致性

C2 从 Joint 和 Action 计算归一化运动证据，只检查 VLM 中预期存在主动操作的阶段。idle、observe、reach、withdraw、unknown 和 precheck_invalid 不进入主动阶段检查。

一个阶段至少要有 0.4 秒的初筛候选帧才会检查。若该阶段中超过阈值的运动帧比例低于 0.08，整个阶段判为待复核，原因记为“VLM 阶段缺少同步 State/Action 运动证据”。

没有可对齐 Joint/Action 时，C2 跳过，不把缺少数值证据自动解释为通过或失败。

### 8.5 当前待复核状态的后处理语义

finalize_episode_curation() 当前使用 resolve_post_vlm_review() 合并待复核状态：

- 始终保留 S1、S2、S3、C3 等初筛阶段已经产生的待复核帧；
- C1 和 C2 只能新增待复核帧，不能清除前序证据；
- 任一阶段已经判为坏帧的帧仍保持最高优先级。

因此 S3 候选、C3 部分手可见和“无 Action 的静止复核”等状态会一直保留到最终报告和导出掩码，除非人工审核流程在后续版本中明确改变其状态。

## 9. 报告提交和源版本锁

每个媒体流单独保存一份清洗报告：

    <dataset>/.alicePD/curation/
      <episode>--media-<media_id>.curation.alice

非 Full 还保留一个 Episode 级兼容别名。S1 修复补丁写入：

    <dataset>/.alicePD/curation-repairs/
      <episode>--media-<media_id>.s1-repair.alice

报告至少保存：

- source_video 和 analysis timeline 信息；
- T0 artifact；
- pressure_integrity 和 tactile_s1；
- stream_bindings；
- S1 repair 摘要；
- S3 cohort 身份；
- stages、findings、pre_vlm_segments 和最终 segments；
- 最多约 300 个界面趋势 samples；
- summary 和 recommendation。

segments 是完整连续帧状态；samples 只是界面趋势摘要，不能代替逐帧状态。

写报告前，管道对实际使用的视频、Joint/Action、压力流、手部变换源、外部手部标定文件和当前目标实际使用的 S3 留一参考源重新计算 source signatures。任一文件在任务期间发生变化，报告不会提交。

加载旧报告时还会检查：

- dataset_id、episode_id 和 media_file_id；
- source signatures；
- CURATION_PIPELINE_VERSION。

源文件变化或版本不一致时要求重新运行，避免把旧坏帧结论套到新数据。

## 10. Full 导出策略

Full 支持四种 output_format。

### 10.1 lerobot

普通 LeRobot 模式只输出最终有效片段：

    最终 valid mask
    → 只填补不超过 0.25 秒且没有任何质量证据的内部短缺口
    → 再删除 VLM idle，默认保留 reach
    → 若请求 Action，删除 horizon 无效或会越过片段边界的帧
    → 删除短于 0.75 秒的碎片
    → 分类并写入 LeRobot

坏帧、待复核帧、未覆盖帧以及 VLM 明确删除的 idle 都属于禁止填补区域，不会被防碎片逻辑重新加入。Reach 默认保留，以维持从目标定位、接近到接触操作的动作因果链。请求 Action 时，系统在最终 analysis timeline 上重新计算 Observation/Action，并保证 target frame 与中间 horizon 不跨越质量缺口或输出片段边界；片段末尾没有未来目标的帧会被删除。每个导出片段会校验视频帧数和数据帧数。

### 10.2 hdf5_mp4

过滤策略与普通 LeRobot 相同。每个片段输出 video.mp4 和 data.hdf5；请求 Action 时还会写入 `observation/state`、`action`、`action_target_source_frame_index` 和 `action_valid`。输出会校验：

- 视频帧数；
- MANO 变换帧数；
- camera 变换帧数；
- MANO 形状为 T × 44 × 4 × 4。

44 个节点由左右两侧各一个 Forearm 和 21 个手节点组成。

### 10.3 subtask_json

每个源 Episode 输出一个 subtasks.json，不要求固定 MANO/LeRobot 变换源。JSON 包含：

- 多个 medium-level subtask；
- 每个 subtask 的开始帧、结束帧和 inclusive 帧范围；
- 嵌套 fine_segments；
- bad_frames 和 bad_frame_ranges；
- review_frames 和 review_frame_ranges；
- 每个中层子任务内部的坏帧、待复核帧和质量状态。

Full 模式下 JSON 的帧号和 FPS 固定使用 Full analysis video 时间轴，并保存 source_frame_positions 追溯到源视频；不会错误地回退到平滑前的高帧率 Episode 帧空间。

如果 VLM 没有返回中层子任务，当前会生成一个覆盖全 Episode 的低置信度 fallback subtask。

### 10.4 episode_lerobot_json

每个 Episode 输出一套独立的完整 LeRobot 数据和同目录 subtasks.json。此模式保留全部 analysis_media 帧，不删除坏帧或待复核帧：

- LeRobot 中保留完整 Episode；
- 坏帧、待复核帧同时写入 Parquet 的 `quality.state`、`quality.is_bad`、`quality.needs_review` 和 JSON；
- 请求 Action 时保留固定行数，但 `quality.action_valid=false` 的行不得用于策略损失；
- filtering.policy 为 keep_all_frames_mark_quality_in_parquet_and_json。

EgoDex/固定 MANO 导出仍需要可验证的手部变换和相机变换。Nexus 是明确例外：当模式为 `nexus_multimodal` 且能力声明包含 `can_nexus_mano21_adapter=true` 时，此按钮会分派到 Nexus 多传感器转换器，不使用固定 MANO 4×4 EgoDex 导出器。

### 10.5 can_full_export=false

固定 MANO/LeRobot 导出能力不足不会阻止前面的平滑、清洗和 VLM。管道允许 Full 先执行到最终报告：

- subtask_json 仍可输出；
- lerobot 和 hdf5_mp4 的固定 MANO 导出会失败；
- episode_lerobot_json 通常会失败，但满足 Nexus 专用适配能力时可以输出 Nexus 完整 Episode 包；
- Episode 标记为 partial，清洗结果保留。

## 11. completed、partial 和 failed

Episode 状态语义如下：

| 状态 | 含义 |
|---|---|
| completed | 平滑、最终清洗和所请求导出均完成；可选 Action 如果未请求不影响完成 |
| partial | 最终清洗报告已完成，但可选 Action 或导出失败 |
| failed | 在必需阶段失败，未形成可发布的完整 Episode 结果 |

Action 生成或 S2 校验失败时，系统仍继续执行质量清洗、VLM 和 C1/C2。若用户明确请求生成 Action，`lerobot`、`hdf5_mp4` 和 `episode_lerobot_json` 会阻止输出缺少 Action 的训练数据；只记录标注的 `subtask_json` 仍可输出。未请求 Action 时，Observation-only 输出不受影响。

输出目录创建失败或导出异常时，已完成的清洗报告不会删除。Full job 的 failures 会记录具体 Episode、失败阶段、错误和清洗产物路径。

只有 completed 或 partial Episode 可以发布为 latest Full run。发布还会校验 media_file_id，防止所选视频在发布前发生身份切换。

整个 run 的状态：

- 所有 Episode 都失败：failed；
- 有成功结果但同时存在失败或 partial：partial；
- 没有失败：completed；
- 用户取消：cancelled。

## 12. 当前实现边界

当前管道仍有以下明确边界：

1. S4 FK 一致性未实现。
2. S5 基座和方向统一未实现。
3. Nexus/OpenXR 的外部 C3 不估算缺失外参，只验证并使用明确提供的静态 4×4 标定；时间变化外参尚未支持。
4. 整手可见性当前判断 MANO21 点是否位于 RGB 视锥和画面范围内，不使用深度遮挡、物体遮挡或手部自遮挡模型。
5. Action/S2 中间产物可以保持源传感器时间轴；正式导出不直接切片该旧数组，而是在冻结后的 Full analysis timeline 上按最终 source positions、S1 修复和投影归正重新计算 Action。
6. C3 视频画质保证至少 15 FPS，但高于 15 FPS 的视频仍可能抽样，不等同于逐帧完整解码评分。
7. C1 是词集合匹配，不是语义蕴含模型。
8. S3 留一参考仍依赖 embodiment_id 和完整 dimension_names；无法建立安全 cohort 时只产生待复核候选。

## 13. 代码与测试导航

| 责任 | 文件或入口 |
|---|---|
| 请求参数 | app/schemas.py::CurationJobRequest |
| API 提交、查询与预检 | app/main.py |
| 阶段编排、状态合并和报告 | app/curation_pipeline.py |
| 统一质量证据与 provenance | app/quality_evidence.py |
| 数据集模式契约 | app/dataset_modes.py |
| EgoDex/Nexus/OpenXR C3 整手可见性 | app/hand_visibility.py |
| Nexus20 到 MANO21 | app/nexus_mano.py |
| OpenXR26 到 MANO21 | app/openxr_mano.py |
| 视频平滑和 EIS | app/video_smoothing.py |
| Full run、timeline_id 和 latest 发布 | app/full_run.py |
| Full 过滤和四种输出格式 | app/full_export.py |
| 默认 LeRobot Parquet、Action 与质量列 | app/lerobot_export.py |
| Nexus 多传感器 Episode Package | app/nexus_lerobot_export.py |
| 清洗阶段测试 | tests/test_curation_pipeline.py |
| C3 测试 | tests/test_hand_visibility.py |
| Full run 一致性测试 | tests/test_full_run.py |
| Full 导出测试 | tests/test_full_export.py |
| 统一质量证据测试 | tests/test_quality_evidence.py |
| 模式分流测试 | tests/test_dataset_modes.py、tests/test_nexus_mano.py、tests/test_openxr_mano.py |

下一章进入 VLM 三级行为标注，说明候选帧抽样、Coarse/Medium/Fine 协议、Joint 边界微调、结果复用以及与 C1/C2 和导出的衔接。
