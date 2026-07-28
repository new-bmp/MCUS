# alice blue

## minRE minimal command line pipeline

`minRE` runs without the web UI. It indexes the source in place, then processes
Episodes one at a time: video smoothing, Qwen VLM behavior annotation, the
eight-stage curation report, useless-phase removal, aligned cutting, VLM task
classification, and verified MP4/HDF5 pair export.

```sh
sh minRE.sh "/data/SPM"
sh minRE.sh "/data/SPM" --output "/data/SPM_clean"
sh minRE.sh "/data/SPM" --index-only
```

Windows PowerShell uses the same arguments:

```powershell
.\minRE.ps1 "F:\SPM"
```

The default output is a sibling directory named `<source>_minRE`. Progress and
resume state are stored in `minre-state.json`; the final searchable pair index
is `dataset.json`. Re-running the same command resumes completed stages and
verifies existing pairs before skipping them. Source files are never modified.

VLA 数据集操作有效性审查工具。FastAPI 后端负责目录扫描、媒体解码、模型加载、Qwen 数据结构理解、后台推理、标注持久化和导出。

## 一键启动

Windows 下双击：

```text
一键启动.bat
```

启动器会检查依赖、选择 8000–8010 的可用端口、后台启动服务、等待 YOLOE warm-up 和 API 健康检查，然后打开默认浏览器。重复双击会复用已运行的服务。

也可以运行：

```powershell
.\run.ps1
```

首次恢复环境时运行：

```powershell
.\run.ps1 -Setup
```

Linux/macOS 使用：

```sh
sh run.sh --setup       # 首次创建 .venv 并安装依赖
sh run.sh               # 后台启动，不打开浏览器
sh run.sh --foreground  # 前台运行，日志直接输出到终端
sh stop.sh
```

启动器优先使用项目 `.venv`，API 就绪不再等待 YOLOE warm-up。模型在后台加载，状态可通过健康接口或 CLI 查看。启动时不会自动扫描旧数据集或运行传感器对齐任务；这些任务在打开数据集或进入对应分析流程时按需执行。

应用地址：<http://127.0.0.1:8000/>  
API 文档：<http://127.0.0.1:8000/docs>

使用 `停止服务.bat` 关闭由启动器创建的服务。日志位于 `.vla_lens/server.out.log` 和 `.vla_lens/server.err.log`。

## 纯命令行

项目提供不依赖浏览器的命令行入口：

```text
python -m app.cli doctor
python -m app.cli start
python -m app.cli status
python -m app.cli open "D:\SPM"
python -m app.cli datasets --json
python -m app.cli schema <dataset-id> --analyze
python -m app.cli stop
```

Linux Full 命令行：

```sh
# 自动启动服务，处理整个数据集，并持续显示进度
sh full.sh /data/insert_remove_usb --all

# 只处理指定 Episode；支持重复、逗号分隔和通配符
sh full.sh /data/insert_remove_usb --episode 'episode_00*'

# 查看可选机器人类型
sh full.sh --robots

# 全部分片统一生成 SO-100 / SO-101 的 7D Action
sh full.sh /data/insert_remove_usb --all --robot so100_so101 --source-hand right

# 只提交后台任务，不等待完成
sh full.sh /data/insert_remove_usb --all --detach
```

`full.sh` 会用 `nvidia-smi` 自动发现 GPU。8 张 A800 时默认设置 8 个 Full worker，把 Episode 按帧数均衡拆成 8 个任务，并让全分辨率稳像变换与锐化在各 GPU 间轮转。可显式覆盖：

Action 是可选项。默认 Full 不生成派生 Action：数据里已有原生 Action 或已生成映射时执行 S2，没有时明确跳过 S2；只有传入 `--robot` 时才会为所有分片生成同一种机器人 Action。当前可选类型包括通用单双臂、ALOHA、SO-100/SO-101、Franka Panda、UR5/UR5e、xArm 7 和 AgileX Piper。旧参数 `--action-profile` 仍可继续使用。

```sh
ALICE_GPU_DEVICES=0,1,2,3,4,5,6,7 \
ALICE_FULL_WORKERS=8 \
ALICE_FULL_PARALLEL=8 \
ALICE_FULL_RESTART=1 \
sh full.sh /data/insert_remove_usb --all
```

低分辨率光流估计、S1/S2/C2 和 HDF5 处理仍主要受 CPU 与磁盘限制；A800 负责全分辨率画面变换和清晰度增强。A800/A100 通常不提供 NVENC，视频编码器会在启动时探测并自动回退，不会把“检测到 CUDA”误认为“可以使用 NVENC”。本地 YOLO 不属于 Full 链路，因此 `full.sh` 默认不加载它，避免占用显存。`ALICE_FULL_LOAD_YOLO=1` 可恢复自动加载。CPU 线程会按 worker 数量切分，避免 8 个任务各自占满全部 CPU 核；若 PyTorch 不是 CUDA 构建，启动器会显示警告并回退到 CPU。

`start` 使用 `.vla_lens/server.json` 记录实例 ID、PID、端口和 Python 路径。重复启动会复用健康实例；陈旧状态文件不会导致误杀无关进程；默认在 8000–8010 中选择可用端口。

## 打开数据集

点击页面顶部的 `打开文件夹`，后端会显示 Windows 原生目录选择器。

文件夹采用一级懒加载：如果所选目录包含直接子目录，每个子目录会登记为独立数据集，只有选中的数据集才会扫描并建立 Episode/文件索引；切换到其他数据集时再按需加载。已加载数据集最多在前端保留三个缓存。如果所选目录没有子目录，则按单数据集加载。集合模式的首次切换不自动等待 Qwen，结构理解可通过“再次运行”单独启动。

- 数据集在原路径就地打开，不通过浏览器上传。
- 视频、图像、HDF5、NPY、CSV 等源文件不会被复制。
- Parquet 是受支持的结构化源数据，可直接承载 LeRobot Episode、action、state 和 timestamp。
- `.COMPLETE`、校验和、临时控制文件与 Alice 自有 `.alice` 标注不进入源文件树、Episode 分组或 Schema 输入。
- 每个数据集在源目录旁使用 `.alicePD/<dataset-id>/` 保存 manifest、标注、缓存、导出与辅助文件引用索引。
- 适合直接打开大型 LeRobot、RLDS 转换数据或自定义机器人数据目录。

## 数据结构理解

打开文件夹后，系统会扫描 JSON/JSONL、HDF5、Parquet、NPY/NPZ、CSV/TSV 和媒体文件，记录真实 path、key、shape、dtype。

Schema 探测采用文件夹/扩展名分层抽样：在不同目录和 Episode 范围内均匀选择代表文件，每个文件夹最多探测少量结构化文件。Qwen 只接收这些真实抽样文件的压缩字段清单，不接收整个数据集内容；返回的 source ID 仍必须通过真实清单校验。目录索引本身保持完整，因此文件管理器与导出不会漏文件。

配置 Qwen-VLM 后，系统会识别：

- RGB、Depth 和多相机 vision stream
- joint position、joint velocity、proprio 和 action
- 左手、右手、双臂或共享流
- 压力、力、力矩和触觉传感器
- timestamp、采样率和时间同步方式
- Vision → Joint → Sensor 的对应关系

Qwen 返回的 source id 会与真实结构清单校验。不存在或类型不正确的映射会被丢弃并记录警告。未完成结构理解时，Episode 分析接口会拒绝执行。

H5/HDF5/H5DF 审阅器不会在文件概览阶段读取样本切片。选中字段后，中央区域并排读取第 n 帧和第 n+1 帧，支持帧号、滑杆和字段同步对照。

Parquet 使用同一套字段导航、相邻帧双窗和自动差异比较。后端根据 Row Group 仅读取选定列的目标行，不把整张表载入内存。

## 模型

项目根目录的 `yoloe-26x-seg.pt` 会在启动时自动加载，并由 Ultralytics 识别为 YOLOE segmentation 模型。也可以在模型配置中加载 SAM 或其他 YOLO Segmentation 权重。

Qwen-VLM 使用 OpenAI-compatible `/chat/completions` API。配置保存在本机忽略版本控制的 `.vla_lens` 运行目录中。

## 分析流程

1. 就地打开数据集并生成真实结构清单。
2. Qwen 理解并校验 vision、joint、左右手和传感器映射。
3. OpenCV 抽帧并计算时序运动区域。
4. YOLOE/SAM 执行实例分割与手物接触估计。
5. 可选 Qwen-VLM 使用已校验的数据结构作为上下文进行时窗复核。
6. 无效片段写入版本化 `.alicePD/<dataset>/annotations/<episode>.alice`。
7. 同步生成 `indices/invalid/*.invalid.alice` 区间索引和 `*.invalid.bin` 一帧一位的快速 bitmap。

## Full 标准数据集

页面上的 `Full` 按钮按以下顺序执行：

```text
视频平滑 → S1-S5/C3 → 非红色片段 VLM 标注 → C1/C2 → 去除静止和伸手 → 分类导出
```

S1-S5/C3 后的红色片段不会送入 VLM；绿色和黄色片段都会参与标注。C1/C2 结束后只导出最终通过的连续片段，并使用 VLM 高层任务标签写入 LeRobot task 元数据。

S1 除通用突变、加速度和 Jerk 外，还会对明确标记为 `endpose/rot6d` 的末端位姿计算相邻帧相对旋转角，拒绝旋转跳变及退化的 6D 旋转基向量。普通六维向量不会触发该专用检查。

```text
<源数据集>/
  output/
    data/chunk-000/episode_000000.parquet
    body/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.main/episode_000000.mp4
    meta/info.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    meta/stats.json
    dataset.json
```

默认数据 Parquet 中左右手分别固定为以下 21 点局部顺序；两只手使用完全相同的 `0-20` 定义：

| 局部索引 | 节点 |
|---:|---|
| 0 | `Hand`（腕部） |
| 1-4 | `ThumbKnuckle → ThumbIntermediateBase → ThumbIntermediateTip → ThumbTip` |
| 5-8 | `IndexFingerKnuckle → IndexFingerIntermediateBase → IndexFingerIntermediateTip → IndexFingerTip` |
| 9-12 | `MiddleFingerKnuckle → MiddleFingerIntermediateBase → MiddleFingerIntermediateTip → MiddleFingerTip` |
| 13-16 | `RingFingerKnuckle → RingFingerIntermediateBase → RingFingerIntermediateTip → RingFingerTip` |
| 17-20 | `LittleFingerKnuckle → LittleFingerIntermediateBase → LittleFingerIntermediateTip → LittleFingerTip` |

主数据还包含 camera transform、左右腕 `xyz+rot6d`、VLM phase、源帧号、源 HDF5 行号和时间戳。除左右手这 42 个节点及 `camera` 外，源 HDF5 中所有形状为 `[T,4,4]` 的命名 transform（包括 forearm）都写入独立 Body Parquet，并在 `meta/info.json` 记录顺序。所有 Episode 使用 LeRobot 的全局 `episode_index`、`frame_index`、`index` 和 `task_index`。

`output` 是固定导出根目录，会被 Alice 的源数据扫描器忽略。重复执行 Full 时不会覆盖已有 Episode，而是继续分配新的 LeRobot Episode；`dataset.json` 作为 Alice 审计索引合并保留既有输出。

旧版 HDF5 + MP4 仍可在 Full 对话框选择，或使用 `sh full.sh ... --output-format hdf5_mp4`。该兼容格式继续输出分类目录、`mano/transforms [T,44,4,4]` 与配对视频。默认 LeRobot 只强制要求左右手各 21 个 transform 与 camera；Body 关节按源数据实际存在情况写入。源数据缺少所选格式的必需 transform 时，该 Episode 会明确失败，不会使用补零数据。

## 无效帧快速索引

每次自动分析或人工修改标注后都会原子更新无效帧索引。`.alice` 是带 `alice/annotation/v1` Schema 的 UTF-8 JSON 文档，保存合并后的标注区间；二进制文件以 `ALPDINV1` 开头，随后为 little-endian `uint64` 总帧数和 LSB0 bitmap，其中 `1` 表示该帧已标注无效。旧版 `.json` 标注仍可读取，并在下次保存时迁移。

## 主要 API

- `POST /api/system/open-dataset-folder`
- `POST /api/datasets/open-path`（自动化或服务端调用，不复制文件）
- `GET /api/datasets/{id}/schema`
- `POST /api/datasets/{id}/analyze-schema`
- `GET /api/datasets/{id}/episodes/{episode}/frame`
- `GET /api/datasets/{id}/episodes/{episode}/invalid-index`
- `GET /api/datasets/{id}/episodes/{episode}/invalid-index?frame={frame}`
- `POST /api/models/upload`
- `POST /api/models/configure`
- `POST /api/datasets/{id}/episodes/{episode}/analyze`
- `GET /api/jobs/{job_id}`
- `PATCH /api/datasets/{id}/episodes/{episode}/segments`
- `GET /api/datasets/{id}/export.zip`
- `POST /api/datasets/{id}/export-folder`
