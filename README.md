# alice blue

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

应用地址：<http://127.0.0.1:8000/>  
API 文档：<http://127.0.0.1:8000/docs>

使用 `停止服务.bat` 关闭由启动器创建的服务。日志位于 `.vla_lens/server.out.log` 和 `.vla_lens/server.err.log`。

## 打开数据集

点击页面顶部的 `打开文件夹`，后端会显示 Windows 原生目录选择器。

- 数据集在原路径就地打开，不通过浏览器上传。
- 视频、图像、HDF5、NPY、CSV 等源文件不会被复制。
- Parquet 是受支持的结构化源数据，可直接承载 LeRobot Episode、action、state 和 timestamp。
- `.COMPLETE`、校验和、临时控制文件与 Alice 自有 `.alice` 标注不进入源文件树、Episode 分组或 Schema 输入。
- 每个数据集在源目录旁使用 `.alicePD/<dataset-id>/` 保存 manifest、标注、缓存、导出与辅助文件引用索引。
- 适合直接打开大型 LeRobot、RLDS 转换数据或自定义机器人数据目录。

## 数据结构理解

打开文件夹后，系统会扫描 JSON/JSONL、HDF5、Parquet、NPY/NPZ、CSV/TSV 和媒体文件，记录真实 path、key、shape、dtype。

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
