# Mano21 手部21点关键点检测 Demo

基于 MediaPipe Hands 的手部关键点检测演示项目，支持视频输入，输出带有关键点标注的视频。

## 功能特性

- ✅ **21点手部关键点检测** - 标定手掌21个骨骼点
- ✅ **双手检测** - 支持同时检测1-2只手
- ✅ **骨骼可视化** - 关键点 + 骨骼连线绘制
- ✅ **手势识别** - 石头/剪刀/布、点赞、举手等
- ✅ **关节角度计算** - 实时计算手指弯曲角度
- ✅ **视频处理** - 读取视频文件，输出处理结果

## 快速开始

### 1. 安装依赖

```bash
cd mano21-demo
pip install -r requirements.txt
```

### 2. 运行 Demo

```bash
# 基本用法
python3 src/mano21_demo.py --video assets/your_hand_video.mp4

# 输出处理后的视频
python3 src/mano21_demo.py --video assets/your_hand_video.mp4 --output output/result.mp4

# 启用手势识别
python3 src/mano21_demo.py --video assets/your_hand_video.mp4 --gesture

# 显示关节角度
python3 src/mano21_demo.py --video assets/your_hand_video.mp4 --gesture --show-angles
```

### 3. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video, -v` | 输入视频路径 | 必需 |
| `--output, -o` | 输出视频路径 | 无 |
| `--max-hands, -m` | 最大检测手数量 | 2 |
| `--gesture, -g` | 启用手势识别 | False |
| `--show-angles, -a` | 显示关节角度 | False |
| `--show-id` | 显示关键点编号 | False |
| `--confidence` | 检测置信度阈值 | 0.5 |

## 项目结构

```
mano21-demo/
├── src/
│   └── mano21_demo.py    # 主程序
├── assets/                # 输入视频 (需用户自己放置)
├── output/               # 输出视频
├── docs/
│   ├── BACKGROUND.md     # 项目背景与技术详情
│   └── HAND_ANATOMY.md  # 手部解剖学参考
├── README.md             # 本文件
├── REQUIREMENTS.md       # 依赖说明
└── PROJECT_SPEC.md       # 项目规格说明
```

## 手部21点关键点

MediaPipe 定义的手部21个关键点：

```
0:  WRIST       - 手腕
1:  THUMB_CMC   - 拇指腕掌关节
2:  THUMB_MCP   - 拇指掌指关节
3:  THUMB_IP    - 拇指指间关节
4:  THUMB_TIP   - 拇指指尖
5:  INDEX_MCP   - 食指掌指关节
6:  INDEX_PIP   - 食指近端指间关节
7:  INDEX_DIP   - 食指远端指间关节
8:  INDEX_TIP   - 食指指尖
9:  MIDDLE_MCP  - 中指掌指关节
10: MIDDLE_PIP  - 中指近端指间关节
11: MIDDLE_DIP  - 中指远端指间关节
12: MIDDLE_TIP  - 中指指尖
13: RING_MCP    - 无名指掌指关节
14: RING_PIP    - 无名指近端指间关节
15: RING_DIP    - 无名指远端指间关节
16: RING_TIP    - 无名指指尖
17: PINKY_MCP   - 小指掌指关节
18: PINKY_PIP   - 小指近端指间关节
19: PINKY_DIP   - 小指远端指间关节
20: PINKY_TIP   - 小指尖
```

## 手势识别

支持以下手势识别：

| 手势 | 说明 | 识别规则 |
|------|------|---------|
| THUMBS_UP | 点赞 | 拇指向上 |
| THUMBS_DOWN | 点踩 | 拇指向下 |
| OPEN_PALM | 摊手 | 所有手指伸展 |
| FIST | 拳头 | 所有手指弯曲 |
| ROCK | 石头 | 食指和小指伸出 |
| SCISSORS | 剪刀 | 食指和中指伸出 |

## 技术栈

- **MediaPipe Hands** - Google开源的手部检测方案
- **OpenCV** - 视频处理和图像渲染
- **NumPy** - 数值计算

## 下一步迭代建议 (for Cursor)

1. **实时摄像头模式** - 支持摄像头实时输入
2. **机械手控制** - 映射到机械手关节角度
3. **多手势扩展** - 增加更多手势识别
4. **三维姿态估计** - 加入深度信息
5. **性能优化** - 减少延迟，提高FPS
6. **UI界面** - 添加滑块、控制面板

## 参考资料

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [MediaPipe Python SDK](https://pypi.org/project/mediapipe/)
