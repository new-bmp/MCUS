# Mano21 Demo 项目规格说明

## 项目概述

**项目名称**: Mano21 Hand Tracking Demo
**版本**: v1.0
**目标**: 基于MediaPipe Hands的手部21点关键点检测演示系统
**用户**: 技术团队（机器人遥操作研究）

---

## 功能规格

### F1: 视频文件读取
- 支持 MP4, MOV 格式
- 支持任意分辨率
- 支持 30fps / 60fps
- **优先级**: P0 (核心功能)

### F2: 21点关键点检测
- 使用 MediaPipe Hands
- 支持单手检测
- 支持双手同时检测
- 21个关键点全部标定
- **优先级**: P0 (核心功能)

### F3: 骨骼连线可视化
- 绘制21个关键点
- 绘制手指骨骼连线
- 绘制手掌轮廓
- **优先级**: P0 (核心功能)

### F4: 手势识别
- 支持手势: THUMBS_UP, THUMBS_DOWN, OPEN_PALM, FIST, ROCK, SCISSORS
- 实时显示识别结果
- **优先级**: P1 (重要功能)

### F5: 关节角度计算
- 计算每个手指的PIP关节角度
- 可选显示角度数值
- **优先级**: P1 (重要功能)

### F6: 输出视频保存
- 保存处理后的视频
- 包含所有可视化叠加
- **优先级**: P2 (辅助功能)

### F7: 实时摄像头模式
- 读取摄像头实时画面
- 低延迟显示
- **优先级**: P2 (未来迭代)

### F8: ROS接口
- 发布关键点数据到ROS
- 订阅机械手状态
- **优先级**: P3 (长期目标)

---

## 非功能规格

### 性能
- 视频处理: ≥20 FPS (720p)
- 检测延迟: <100ms
- 内存占用: <500MB

### 精度
- 关键点检测精度: 平均误差 <2px (标准条件下)
- 手势识别准确率: >85% (清晰手势)

### 可用性
- 命令行参数清晰
- 错误提示友好
- 处理进度可见

---

## 输入输出规格

### 输入
```yaml
video:
  path: string (required)
  format: [mp4, mov]
  max_hands: int (default: 2)
  detection_confidence: float (default: 0.5)
```

### 输出
```yaml
video:
  path: string (optional, default: none)
  format: mp4
  codec: mp4v
  annotations:
    - landmarks: true
    - connections: true
    - hand_labels: true
    - gesture_label: true
    - angles: optional
```

### 处理数据
```json
{
  "frame": int,
  "hands": [
    {
      "id": int,
      "landmarks": [
        {"x": float, "y": float, "z": float, "name": string}
      ],
      "angles": {
        "thumb_tip": float,
        "thumb_ip": float,
        "index_pip": float,
        ...
      },
      "gesture": string
    }
  ]
}
```

---

## 项目配置

### 环境要求
- Python 3.8+
- macOS / Linux / Windows
- 不需要GPU (MediaPipe CPU版足够)

### 依赖包
```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
```

---

## 迭代计划

### v1.0 (当前)
- [x] 视频文件输入
- [x] 21点检测可视化
- [x] 基本手势识别
- [x] 关节角度计算

### v1.1 (下一迭代)
- [ ] 摄像头实时模式
- [ ] 关键点平滑滤波
- [ ] 多手势扩展
- [ ] JSON数据输出

### v2.0 (未来)
- [ ] ROS节点
- [ ] 深度相机支持
- [ ] 机械手控制接口
- [ ] 移动端部署

---

## 验收标准

1. ✅ 运行 `python3 mano21_demo.py --video test.mp4` 无报错
2. ✅ 输出视频包含21点标注和骨骼连线
3. ✅ 手势识别能正确识别至少3种手势
4. ✅ 处理30秒1080p视频 ≤ 60秒
5. ✅ 代码结构清晰，便于迭代

---

## 文件清单

```
mano21-demo/
├── src/
│   └── mano21_demo.py      # 主程序 (11KB)
├── assets/                  # 输入视频目录 (需用户放置)
├── output/                  # 输出视频目录
├── docs/
│   ├── BACKGROUND.md        # 项目背景 (4KB)
│   └── HAND_ANATOMY.md     # 手部解剖学 (2KB)
├── README.md               # 使用说明
├── REQUIREMENTS.md          # 依赖说明
└── PROJECT_SPEC.md          # 本规格文档
```
