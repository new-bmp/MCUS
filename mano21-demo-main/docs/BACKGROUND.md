# 项目背景与技术详情

> 本文件为 Cursor AI 提供完整项目背景，用于后续迭代开发。

## 1. 项目起源

### 背景
用户（绿龙/技术团队）正在研究具身数据采集和机器人遥操作技术。手部是关键交互器官，手部检测是遥操作的重要基础。

### 目标
快速搭建一个基于 **MediaPipe Hands** 的手部21点关键点检测演示系统：
- 输入：iPhone拍摄的手部/双手视频
- 处理：MediaPipe Hands 21点检测
- 输出：带有关键点标注的视频 + 手势识别结果

### 设计原则
1. **快速落地** - 2小时内可运行
2. **技术验证** - 验证MediaPipe在iPhone视频上的检测效果
3. **可迭代** - 代码结构清晰，方便后续扩展到机器人控制

---

## 2. 技术架构

### 核心组件

```
输入视频 (MP4/MOV)
    ↓
OpenCV 视频读取
    ↓
MediaPipe Hands (21点检测)
    ├── 21个关键点坐标 (x, y, z)
    ├── 骨骼连接 (HAND_CONNECTIONS)
    └── 手掌外接圆
    ↓
手势识别模块
├── 关节角度计算
└── 规则匹配 (THUMBS_UP, FIST, ROCK, etc.)
    ↓
可视化输出
├── 关键点 + 骨骼绘制
├── 手势标签
└── 关节角度显示
    ↓
输出视频 (可选)
```

### MediaPipe Hands 原理

MediaPipe Hands 使用：
1. **BlazeFace** 作为手掌检测器（轻量快速）
2. **3D关键点回归网络** 预测21个关键点的3D坐标
3. **航向推断**（Dead reckoning）追踪多帧间的关键点

21个关键点包含：
- 4个指尖点（拇指、食指、中指、无名指、小指）
- 每个手指3个关节点
- 1个手腕点

### 坐标系

MediaPipe使用归一化坐标 [0, 1]：
- `x`, `y`: 图像平面坐标
- `z`: 垂直于图像平面的深度坐标（相对手腕的距离）

---

## 3. 当前实现状态

### 已完成
- ✅ Python脚本 `mano21_demo.py`
- ✅ 视频文件读取支持
- ✅ 21点关键点检测和可视化
- ✅ 双手检测
- ✅ 基本手势识别（THUMBS_UP, FIST, ROCK, SCISSORS等）
- ✅ 关节角度计算
- ✅ 输出视频保存

### 待完成 / 可迭代方向
- ⏳ 实时摄像头模式
- ⏳ ROS/机器人控制接口
- ⏳ 深度相机支持 (Intel RealSense, Azure Kinect)
- ⏳ 手指尖跟踪增强
- ⏳ 多手势扩展
- ⏳ 性能优化

---

## 4. 数据格式

### 输入视频规格
- iPhone拍摄
- 分辨率：通常1080p (1920x1080) 或 4K
- 格式：MOV 或 MP4
- 帧率：30fps 或 60fps

### 关键点数据结构

```python
# MediaPipe 返回的关键点
results.multi_hand_landmarks[hand_idx].landmark[idx]

# 每个landmark包含:
landmark.x  # 归一化x [0, 1]
landmark.y  # 归一化y [0, 1]
landmark.z  # 深度z (相对于手腕的距离)
landmark.visibility  # 可见性置信度
```

### 骨骼连接定义

```python
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
    (0, 9), (9, 10), (10, 11), (11, 12), # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20), # 小指
    (5, 9), (9, 13), (13, 17)             # 手掌横联
]
```

---

## 5. 机器人遥操作集成

### 目标应用场景

```
人类手部动作 → 视频检测 → 关键点 → 映射 → 机械手控制
```

### 关键考量

1. **延迟**：从检测到控制信号的延迟需要 < 100ms
2. **精度**：关键点检测精度影响机械手抓取成功率
3. **鲁棒性**：光线变化、遮挡、手势速度的影响
4. **坐标系转换**：图像坐标 → 相机坐标 → 机械手坐标

### 关节角度映射

```python
# 示例：计算每个手指的弯曲角度
def finger_angle(tip, dip, pip, mcp):
    """计算手指弯曲角度"""
    # 从指尖到手腕的方向向量
    v1 = tip - pip
    v2 = mcp - pip
    # 计算角度
    angle = arccos(dot(v1, v2) / (|v1| * |v2|))
    return angle

# 映射到机械手关节
robot_finger_angle = map_human_to_robot_angle(finger_angle)
```

---

## 6. 技术选型理由

### 为什么选择 MediaPipe？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **MediaPipe Hands** | 快速、免费、准确、开源 | 需要Google环境 |
| OpenPose | 功能全面 | 慢、吃GPU |
| TensorFlow Lite | 可嵌入式 | 需要自己训练 |
| Apple Vision | 原生iOS | 跨平台困难 |

### 为什么用 Python？

1. **快速验证** - Python原型开发快
2. **MediaPipe原生支持** - Python SDK完善
3. **方便迭代** - 逻辑清晰，易修改
4. **后续可移植** - 核心逻辑可移植到C++/ROS

---

## 7. 视频素材说明

### 用户iPhone视频
用户会提供iPhone拍摄的手部视频进行演示：
- 建议双手在镜头前
- 包含不同手势
- 分辨率足够（iPhone默认1080p+）

### 测试数据
`~/Downloads/teleop_data_1/` 包含机器人遥操作数据：
- `observation.images.head/` - 头部相机视角
- `observation.images.left_wrist/` - 左腕部视角
- `observation.images.right_wrist/` - 右腕部视角

可用于算法测试参考。

---

## 8. 扩展方向

### 短期迭代
1. 添加摄像头实时模式
2. 优化手势识别准确率
3. 添加关键点追踪（平滑）
4. 输出关节角度数据到JSON

### 中期迭代
1. ROS节点封装
2. 深度信息融合
3. 接入实际机械手（UR5/Franka）
4. 移动端部署（iOS/Android）

### 长期愿景
1. 双手协调动作分析
2. 力反馈遥操作
3. 多模态感知融合（视觉+触觉）
4. 端到端模仿学习数据采集

---

## 9. 已知问题和限制

1. **遮挡问题**：手指重叠时检测不稳定
2. **角度计算**：当前使用简化模型，不够精确
3. **深度信息**：z坐标精度有限
4. **快速运动**：高速手势可能出现跳变
5. **侧向视角**：侧握手机时检测效果下降

---

## 10. 参考资料

- [MediaPipe Hands 官方文档](https://google.github.io/mediapipe/solutions/hands)
- [MediaPipe Model Card](https://storage.googleapis.com/mediapipe-assets/hand_landmark_model_card.pdf)
- [Hand Keypoint Detection](https://arxiv.org/abs/2212.09594)
- [MediaPipe Python Examples](https://github.com/google/mediapipe/tree/master/mediapipe/python/solutions)
