#!/usr/bin/env python3
"""
Mano21 Hand Tracking Demo - 双手21点关键点检测演示

功能:
- 读取视频文件进行手部21点关键点检测
- 支持单手/双手检测
- 可视化21个关键点和骨骼连线
- 支持手势识别（石头/剪刀/布、点赞等）
- 输出处理后的视频

依赖:
    pip install mediapipe opencv-python

用法:
    python3 src/mano21_demo.py --video assets/hand_video.mp4
    python3 src/mano21_demo.py --video assets/hand_video.mp4 --output output/result.mp4
    python3 src/mano21_demo.py --video assets/hand_video.mp4 --gesture  # 启用手势识别
"""

import cv2
import mediapipe as mp
import argparse
import numpy as np
from pathlib import Path

# 21个手部关键点定义（MediaPipe标准）
# 参考: https://google.github.io/mediapipe/solutions/hands
HAND_LANDMARKS = {
    0: "WRIST",           # 手腕
    1: "THUMB_CMC",       # 拇指腕掌关节
    2: "THUMB_MCP",       # 拇指掌指关节
    3: "THUMB_IP",        # 拇指指间关节
    4: "THUMB_TIP",       # 拇指指尖
    5: "INDEX_MCP",       # 食指掌指关节
    6: "INDEX_PIP",       # 食指近端指间关节
    7: "INDEX_DIP",       # 食指远端指间关节
    8: "INDEX_TIP",       # 食指指尖
    9: "MIDDLE_MCP",      # 中指掌指关节
    10: "MIDDLE_PIP",     # 中指近端指间关节
    11: "MIDDLE_DIP",     # 中指远端指间关节
    12: "MIDDLE_TIP",     # 中指指尖
    13: "RING_MCP",       # 无名指掌指关节
    14: "RING_PIP",       # 无名指近端指间关节
    15: "RING_DIP",       # 无名指远端指间关节
    16: "RING_TIP",       # 无名指指尖
    17: "PINKY_MCP",      # 小指掌指关节
    18: "PINKY_PIP",      # 小指近端指间关节
    19: "PINKY_DIP",      # 小指远端指间关节
    20: "PINKY_TIP",      # 小指尖
}

# 手势识别规则
GESTURE_RULES = {
    "THUMBS_UP": lambda angles: angles["thumb_tip"] < angles["thumb_ip"] * 0.5,
    "THUMBS_DOWN": lambda angles: angles["thumb_tip"] > angles["thumb_ip"] * 1.5,
    "OPEN_PALM": lambda angles: all([
        angles["index_pip"] > 150,
        angles["middle_pip"] > 150,
        angles["ring_pip"] > 150,
        angles["pinky_pip"] > 150,
    ]),
    "FIST": lambda angles: all([
        angles["index_pip"] < 90,
        angles["middle_pip"] < 90,
        angles["ring_pip"] < 90,
        angles["pinky_pip"] < 90,
    ]),
    "ROCK": lambda angles: all([
        angles["index_tip"] < 60,
        angles["pinky_tip"] < 60,
        angles["middle_pip"] > 150,
        angles["ring_pip"] > 150,
    ]),
    "SCISSORS": lambda angles: all([
        angles["index_pip"] < 90,
        angles["middle_pip"] < 90,
        angles["ring_pip"] > 150,
        angles["pinky_pip"] > 150,
    ]),
}


def calculate_joint_angles(landmarks):
    """计算手指关节弯曲角度
    
    按照 PROJECT_SPEC.md 规格，计算每个手指的关节角度：
    - thumb_mcp: 拇指掌指关节角度
    - thumb_ip: 拇指指间关节角度
    - index_pip/dip: 食指 PIP/DIP 角度
    - middle_pip/dip: 中指 PIP/DIP 角度
    - ring_pip/dip: 无名指 PIP/DIP 角度
    - pinky_pip/dip: 小指 PIP/DIP 角度
    """
    angles = {}

    def angle_3points(p1, p2, p3):
        """计算三点形成的角度 (p2为顶点)"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    def finger_extension_angle(tip, pip, mcp):
        """计算手指伸展角度 (指尖相对于MCP的伸展程度)
        返回角度值：180表示完全伸展，0表示完全弯曲
        """
        return angle_3points(mcp, pip, tip)

    # 拇指角度
    angles["thumb_mcp"] = angle_3points(landmarks[1], landmarks[2], landmarks[3])
    angles["thumb_ip"] = angle_3points(landmarks[2], landmarks[3], landmarks[4])
    angles["thumb_tip"] = angle_3points(landmarks[1], landmarks[3], landmarks[4])  # 拇指伸展角度

    # 食指角度
    angles["index_pip"] = angle_3points(landmarks[5], landmarks[6], landmarks[7])
    angles["index_dip"] = angle_3points(landmarks[6], landmarks[7], landmarks[8])
    angles["index_tip"] = finger_extension_angle(landmarks[8], landmarks[7], landmarks[6])

    # 中指角度
    angles["middle_pip"] = angle_3points(landmarks[9], landmarks[10], landmarks[11])
    angles["middle_dip"] = angle_3points(landmarks[10], landmarks[11], landmarks[12])
    angles["middle_tip"] = finger_extension_angle(landmarks[12], landmarks[11], landmarks[10])

    # 无名指角度
    angles["ring_pip"] = angle_3points(landmarks[13], landmarks[14], landmarks[15])
    angles["ring_dip"] = angle_3points(landmarks[14], landmarks[15], landmarks[16])
    angles["ring_tip"] = finger_extension_angle(landmarks[16], landmarks[15], landmarks[14])

    # 小指角度
    angles["pinky_pip"] = angle_3points(landmarks[17], landmarks[18], landmarks[19])
    angles["pinky_dip"] = angle_3points(landmarks[18], landmarks[19], landmarks[20])
    angles["pinky_tip"] = finger_extension_angle(landmarks[20], landmarks[19], landmarks[18])

    return angles


def get_angle_summary(angles):
    """获取角度摘要字符串 (用于显示在视频上)"""
    return (
        f"I:{angles['index_pip']:.0f} "
        f"M:{angles['middle_pip']:.0f} "
        f"R:{angles['ring_pip']:.0f} "
        f"P:{angles['pinky_pip']:.0f}"
    )


def get_full_angle_report(angles):
    """获取完整角度报告 (用于调试)"""
    lines = [
        "=== 关节角度报告 ===",
        f"拇指: MCP={angles['thumb_mcp']:.1f}° IP={angles['thumb_ip']:.1f}°",
        f"食指: PIP={angles['index_pip']:.1f}° DIP={angles['index_dip']:.1f}° TIP={angles['index_tip']:.1f}°",
        f"中指: PIP={angles['middle_pip']:.1f}° DIP={angles['middle_dip']:.1f}° TIP={angles['middle_tip']:.1f}°",
        f"无名指: PIP={angles['ring_pip']:.1f}° DIP={angles['ring_dip']:.1f}° TIP={angles['ring_tip']:.1f}°",
        f"小指: PIP={angles['pinky_pip']:.1f}° DIP={angles['pinky_dip']:.1f}° TIP={angles['pinky_tip']:.1f}°",
    ]
    return "\n".join(lines)


def recognize_gesture(angles):
    """识别手势"""
    for gesture, rule in GESTURE_RULES.items():
        try:
            if rule(angles):
                return gesture
        except:
            pass
    return "UNKNOWN"


def draw_hand_info(frame, hand_landmarks, hand_idx, show_details=True):
    """在手部上绘制详细信息"""
    h, w = frame.shape[:2]

    # 绘制每个关键点编号
    for idx, landmark in enumerate(hand_landmarks.landmark):
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        if show_details:
            cv2.putText(frame, str(idx), (cx + 5, cy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 手腕位置
    wrist = hand_landmarks.landmark[0]
    wrist_cx, wrist_cy = int(wrist.x * w), int(wrist.y * h)

    # 计算手掌中心
    palm_landmarks = [0, 1, 5, 9, 13, 17]
    cx_list = [hand_landmarks.landmark[i].x for i in palm_landmarks]
    cy_list = [hand_landmarks.landmark[i].y for i in palm_landmarks]
    palm_cx = int(np.mean(cx_list) * w)
    palm_cy = int(np.mean(cy_list) * h)

    # 绘制手掌外接圆
    palm_radius = int(np.sqrt((max(cx_list) - min(cx_list))**2 +
                               (max(cy_list) - min(cy_list))**2) * h * 0.5)
    cv2.circle(frame, (palm_cx, palm_cy), palm_radius, (255, 255, 0), 2)

    return wrist_cx, wrist_cy


def main():
    parser = argparse.ArgumentParser(
        description="Mano21 Hand Tracking Demo - 双手21点关键点检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 mano21_demo.py --video hand.mp4
  python3 mano21_demo.py --video hand.mp4 --output result.mp4
  python3 mano21_demo.py --video hand.mp4 --gesture --show-angles
        """
    )
    parser.add_argument('--video', '-v', type=str, required=True,
                       help='输入视频路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出视频路径 (可选)')
    parser.add_argument('--max-hands', '-m', type=int, default=2,
                       help='最大检测手数量 (默认: 2)')
    parser.add_argument('--gesture', '-g', action='store_true',
                       help='启用手势识别')
    parser.add_argument('--show-angles', '-a', action='store_true',
                       help='显示关节角度')
    parser.add_argument('--show-id', action='store_true',
                       help='显示关键点编号')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示窗口，仅处理视频')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='检测置信度阈值 (默认: 0.5)')

    args = parser.parse_args()

    # 检查文件存在
    if not Path(args.video).exists():
        print(f"❌ 错误: 视频文件不存在: {args.video}")
        return

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开视频: {args.video}")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 视频: {width}x{height}, {fps}fps, {total_frames}帧")

    # 创建输出视频写入器
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"💾 输出: {args.output}")

    # 初始化MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    with mp_hands.Hands(
        max_num_hands=args.max_hands,
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence
    ) as hands:

        frame_idx = 0
        stats = {"hands_detected": 0, "gestures": {}}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # 绘制关键点和骨骼
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 绘制骨骼
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    # 绘制手部信息
                    wrist_x, wrist_y = draw_hand_info(
                        frame, hand_landmarks, hand_idx, args.show_id
                    )

                    # 手势识别
                    if args.gesture:
                        angles = calculate_joint_angles(hand_landmarks.landmark)
                        gesture = recognize_gesture(angles)

                        # 更新统计
                        stats["hands_detected"] += 1
                        stats["gestures"][gesture] = stats["gestures"].get(gesture, 0) + 1

                        # 显示手势标签
                        cv2.putText(frame, f"🤟 {gesture}", (wrist_x - 30, wrist_y - 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        # 显示角度信息
                        if args.show_angles:
                            angle_text = f"I:{angles['index_pip']:.0f} M:{angles['middle_pip']:.0f} R:{angles['ring_pip']:.0f} P:{angles['pinky_pip']:.0f}"
                            cv2.putText(frame, angle_text, (20, 100 + hand_idx * 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # 标注手编号
                    cv2.putText(frame, f"Hand {hand_idx + 1}",
                               (wrist_x - 30, wrist_y - 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 底部状态栏
                num_hands = len(results.multi_hand_landmarks)
                status = f"🖐️ {num_hands} hand(s) detected"
                if args.gesture:
                    status += " | Gesture: ON"
                cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
                cv2.putText(frame, status, (10, height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                # 无检测结果
                cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
                cv2.putText(frame, "❌ No hands detected", (10, height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # 帧信息
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (width - 180, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 写入输出视频
            if writer:
                writer.write(frame)

            # 显示
            if not args.no_display:
                display = cv2.resize(frame, (min(1280, width), min(720, height)))
                cv2.imshow("Mano21 Demo", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)

            # 进度
            if frame_idx % 30 == 0:
                pct = 100 * frame_idx // total_frames
                print(f"\r  进度: {frame_idx}/{total_frames} ({pct}%)", end="", flush=True)

    print(f"\n✅ 完成! 处理了 {frame_idx} 帧")

    # 打印统计
    if args.gesture and stats["gestures"]:
        print("\n📊 手势统计:")
        for g, count in sorted(stats["gestures"].items(), key=lambda x: -x[1]):
            print(f"  {g}: {count} 次")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
