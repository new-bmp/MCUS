#!/usr/bin/env python3
"""
Mano21 Web UI - 手部21点关键点检测可视化

简洁的Web交互界面:
- 上传视频 → 点击RUN → 显示带轨迹的视频

依赖:
    pip install streamlit mediapipe opencv-python numpy

运行:
    streamlit run src/app.py
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from io import BytesIO
import tempfile
import os

# ============== 配置 ==============
st.set_page_config(
    page_title="Mano21 手部检测",
    page_icon="🖐️",
    layout="centered"
)

# 21个手部关键点定义
HAND_LANDMARKS = {
    0: "WRIST", 1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
    5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
    9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
    13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
    17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
}

# 手指颜色 (BGR)
FINGER_COLORS = {
    "thumb": (0, 255, 255),   # 黄色
    "index": (0, 0, 255),     # 红色
    "middle": (0, 255, 0),    # 绿色
    "ring": (255, 0, 0),      # 蓝色
    "pinky": (255, 0, 255),   # 紫色
    "palm": (255, 255, 0),    # 青色
}

# ============== 核心处理函数 ==============

def calculate_joint_angles(landmarks):
    """计算手指关节弯曲角度"""
    angles = {}
    
    def angle_3points(p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    angles["thumb_tip"] = angle_3points(landmarks[2], landmarks[3], landmarks[4])
    angles["thumb_ip"] = angle_3points(landmarks[1], landmarks[2], landmarks[3])
    angles["index_pip"] = angle_3points(landmarks[5], landmarks[6], landmarks[7])
    angles["index_dip"] = angle_3points(landmarks[6], landmarks[7], landmarks[8])
    angles["middle_pip"] = angle_3points(landmarks[9], landmarks[10], landmarks[11])
    angles["ring_pip"] = angle_3points(landmarks[13], landmarks[14], landmarks[15])
    angles["pinky_pip"] = angle_3points(landmarks[17], landmarks[18], landmarks[19])
    
    return angles


def recognize_gesture(angles):
    """识别手势"""
    if all([
        angles["index_pip"] > 150, angles["middle_pip"] > 150,
        angles["ring_pip"] > 150, angles["pinky_pip"] > 150,
    ]):
        return "OPEN_PALM ✋"
    elif all([
        angles["index_pip"] < 90, angles["middle_pip"] < 90,
        angles["ring_pip"] < 90, angles["pinky_pip"] < 90,
    ]):
        return "FIST ✊"
    elif angles["thumb_tip"] < angles["thumb_ip"] * 0.5:
        return "THUMBS_UP 👍"
    elif angles["thumb_tip"] > angles["thumb_ip"] * 1.5:
        return "THUMBS_DOWN 👎"
    elif all([
        angles["index_pip"] < 90, angles["middle_pip"] < 90,
        angles["ring_pip"] > 150, angles["pinky_pip"] > 150,
    ]):
        return "SCISSORS ✌️"
    return "👋"


def process_video(input_bytes, max_hands=2, confidence=0.5, show_gesture=True, show_angles=True):
    """处理视频并返回带注释的视频字节"""
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_input = f.name
        f.write(input_bytes)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_output = f.name
    
    try:
        # 打开视频
        cap = cv2.VideoCapture(temp_input)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建输出视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        # 初始化MediaPipe
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        
        with mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        ) as hands:
            
            frame_count = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 绘制骨骼和关键点
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_draw.DrawingSpec(
                                color=(0, 255, 0), thickness=2, circle_radius=3
                            ),
                            connection_drawing_spec=mp_draw.DrawingSpec(
                                color=(0, 200, 0), thickness=2
                            )
                        )
                        
                        # 获取手腕位置用于显示
                        wrist = hand_landmarks.landmark[0]
                        wrist_x, wrist_y = int(wrist.x * width), int(wrist.y * height)
                        
                        # 手势识别
                        if show_gesture:
                            angles = calculate_joint_angles(hand_landmarks.landmark)
                            gesture = recognize_gesture(angles)
                            
                            # 绘制手势标签背景
                            cv2.rectangle(frame, (wrist_x - 50, wrist_y - 90), 
                                         (wrist_x + 80, wrist_y - 40), (0, 0, 0), -1)
                            cv2.putText(frame, gesture, (wrist_x - 45, wrist_y - 55),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # 显示角度
                            if show_angles:
                                angle_text = f"I:{angles['index_pip']:.0f}° M:{angles['middle_pip']:.0f}°"
                                cv2.rectangle(frame, (10, 30), (280, 70), (0, 0, 0), -1)
                                cv2.putText(frame, angle_text, (15, 55),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # 更新进度
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    progress = frame_count / total_frames
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"处理中... {frame_count}/{total_frames} 帧")
                
                out.write(frame)
        
        progress_bar.progress(1.0)
        status_text.text("处理完成!")
        
        cap.release()
        out.release()
        
        # 读取输出视频
        with open(temp_output, 'rb') as f:
            output_bytes = f.read()
        
        return output_bytes
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_input):
            os.unlink(temp_input)
        if os.path.exists(temp_output):
            os.unlink(temp_output)


# ============== Streamlit UI ==============

def main():
    # 视频宽度CSS
    st.markdown("""
    <style>
    .stVideo video {
        width: 80% !important;
        display: block;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # 标题
    st.title("🖐️ Mano21 手部关键点检测")
    st.markdown("---")
    
    # 上传视频
    st.subheader("📤 上传视频")
    uploaded_file = st.file_uploader(
        "选择视频文件 (MP4/MOV)",
        type=['mp4', 'mov', 'avi'],
        help="支持iPhone或其他设备拍摄的手部视频",
        label_visibility="collapsed"
    )
    
    # 设置（折叠）
    with st.expander("⚙️ 高级设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_hands = st.slider("最大手数量", 1, 2, 2)
            confidence = st.slider("检测置信度", 0.3, 1.0, 0.5, 0.1)
        with col2:
            show_gesture = st.checkbox("显示手势", value=True)
            show_angles = st.checkbox("显示关节角度", value=True)

    st.markdown("")
    
    # RUN 按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        run_button = st.button("🚀 RUN", type="primary", use_container_width=True)

    st.markdown("")
    
    # 处理和显示
    if run_button and uploaded_file is not None:
        st.info(f"📹 已上传: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        with st.spinner("⏳ 处理中，请稍候..."):
            try:
                output_bytes = process_video(
                    uploaded_file.getvalue(),
                    max_hands=max_hands,
                    confidence=confidence,
                    show_gesture=show_gesture,
                    show_angles=show_angles
                )
                
                st.success("✅ 处理完成!")
                st.markdown("")
                
                # 显示结果视频 - 占页面80%宽度
                st.markdown("### 📺 检测结果")
                st.video(output_bytes, use_container_width=True)
                
                # 下载按钮
                col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
                with col_dl1:
                    st.download_button(
                        "📥 下载结果视频",
                        output_bytes,
                        file_name="mano21_result.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"❌ 处理失败: {str(e)}")
    
    elif run_button and uploaded_file is None:
        st.warning("⚠️ 请先上传视频文件")
    
    # 底部说明
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.9em;">
    <b>21个关键点</b>: 手腕 → 拇指(4点) → 食指(4点) → 中指(4点) → 无名指(4点) → 小指(4点)<br>
    <b>技术栈</b>: MediaPipe Hands + OpenCV + Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()