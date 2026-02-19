"""
Genesis Visual Mocap (MediaPipe Edition)
直接使用摄像头捕捉全身动作，驱动 Genesis 里的火柴人。
无需 SlimeVR，无需穿戴设备。
"""
import cv2
import mediapipe as mp
import numpy as np
import genesis as gs
import time
import threading

# ================= 配置 =================
CAMERA_INDEX = 0  # 摄像头ID，通常是0或1
VIS_SCALE = 1.5   # 视觉坐标缩放系数 (根据画面里人的大小微调)
GLOBAL_OFFSET = np.array([0.0, 0.0, 0.85]) # 让人站在地上

# 骨骼连接关系 (MediaPipe 索引)
# 索引参考: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# 共享数据
latest_landmarks = None
data_lock = threading.Lock()

# ================= 视觉捕捉线程 =================
def webcam_thread():
    global latest_landmarks
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # 初始化 MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1 # 0=快, 1=中, 2=准(慢)
    )

    print(f"📷 摄像头已启动...")

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        # 转为 RGB 供 MediaPipe 使用
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 提取关键点
        if results.pose_world_landmarks:
            with data_lock:
                latest_landmarks = results.pose_world_landmarks.landmark
        
        # (可选) 在窗口显示画面，按 Q 退出画面
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Visual Mocap (Press Q to quit view)', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()

# ================= Genesis 渲染 =================
def main():
    # 启动摄像头线程
    t = threading.Thread(target=webcam_thread, daemon=True)
    t.start()

    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=True, rigid_options=gs.options.RigidOptions(dt=0.01))
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()
    
    print("\n✅ 系统就绪。请站在摄像头前，确保全身入镜。")

    while True:
        try: scene.clear_debug_objects()
        except: pass
        
        current_pose = None
        with data_lock:
            if latest_landmarks:
                current_pose = list(latest_landmarks) # 复制一份

        if current_pose:
            # 1. 提取关键点并转换坐标
            # MediaPipe World Landmark: 
            # x: 左/右 (米), y: 上/下 (米, 负数在头顶), z: 深度 (米)
            # Genesis: z-up
            
            # 我们需要把 MediaPipe 的点转换成 Genesis 坐标
            pts = []
            for lm in current_pose:
                # 坐标变换: 
                # MP X (右) -> Gen -Y (左)
                # MP Y (下) -> Gen -Z (下) -> 取反变上
                # MP Z (深) -> Gen -X (前)
                
                # 注意：MediaPipe 的 Hip 中心大约是 (0,0,0)，我们需要偏移
                gx = -lm.z * VIS_SCALE       # 深度变前后
                gy = -lm.x * VIS_SCALE       # 左右变左右
                gz = -lm.y * VIS_SCALE       # 上下变上下
                
                pos = np.array([gx, gy, gz]) + GLOBAL_OFFSET
                pts.append(pos)

            # 2. 绘制点
            # 关键点索引: 11=左肩, 12=右肩, 23=左胯, 24=右胯
            # 15=左手腕, 16=右手腕, 27=左脚踝, 28=右脚踝
            # 0=鼻子
            
            # 绘制所有 33 个点
            for i, pos in enumerate(pts):
                color = (0.5, 0.5, 0.5)
                radius = 0.03
                
                if i in [15, 16]: color = (1, 0, 0) # 手 = 红
                if i in [27, 28]: color = (0, 0, 1) # 脚 = 蓝
                if i == 0:        color = (1, 1, 0); radius=0.08 # 头 = 黄
                
                scene.draw_debug_sphere(pos=pos.tolist(), radius=radius, color=color)

            # 3. 绘制连线
            for connection in CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(pts) and end_idx < len(pts):
                    scene.draw_debug_line(
                        start=pts[start_idx].tolist(),
                        end=pts[end_idx].tolist(),
                        radius=0.005,
                        color=(0.8, 0.8, 0.8)
                    )

        scene.step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()