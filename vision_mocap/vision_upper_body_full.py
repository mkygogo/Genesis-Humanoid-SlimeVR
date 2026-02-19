"""
Genesis Visual Mocap (Full Upper Body Edition)
在 Genesis 中完整复现 MediaPipe 捕捉到的上半身骨架。
包含：头、双肩、双肘、双手。
核心逻辑：依然以【鼻子】为锚点，计算其他所有关节的相对位置。
"""
import cv2
import mediapipe as mp
import numpy as np
import genesis as gs
import time
import threading

# ================= 配置 =================
CAMERA_INDEX = 0
VIS_SCALE = 1.2   # 动作幅度放大倍数
HEAD_HEIGHT = 1.3 # 头在 Genesis 里的高度

# 关键点映射 (MediaPipe Index -> Name)
KEYPOINTS = {
    0:  "Nose",
    11: "L_Shoulder", 12: "R_Shoulder",
    13: "L_Elbow",    14: "R_Elbow",
    15: "L_Wrist",    16: "R_Wrist"
}

# 颜色定义
COLORS = {
    "Nose": (1, 1, 0),       # 黄
    "L_Shoulder": (0.5,0,0), "R_Shoulder": (0,0.5,0), # 深红/深绿
    "L_Elbow": (0.8,0,0),    "R_Elbow": (0,0.8,0),    # 鲜红/鲜绿
    "L_Wrist": (1,0,0),      "R_Wrist": (0,1,0)       # 亮红/亮绿
}

# 骨骼连线 (父 -> 子)
BONES = [
    ("Nose", "L_Shoulder"), ("Nose", "R_Shoulder"), # 简化：脖子连向鼻子
    ("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_Wrist"),
    ("R_Shoulder", "R_Elbow"), ("R_Elbow", "R_Wrist"),
    ("L_Shoulder", "R_Shoulder") # 锁骨连线
]

# 共享数据
current_vectors = {} # 存储相对于 Nose 的向量
data_lock = threading.Lock()

def webcam_thread():
    global current_vectors
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1
    )

    print(f"📷 摄像头启动... 请上半身入镜")

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        # 镜像，方便直观操作
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # 获取鼻子(0)作为原点
            nose = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
            
            temp_vecs = {}
            
            # 计算所有关键点相对于鼻子的向量
            for idx, name in KEYPOINTS.items():
                if idx == 0: continue # 跳过鼻子本身
                
                # 获取当前点坐标
                curr = np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
                
                # 计算相对向量
                vec = curr - nose
                temp_vecs[name] = vec
            
            with data_lock:
                current_vectors = temp_vecs
        
        # 显示预览窗口
        image.flags.writeable = True
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Mocap View', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================= Genesis 渲染 =================
def main():
    threading.Thread(target=webcam_thread, daemon=True).start()

    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=True, rigid_options=gs.options.RigidOptions(dt=0.01))
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()
    
    print("\n✅ 系统就绪。")

    # 预定义的固定头部位置
    head_pos_fixed = np.array([0.5, 0.0, HEAD_HEIGHT])

    while True:
        try: scene.clear_debug_objects()
        except: pass
        
        # 获取最新数据
        vecs = None
        with data_lock:
            if current_vectors:
                vecs = current_vectors.copy()

        if vecs:
            # 1. 存储计算后的绝对坐标
            abs_positions = {"Nose": head_pos_fixed}
            
            # 2. 转换坐标系并计算位置
            for name, v in vecs.items():
                # 映射规则：MediaPipe -> Genesis
                # MP x(右) -> Gen -y (左)
                # MP y(下) -> Gen -z (下)
                # MP z(深) -> Gen -x (后)
                gx = -v[2] * VIS_SCALE 
                gy = -v[0] * VIS_SCALE
                gz = -v[1] * VIS_SCALE
                
                pos = head_pos_fixed + np.array([gx, gy, gz])
                abs_positions[name] = pos

            # 3. 绘制球体
            for name, pos in abs_positions.items():
                color = COLORS.get(name, (0.5, 0.5, 0.5))
                radius = 0.08 if name == "Nose" else 0.04
                scene.draw_debug_sphere(pos=pos.tolist(), radius=radius, color=color)

            # 4. 绘制骨骼连线
            for p1_name, p2_name in BONES:
                if p1_name in abs_positions and p2_name in abs_positions:
                    scene.draw_debug_line(
                        start=abs_positions[p1_name].tolist(),
                        end=abs_positions[p2_name].tolist(),
                        radius=0.005,
                        color=(0.8, 0.8, 0.8)
                    )

        scene.step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()