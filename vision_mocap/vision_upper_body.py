"""
Genesis Visual Mocap (Upper Body Edition)
专为【半身/坐姿】设计。
核心逻辑：以【头部】为锚点，计算双手的相对位置。
解决了只拍上半身时，因为髋部丢失导致的全身乱飘问题。
"""
import cv2
import mediapipe as mp
import numpy as np
import genesis as gs
import time
import threading

# ================= 配置 =================
CAMERA_INDEX = 0
VIS_SCALE = 1.2   # 动作幅度放大倍数 (建议 1.0 ~ 1.5)
HEAD_HEIGHT = 1.3 # 设定头在 Genesis 里的固定高度 (米)

# 共享数据
current_pose = {}
data_lock = threading.Lock()

def webcam_thread():
    global current_pose
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # 使用 0.10.9 版本的 API (solutions)
    # 如果报错，请确保安装的是 mediapipe==0.10.9
    mp_pose = mp.solutions.pose
    
    # 启用半身模式 (model_complexity=1 兼顾速度和精度)
    pose = mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1 
    )

    print(f"📷 摄像头启动... 请确保【头】和【手】在画面内")

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        # 镜像翻转 (让你像照镜子一样，操作更直观)
        image = cv2.flip(image, 1)
        
        # 识别
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # 提取关键点 (0=Nose, 15=L_Wrist, 16=R_Wrist)
            # 我们用 鼻子(0) 代表头部中心
            nose = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
            l_hand = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])
            r_hand = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
            
            # === 核心算法：计算相对向量 ===
            # MediaPipe 坐标系: x右, y下, z深(前)
            # 我们计算手相对于鼻子的向量
            vec_l = l_hand - nose
            vec_r = r_hand - nose
            
            # 存入共享数据
            with data_lock:
                current_pose["vec_l"] = vec_l
                current_pose["vec_r"] = vec_r
        
        # 显示画面 (画个框框辅助对准)
        image.flags.writeable = True
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Upper Body Mocap', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================= Genesis 渲染 =================
def main():
    threading.Thread(target=webcam_thread, daemon=True).start()

    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=True, rigid_options=gs.options.RigidOptions(dt=0.01))
    
    # 地面
    scene.add_entity(gs.morphs.Plane())
    scene.build()
    
    # 颜色
    c_head = (1, 1, 0) # 黄
    c_l = (1, 0, 0)    # 红 (左手)
    c_r = (0, 1, 0)    # 绿 (右手)

    print("\n✅ Genesis 就绪。")
    print("黄色=头 (固定), 红色=左手, 绿色=右手")

    while True:
        try: scene.clear_debug_objects()
        except: pass
        
        # 1. 设定头的固定位置 (Genesis 坐标系)
        # Genesis: x前, y左, z上
        head_pos_gen = np.array([0.5, 0.0, HEAD_HEIGHT]) # 头在前方0.5米，高1.3米
        
        vec_l = None
        vec_r = None
        
        with data_lock:
            if "vec_l" in current_pose:
                vec_l = current_pose["vec_l"]
                vec_r = current_pose["vec_r"]

        if vec_l is not None:
            # 2. 坐标映射 (MediaPipe -> Genesis)
            # MP: x(右), y(下), z(深)
            # Gen: x(前), y(左), z(上)
            
            # 映射规则：
            # MP x(右) -> Gen -y (左)
            # MP y(下) -> Gen -z (下)
            # MP z(深) -> Gen -x (后)  <-- 注意这里，手伸出去 z 变负
            
            def map_vec(v):
                # 这里的符号需要根据实际体感微调
                gx = -v[2] * VIS_SCALE # 深度 -> 前后
                gy = -v[0] * VIS_SCALE # 左右 -> 左右
                gz = -v[1] * VIS_SCALE # 上下 -> 上下
                return np.array([gx, gy, gz])

            # 3. 计算手的绝对坐标
            l_hand_pos = head_pos_gen + map_vec(vec_l)
            r_hand_pos = head_pos_gen + map_vec(vec_r)
            
            # 4. 绘制
            # 头
            scene.draw_debug_sphere(pos=head_pos_gen.tolist(), radius=0.1, color=c_head)
            # 左手
            scene.draw_debug_sphere(pos=l_hand_pos.tolist(), radius=0.06, color=c_l)
            scene.draw_debug_line(start=head_pos_gen.tolist(), end=l_hand_pos.tolist(), radius=0.005, color=(0.5,0,0))
            # 右手
            scene.draw_debug_line(start=head_pos_gen.tolist(), end=r_hand_pos.tolist(), radius=0.005, color=(0,0.5,0))
            scene.draw_debug_sphere(pos=r_hand_pos.tolist(), radius=0.06, color=c_r)

        scene.step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()