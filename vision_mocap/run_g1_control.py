"""
Genesis G1 Vision Teleop (V16 Final)
1. 修复 AttributeError: 'RigidEntity' object has no attribute 'n_q'
2. 采用【动态推导】方式获取自由度，不再依赖属性读取。
3. 完美处理 G1 的 50维状态 -> 49维控制 的映射。
"""
import cv2
import mediapipe as mp
import numpy as np
import genesis as gs
import time
import threading
import os

# ================= 配置 =================
CAMERA_INDEX = 0
VIS_SCALE = 1.2      # 动作幅度放大倍数
SMOOTHING = 0.5      # 平滑系数

# 手部 Link 名称 (Unitree G1)
LINK_NAME_L = "left_wrist_roll_link" 
LINK_NAME_R = "right_wrist_roll_link"

# 机器人头部参考高度
ROBOT_HEAD_HEIGHT = 1.25

# 共享数据
target_pose = {"l_hand": None, "r_hand": None}
data_lock = threading.Lock()

# ================= 视觉捕捉线程 =================
def webcam_thread():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1)
    
    prev_l = np.zeros(3)
    prev_r = np.zeros(3)

    print(f"📷 视觉线程启动...")

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_world_landmarks:
            lm = results.pose_world_landmarks.landmark
            nose = np.array([lm[0].x, lm[0].y, lm[0].z])
            l_raw = np.array([lm[15].x, lm[15].y, lm[15].z])
            r_raw = np.array([lm[16].x, lm[16].y, lm[16].z])
            
            vec_l = l_raw - nose
            vec_r = r_raw - nose
            
            # 坐标映射: MP -> Genesis
            def map_vec(v):
                gx = -v[2] * VIS_SCALE 
                gy = -v[0] * VIS_SCALE 
                gz = -v[1] * VIS_SCALE 
                return np.array([gx, gy, gz])

            off_l = map_vec(vec_l)
            off_r = map_vec(vec_r)
            
            # 滤波
            if np.linalg.norm(prev_l) > 0:
                off_l = prev_l * SMOOTHING + off_l * (1 - SMOOTHING)
                off_r = prev_r * SMOOTHING + off_r * (1 - SMOOTHING)
            prev_l, prev_r = off_l, off_r

            with data_lock:
                target_pose["l_hand"] = off_l
                target_pose["r_hand"] = off_r
        
        cv2.imshow('Robot Vision Control', image)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# ================= 机器人仿真 =================
def main():
    threading.Thread(target=webcam_thread, daemon=True).start()

    gs.init(backend=gs.cpu)
    
    # 零重力模式
    scene = gs.Scene(
        show_viewer=True,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0)
        )
    )
    scene.add_entity(gs.morphs.Plane())
    
    # 路径构建
    g1_path = os.path.join("assets", "robot", "unitree_g1", "g1_mocap_29dof_with_hands.xml")
    
    print(f"🤖 加载机器人: {g1_path}")
    
    try:
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file=g1_path,
                pos=(0, 0, 0.78),
            )
        )
    except Exception as e:
        print(f"❌ 加载崩溃: {e}")
        return

    scene.build()
    
    # 获取 Link
    print(f"🔗 正在获取关节: {LINK_NAME_L} / {LINK_NAME_R}")
    try:
        link_l = robot.get_link(LINK_NAME_L)
        link_r = robot.get_link(LINK_NAME_R)
        print("✅ 关节获取成功")
    except:
        print("❌ Link 获取失败")
        return

    robot_head_pos = np.array([0.0, 0.0, ROBOT_HEAD_HEIGHT]) 
    
    # --- 关键修复：动态推导自由度 ---
    # 做一次虚拟的 IK 解算来获取维度
    print("🔧 正在校准自由度...")
    q_dummy = robot.inverse_kinematics(
        link=link_l, 
        pos=robot_head_pos, 
        quat=np.array([1, 0, 0, 0])
    )
    
    n_q = q_dummy.shape[0]      # 状态维数 (例如 50)
    n_dofs = n_q - 1            # 控制维数 (例如 49, 浮动基座差1)
    
    # G1 浮动基座通常占用:
    # q: 前 7 位 (3 pos + 4 quat)
    # dofs: 前 6 位 (3 linear + 3 angular)
    # 我们只控制剩下的关节:
    q_start_idx = 7
    dofs_start_idx = 6
    
    # 生成我们要控制的电机索引列表 [6, 7, ..., 48]
    motor_indices = np.arange(dofs_start_idx, n_dofs)
    
    print(f"✅ 校准完成: n_q={n_q}, n_dofs={n_dofs}")
    print(f"✅ 激活控制: 接管 {len(motor_indices)} 个关节电机")
    print("\n🚀 系统启动！请动动你的左手！")

    while True:
        tgt_l, tgt_r = None, None
        with data_lock:
            if target_pose["l_hand"] is not None:
                tgt_l = robot_head_pos + target_pose["l_hand"]
                tgt_r = robot_head_pos + target_pose["r_hand"]

        if tgt_l is not None:
            try: scene.clear_debug_objects()
            except: pass
            scene.draw_debug_sphere(pos=tgt_l.tolist(), radius=0.05, color=(1,0,0))
            scene.draw_debug_sphere(pos=tgt_r.tolist(), radius=0.05, color=(0,1,0))

            # IK 解算 (左手)
            q_ik = robot.inverse_kinematics(
                link=link_l, 
                pos=tgt_l, 
                quat=np.array([1, 0, 0, 0]), 
            )
            
            # --- 精确切片 ---
            if q_ik.shape[0] > len(motor_indices):
                # 提取关节数据 (从第7个开始)
                q_joints = q_ik[q_start_idx:]
                # 发送给电机 (从第6个开始)
                robot.control_dofs_position(q_joints, motor_indices)
            else:
                robot.control_dofs_position(q_ik)

        scene.step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()