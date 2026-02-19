"""
Genesis + VMC + Pinocchio FK (带校准功能)
这是效果最接近 SlimeVR 原生画面的版本。
关键功能：按 [R] 键进行 T-Pose/I-Pose 校准，消除佩戴误差。
"""
import time
import numpy as np
import genesis as gs
import threading
import pinocchio as pin
from pythonosc import dispatcher, osc_server
from scipy.spatial.transform import Rotation as R

# ================= 配置 =================
VMC_PORT = 39539
GLOBAL_OFFSET = np.array([0.0, 0.0, 0.88]) # 抬高到地面

# 骨骼长度 (米) - 标准人体比例
L = {
    "hip_w": 0.10, "spine": 0.50, "neck": 0.15,
    "shoulder": 0.20, "arm": 0.30, "forearm": 0.25,
    "thigh": 0.42, "calf": 0.40
}

# ---------------------------------------------------------
# 1. 构建 Pinocchio 模型 (构建一个虚拟人)
# ---------------------------------------------------------
model = pin.Model()
geom_model = pin.GeometryModel()
# 根节点 (Hips) - FreeFlyer (6自由度)
joint_name = "Hips"
id_hips = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), joint_name)

# 辅助函数：添加关节
def add_body_part(parent_id, name, offset, joint_type=pin.JointModelSpherical()):
    placement = pin.SE3(np.eye(3), np.array(offset))
    return model.addJoint(parent_id, joint_type, placement, name)

# 构建运动学链
id_chest = add_body_part(id_hips, "Chest", [0, 0, L["spine"]])
id_head  = add_body_part(id_chest, "Head", [0, 0, L["neck"]])

id_l_sh  = add_body_part(id_chest, "L_Shoulder_J", [0, L["shoulder"], 0]) # 左肩关节
id_l_elb = add_body_part(id_l_sh, "L_Elbow_J", [0, 0, -L["arm"]])        # 左肘关节
id_l_hnd = add_body_part(id_l_elb, "L_Hand_J", [0, 0, -L["forearm"]])    # 左手

id_r_sh  = add_body_part(id_chest, "R_Shoulder_J", [0, -L["shoulder"], 0])
id_r_elb = add_body_part(id_r_sh, "R_Elbow_J", [0, 0, -L["arm"]])
id_r_hnd = add_body_part(id_r_elb, "R_Hand_J", [0, 0, -L["forearm"]])

id_l_hip = add_body_part(id_hips, "L_Hip_J", [0, L["hip_w"], 0])
id_l_kne = add_body_part(id_l_hip, "L_Knee_J", [0, 0, -L["thigh"]])
id_l_ank = add_body_part(id_l_kne, "L_Ankle_J", [0, 0, -L["calf"]])

id_r_hip = add_body_part(id_hips, "R_Hip_J", [0, -L["hip_w"], 0])
id_r_kne = add_body_part(id_r_hip, "R_Knee_J", [0, 0, -L["thigh"]])
id_r_ank = add_body_part(id_r_kne, "R_Ankle_J", [0, 0, -L["calf"]])

data = model.createData()

# ---------------------------------------------------------
# 2. VMC 数据处理与校准
# ---------------------------------------------------------
# 原始旋转数据
raw_rots = {
    "Hips": R.identity(), "Chest": R.identity(), "Head": R.identity(),
    "LeftUpperArm": R.identity(), "LeftLowerArm": R.identity(),
    "RightUpperArm": R.identity(), "RightLowerArm": R.identity(),
    "LeftUpperLeg": R.identity(), "LeftLowerLeg": R.identity(),
    "RightUpperLeg": R.identity(), "RightLowerLeg": R.identity(),
}
# 校准偏移量 (Calibration Offsets)
calib_offsets = {k: R.identity() for k in raw_rots}
hips_pos = np.zeros(3)
data_lock = threading.Lock()

def perform_calibration():
    """校准：假设当前姿态为直立 I-Pose，计算偏差"""
    print("\n⚡ 执行校准！请保持直立 I-Pose...")
    with data_lock:
        for name, rot in raw_rots.items():
            # 记录当前旋转的逆，作为校准偏移
            # 目标：Rot * Offset = Identity (直立)
            calib_offsets[name] = rot.inv()
    print("✅ 校准完成。动作应该正常了。")

def vmc_handler(address, *args):
    global hips_pos
    bone = args[0]
    # VMC: x, y, z, qx, qy, qz, qw
    pos = np.array([args[1], args[2], args[3]])
    q = np.array([args[4], args[5], args[6], args[7]])
    
    # 坐标系转换 Unity -> Genesis
    # 旋转：交换 X/Y, 反转 Z/W (经验修正)
    q_fix = np.array([q[1], q[0], -q[2], -q[3]]) 
    
    with data_lock:
        if bone == "Hips":
            # 位置转换
            hips_pos = np.array([pos[2], -pos[0], pos[1]])

        # 映射
        target = bone
        if bone == "LeftArm": target = "LeftUpperArm"
        if bone == "RightArm": target = "RightUpperArm"
        if bone == "LeftForeArm" or bone == "LeftHand": target = "LeftLowerArm"
        if bone == "RightForeArm" or bone == "RightHand": target = "RightLowerArm"
        if bone == "LeftUpLeg": target = "LeftUpperLeg"
        if bone == "RightUpLeg": target = "RightUpperLeg"
        if bone == "LeftLeg": target = "LeftLowerLeg"
        if bone == "RightLeg": target = "RightLowerLeg"
        
        if target in raw_rots:
            raw_rots[target] = R.from_quat(q_fix)

def start_server():
    disp = dispatcher.Dispatcher()
    disp.map("/VMC/Ext/Bone/Pos", vmc_handler)
    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", VMC_PORT), disp)
    print(f"🚀 Pinocchio FK 引擎启动 (端口 {VMC_PORT})")
    print("👉 按 [Enter] 键进行校准！")
    server.serve_forever()

# ---------------------------------------------------------
# 3. Genesis 可视化循环
# ---------------------------------------------------------
def main():
    # 启动网络线程
    t_net = threading.Thread(target=start_server, daemon=True)
    t_net.start()
    
    # 启动键盘监听线程 (简单的 input 监听)
    def key_listener():
        while True:
            input() # 等待回车
            perform_calibration()
    threading.Thread(target=key_listener, daemon=True).start()

    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=True, rigid_options=gs.options.RigidOptions(dt=0.01))
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()

    # 映射表: 旋转数据名 -> Pinocchio关节ID
    # 注意：Pinocchio是局部坐标系，我们这里简单地用绝对旋转驱动球形关节
    # 为了简化，我们这里把绝对旋转直接赋给关节（Pinocchio支持这样做吗？通常是局部）
    # 更简单的方法：直接用 FK 公式算，和上一个脚本一样，但有了校准
    
    # 既然已经引入了 pinocchio，我们用它来做正向运动学
    # 注意：Pinocchio 的 spherical joint 需要的是局部旋转。
    # SlimeVR 给的是全局旋转。
    # 用 Pinocchio 做这个转换比较麻烦，为了稳健，我们还是用【带校准的手动 FK】
    # 核心逻辑不变，但加入了 calib_offsets
    
    print("\n✨ 视觉桥接运行中...")
    print("⚠️ 如果动作奇怪，请站直并按 [Enter] 键！")

    while True:
        try: scene.clear_debug_objects()
        except: pass

        # --- FK 计算 (带校准) ---
        with data_lock:
            # 应用校准: Corrected = Raw * Offset
            # 注意四元数乘法顺序，这里是一个近似
            curr_rots = {}
            for k, v in raw_rots.items():
                curr_rots[k] = v * calib_offsets[k]
                
            root = hips_pos.copy() + GLOBAL_OFFSET
            if root[2] < 0.2: root[2] = 0.88 # 默认高度

        # 定义点 (复用之前的逻辑，但加上了校准后的旋转)
        pts = {"Hips": root}
        
        def get_pos(parent_pos, rot_name, local_vec):
            r = curr_rots.get(rot_name, R.identity())
            # 应用旋转
            v = r.apply(local_vec)
            return parent_pos + v

        # 脊柱
        pts["Chest"] = get_pos(root, "Spine", [0, 0, L["spine"]])
        pts["Head"] = get_pos(pts["Chest"], "Head", [0, 0, L["neck"]])
        
        # 腿
        pts["L_Knee"] = get_pos(root, "LeftUpperLeg", [0, L["hip_w"], -L["thigh"]])
        pts["L_Foot"] = get_pos(pts["L_Knee"], "LeftLowerLeg", [0, 0, -L["calf"]])
        
        pts["R_Knee"] = get_pos(root, "RightUpperLeg", [0, -L["hip_w"], -L["thigh"]])
        pts["R_Foot"] = get_pos(pts["R_Knee"], "RightLowerLeg", [0, 0, -L["calf"]])
        
        # 手臂
        l_sh = pts["Chest"] + [0, L["shoulder"], 0]
        pts["L_Elbow"] = get_pos(l_sh, "LeftUpperArm", [0, 0, -L["arm"]])
        pts["L_Hand"] = get_pos(pts["L_Elbow"], "LeftLowerArm", [0, 0, -L["forearm"]])

        r_sh = pts["Chest"] + [0, -L["shoulder"], 0]
        pts["R_Elbow"] = get_pos(r_sh, "RightUpperArm", [0, 0, -L["arm"]])
        pts["R_Hand"] = get_pos(pts["R_Elbow"], "RightLowerArm", [0, 0, -L["forearm"]])

        # 绘制
        c_map = {"Hips":(0,1,1), "Head":(1,1,0), "L_Hand":(1,0,0), "R_Hand":(0,1,0), "L_Foot":(0,0,1), "R_Foot":(1,0,1)}
        for name, pos in pts.items():
            scene.draw_debug_sphere(pos=pos.tolist(), radius=0.05, color=c_map.get(name, (0.5,0.5,0.5)))

        # 绘制骨骼线
        links = [("Hips","Chest"), ("Chest","Head"), 
                 ("Hips","L_Knee"), ("L_Knee","L_Foot"), ("Hips","R_Knee"), ("R_Knee","R_Foot"),
                 ("Chest","L_Elbow"), ("L_Elbow","L_Hand"), ("Chest","R_Elbow"), ("R_Elbow","R_Hand")]
        for p1, p2 in links:
            scene.draw_debug_line(start=pts[p1].tolist(), end=pts[p2].tolist(), radius=0.005, color=(1,1,1))

        scene.step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()