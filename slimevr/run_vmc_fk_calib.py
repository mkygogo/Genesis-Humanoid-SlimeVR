"""
Genesis VMC FK Final (Calibration Edition)
不需要 Pinocchio 库！纯 Numpy 实现。
包含【一键校准】功能：站直后在终端按回车，瞬间修复动作畸变。
"""
import time
import numpy as np
import genesis as gs
import threading
from pythonosc import dispatcher, osc_server
from scipy.spatial.transform import Rotation as R

# ================= 配置 =================
VMC_PORT = 39539

# 📐 骨骼长度 (单位: 米)
LENS = {
    "spine": 0.50,      # 脊柱
    "neck": 0.15,       # 脖子
    "shoulder_w": 0.18, # 肩宽一半
    "arm": 0.30,        # 大臂
    "forearm": 0.25,    # 小臂
    "hip_w": 0.08,      # 胯宽一半
    "thigh": 0.45,      # 大腿
    "calf": 0.42,       # 小腿
}

# 全局高度 (米) - 让人站在地面上
GLOBAL_OFFSET = np.array([0.0, 0.0, 0.88]) 

# ----------------------------------------
# 数据容器
# ----------------------------------------
# 1. 原始数据 (来自 SlimeVR)
raw_rots = {
    "Hips": R.identity(), "Spine": R.identity(), "Chest": R.identity(), "Head": R.identity(),
    "LeftUpperArm": R.identity(), "LeftLowerArm": R.identity(),
    "RightUpperArm": R.identity(), "RightLowerArm": R.identity(),
    "LeftUpperLeg": R.identity(), "LeftLowerLeg": R.identity(),
    "RightUpperLeg": R.identity(), "RightLowerLeg": R.identity(),
}
hips_raw_pos = np.zeros(3)

# 2. 校准数据 (修正偏差)
calib_offsets = {k: R.identity() for k in raw_rots}

data_lock = threading.Lock()

# ----------------------------------------
# 核心功能：校准
# ----------------------------------------
def perform_calibration():
    """
    当用户站直 (I-Pose) 时调用。
    原理：记录当前所有骨骼的旋转值，将其视为'零旋转'状态。
    """
    print("\n⚡ 正在执行校准... (请保持直立 I-Pose)")
    with data_lock:
        for name, rot in raw_rots.items():
            # 计算当前旋转的逆，作为修正偏移量
            # 之后：Corrected_Rot = Raw_Rot * Offset
            calib_offsets[name] = rot.inv()
    print("✅ 校准完成！现在你的动作应该完全正常了。")

# ----------------------------------------
# VMC 数据接收
# ----------------------------------------
def vmc_handler(address, *args):
    global hips_raw_pos
    bone_name = args[0]
    
    pos = np.array([args[1], args[2], args[3]])
    q = np.array([args[4], args[5], args[6], args[7]])
    
    # 坐标系转换 Unity -> Genesis (交换XY, 反转ZW)
    # 这是一个经验公式，用于修复手性差异
    q_fix = np.array([q[1], q[0], -q[2], -q[3]])
    
    with data_lock:
        if bone_name == "Hips":
            hips_raw_pos = np.array([pos[2], -pos[0], pos[1]])

        # 名称映射 (兼容 SlimeVR 的命名)
        target = bone_name
        if bone_name == "LeftArm": target = "LeftUpperArm"
        if bone_name == "RightArm": target = "RightUpperArm"
        if bone_name == "LeftForeArm" or bone_name == "LeftHand": target = "LeftLowerArm"
        if bone_name == "RightForeArm" or bone_name == "RightHand": target = "RightLowerArm"
        if bone_name == "LeftUpLeg": target = "LeftUpperLeg"
        if bone_name == "RightUpLeg": target = "RightUpperLeg"
        if bone_name == "LeftLeg": target = "LeftLowerLeg"
        if bone_name == "RightLeg": target = "RightLowerLeg"
        
        if target in raw_rots:
            raw_rots[target] = R.from_quat(q_fix)

def start_server():
    disp = dispatcher.Dispatcher()
    disp.map("/VMC/Ext/Bone/Pos", vmc_handler)
    try:
        server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", VMC_PORT), disp)
        print(f"🚀 接收器就绪 (端口 {VMC_PORT})")
        print("👉 请站直，并在终端按 [Enter] 键进行校准！")
        server.serve_forever()
    except:
        print(f"❌ 端口 {VMC_PORT} 被占用")

# ----------------------------------------
# FK 解算 (手动计算骨骼位置)
# ----------------------------------------
def apply_rot(rot_obj, vec):
    return rot_obj.apply(vec)

def calc_skeleton():
    with data_lock:
        # 获取当前修正后的旋转
        curr_rots = {}
        for k, v in raw_rots.items():
            curr_rots[k] = v * calib_offsets[k]
            
        root = hips_raw_pos.copy() + GLOBAL_OFFSET
    
    # 强制高度修正 (防止还没穿戴好时人陷在地里)
    if root[2] < 0.6: root[2] = 0.88
    
    pts = {"Hips": root}
    
    # 辅助函数: 级联计算
    def add_bone(parent_name, my_name, rot_name, local_vec):
        r = curr_rots.get(rot_name, R.identity())
        global_vec = apply_rot(r, local_vec)
        pts[my_name] = pts[parent_name] + global_vec

    # --- 躯干 ---
    add_bone("Hips", "Chest", "Spine", [0, 0, LENS["spine"]])
    add_bone("Chest", "Head", "Head",  [0, 0, LENS["neck"]])
    
    # --- 腿部 ---
    # 左腿
    l_hip_origin = root + [0, LENS["hip_w"], 0]
    pts["L_Hip_Joint"] = l_hip_origin # 虚拟锚点
    
    r_l_up = curr_rots["LeftUpperLeg"]
    pts["L_Knee"] = l_hip_origin + apply_rot(r_l_up, [0, 0, -LENS["thigh"]])
    
    r_l_low = curr_rots["LeftLowerLeg"]
    pts["L_Foot"] = pts["L_Knee"] + apply_rot(r_l_low, [0, 0, -LENS["calf"]])
    
    # 右腿
    r_hip_origin = root + [0, -LENS["hip_w"], 0]
    pts["R_Hip_Joint"] = r_hip_origin
    
    r_r_up = curr_rots["RightUpperLeg"]
    pts["R_Knee"] = r_hip_origin + apply_rot(r_r_up, [0, 0, -LENS["thigh"]])
    
    r_r_low = curr_rots["RightLowerLeg"]
    pts["R_Foot"] = pts["R_Knee"] + apply_rot(r_r_low, [0, 0, -LENS["calf"]])

    # --- 手臂 ---
    # 左臂
    l_sh_origin = pts["Chest"] + [0, LENS["shoulder_w"], -0.05]
    r_l_arm = curr_rots["LeftUpperArm"]
    pts["L_Elbow"] = l_sh_origin + apply_rot(r_l_arm, [0, 0, -LENS["arm"]])
    
    r_l_fore = curr_rots["LeftLowerArm"]
    pts["L_Hand"] = pts["L_Elbow"] + apply_rot(r_l_fore, [0, 0, -LENS["forearm"]])
    
    # 右臂
    r_sh_origin = pts["Chest"] + [0, -LENS["shoulder_w"], -0.05]
    r_r_arm = curr_rots["RightUpperArm"]
    pts["R_Elbow"] = r_sh_origin + apply_rot(r_r_arm, [0, 0, -LENS["arm"]])
    
    r_r_fore = curr_rots["RightLowerArm"]
    pts["R_Hand"] = pts["R_Elbow"] + apply_rot(r_r_fore, [0, 0, -LENS["forearm"]])
    
    return pts

# ----------------------------------------
# 主程序
# ----------------------------------------
def main():
    # 1. 启动接收线程
    t1 = threading.Thread(target=start_server, daemon=True)
    t1.start()
    
    # 2. 启动键盘监听线程 (用于校准)
    def key_loop():
        while True:
            try:
                input() # 等待用户按回车
                perform_calibration()
            except: break
    t2 = threading.Thread(target=key_loop, daemon=True)
    t2.start()

    # 3. 启动 Genesis
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=True, rigid_options=gs.options.RigidOptions(dt=0.01))
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()
    
    print("\n✅ 启动成功！")
    print("-----------------------------------")
    print("1. 请像军姿一样站直（双手自然下垂）。")
    print("2. 点击这个黑色的终端窗口。")
    print("3. 按下 [Enter] 键进行校准。")
    print("-----------------------------------")
    
    # 颜色映射
    c_map = {
        "Hips":(0,1,1), "Head":(1,1,0), 
        "L_Hand":(1,0,0), "R_Hand":(0,1,0), 
        "L_Foot":(0,0,1), "R_Foot":(1,0,1)
    }

    while True:
        try: scene.clear_debug_objects()
        except: pass
        
        pts = calc_skeleton()
        
        # 绘制点
        for name, pos in pts.items():
            color = c_map.get(name, (0.6, 0.6, 0.6))
            radius = 0.05 if name in c_map else 0.03
            scene.draw_debug_sphere(pos=pos.tolist(), radius=radius, color=color)
            
        # 绘制连线
        lines = [
            ("Hips","Chest"), ("Chest","Head"),
            ("Hips","L_Knee"), ("L_Knee","L_Foot"),
            ("Hips","R_Knee"), ("R_Knee","R_Foot"),
            ("Chest","L_Elbow"), ("L_Elbow","L_Hand"),
            ("Chest","R_Elbow"), ("R_Elbow","R_Hand")
        ]
        for p1, p2 in lines:
            if p1 in pts and p2 in pts:
                scene.draw_debug_line(start=pts[p1].tolist(), end=pts[p2].tolist(), radius=0.005, color=(0.8,0.8,0.8))

        scene.step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()