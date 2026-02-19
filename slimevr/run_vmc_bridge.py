"""
Genesis VMC FK Bridge (Final Fix)
忽略 SlimeVR 发送的手脚位置(因为是0)，直接利用【旋转】和【骨骼长度】计算手脚坐标。
"""
import time
import numpy as np
import genesis as gs
import threading
from pythonosc import dispatcher, osc_server
from scipy.spatial.transform import Rotation as R

# ================= 配置 =================
VMC_PORT = 39539

# 🤖 机器人/人体骨骼长度设置 (单位: 米)
# 你可以根据实际情况微调这些值
LENS = {
    "spine":     0.55,  # 脊柱长度 (腰 -> 胸/头)
    "shoulder":  0.20,  # 肩宽的一半
    "upper_arm": 0.30,  # 大臂长度
    "lower_arm": 0.35,  # 小臂长度
    "thigh":     0.45,  # 大腿长度
    "calf":      0.45,  # 小腿长度
}

# 全局高度补偿 (米)
GLOBAL_OFFSET = np.array([0.0, 0.0, 0.0]) 

# 数据容器
# 存储的是【旋转四元数】，而不是位置
tracker_rots = {
    "Hips":      R.identity(),
    "Spine":     R.identity(),
    "Chest":     R.identity(),
    "Head":      R.identity(),
    "LeftUpperLeg": R.identity(), "LeftLowerLeg": R.identity(),
    "RightUpperLeg":R.identity(), "RightLowerLeg":R.identity(),
    "LeftUpperArm": R.identity(), "LeftLowerArm": R.identity(),
    "RightUpperArm":R.identity(), "RightLowerArm":R.identity(),
}
# 髋部位置是唯一的真实位置源
hips_pos = np.zeros(3)
data_lock = threading.Lock()

# ================= VMC 监听 =================
def vmc_handler(address, *args):
    global hips_pos
    try:
        bone_name = args[0]
        # VMC发送的是 (x, y, z, qx, qy, qz, qw)
        pos = np.array([args[1], args[2], args[3]])
        # 旋转: qx, qy, qz, qw
        q = np.array([args[4], args[5], args[6], args[7]])
        
        # 坐标系转换: Unity(左手) -> Genesis(右手 Z-up)
        # 这是一个经验转换，用于对齐
        # 这里的旋转转换比较 trick，直接使用原始四元数配合后续的轴映射
        
        # 只有 Hips 的位置是可信的 (记得做坐标转换: Unity Z -> Gen X, Unity X -> Gen -Y)
        if bone_name == "Hips":
             # 这里的转换对应: 人朝前 -> 机器人朝前
             new_pos = np.array([pos[2], -pos[0], pos[1]])
             with data_lock:
                 hips_pos = new_pos

        # 处理旋转名称映射 (SlimeVR 有时用不同名字)
        target_bone = bone_name
        if bone_name == "LeftUpLeg": target_bone = "LeftUpperLeg"
        if bone_name == "RightUpLeg": target_bone = "RightUpperLeg"
        if bone_name == "LeftLeg": target_bone = "LeftLowerLeg"
        if bone_name == "RightLeg": target_bone = "RightLowerLeg"
        if bone_name == "LeftArm": target_bone = "LeftUpperArm"
        if bone_name == "RightArm": target_bone = "RightUpperArm"
        if bone_name == "LeftForeArm": target_bone = "LeftLowerArm"
        if bone_name == "RightForeArm": target_bone = "RightLowerArm"

        if target_bone in tracker_rots:
            with data_lock:
                tracker_rots[target_bone] = R.from_quat(q)
                
    except Exception:
        pass

def start_server():
    disp = dispatcher.Dispatcher()
    disp.map("/VMC/Ext/Bone/Pos", vmc_handler)
    try:
        server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", VMC_PORT), disp)
        print(f"🚀 FK解算模式已启动 (端口 {VMC_PORT})...")
        server.serve_forever()
    except OSError:
        print(f"❌ 端口 {VMC_PORT} 占用！请关闭其他窗口。")

# ================= FK 计算核心 =================
def rotate_vec(rot_obj, vec):
    """将局部向量应用旋转"""
    # 注意：这里可能需要处理 Unity -> Genesis 的旋转手性差异
    # 简单方案：直接应用，观察是否反向
    v = rot_obj.apply(vec)
    # 修正坐标轴映射: Unity(x,y,z) -> Genesis(z,-x,y)
    return np.array([v[2], -v[0], v[1]])

def solve_fk():
    """从 Hips 开始，一级级算出末端位置"""
    with data_lock:
        root = hips_pos.copy() + GLOBAL_OFFSET
        rots = tracker_rots.copy()
    
    points = {}
    points["Hips"] = root
    
    # 1. 腿部 FK
    # 左腿 (假设自然下垂是 Unity 的 -Y 轴)
    vec_l_thigh = rotate_vec(rots["LeftUpperLeg"], [0, -LENS["thigh"], 0])
    points["L_Knee"] = root + vec_l_thigh
    
    vec_l_calf = rotate_vec(rots["LeftLowerLeg"], [0, -LENS["calf"], 0])
    points["L_Foot"] = points["L_Knee"] + vec_l_calf
    
    # 右腿
    vec_r_thigh = rotate_vec(rots["RightUpperLeg"], [0, -LENS["thigh"], 0])
    points["R_Knee"] = root + vec_r_thigh
    
    vec_r_calf = rotate_vec(rots["RightLowerLeg"], [0, -LENS["calf"], 0])
    points["R_Foot"] = points["R_Knee"] + vec_r_calf
    
    # 2. 躯干 FK
    vec_spine = rotate_vec(rots["Spine"], [0, LENS["spine"], 0])
    chest_pos = root + vec_spine
    points["Chest"] = chest_pos
    
    # 3. 手臂 FK
    # 左臂 (向左伸展是 Unity -X 轴)
    # 注意：这里简化处理，肩膀位置基于 Chest 固定偏移
    l_shoulder_pos = chest_pos + np.array([0, 0.15, 0]) # 稍微向左(Y)偏移
    vec_l_arm = rotate_vec(rots["LeftUpperArm"], [-LENS["upper_arm"], 0, 0])
    points["L_Elbow"] = l_shoulder_pos + vec_l_arm
    
    vec_l_forearm = rotate_vec(rots["LeftLowerArm"], [-LENS["lower_arm"], 0, 0])
    points["L_Hand"] = points["L_Elbow"] + vec_l_forearm
    
    # 右臂 (向右伸展是 Unity +X 轴)
    r_shoulder_pos = chest_pos + np.array([0, -0.15, 0])
    vec_r_arm = rotate_vec(rots["RightUpperArm"], [LENS["upper_arm"], 0, 0])
    points["R_Elbow"] = r_shoulder_pos + vec_r_arm
    
    vec_r_forearm = rotate_vec(rots["RightLowerArm"], [LENS["lower_arm"], 0, 0])
    points["R_Hand"] = points["R_Elbow"] + vec_r_forearm
    
    return points

# ================= Genesis 主程序 =================
def main():
    threading.Thread(target=start_server, daemon=True).start()

    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=True, rigid_options=gs.options.RigidOptions(dt=0.01))
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()

    print("\n✅ FK 模式运行中。正在尝试重构骨架...")
    
    # 颜色定义
    c_map = {
        "Hips": (0,1,1), "Chest": (1,1,0),
        "L_Foot": (0,0,1), "R_Foot": (1,0,1),
        "L_Hand": (1,0,0), "R_Hand": (0,1,0)
    }

    while True:
        try:
            scene.clear_debug_objects()
        except: pass

        # 计算所有点的位置
        pts = solve_fk()
        
        # 绘制
        for name, pos in pts.items():
            # 画点
            color = c_map.get(name, (0.8, 0.8, 0.8)) # 默认灰色
            radius = 0.06 if name in c_map else 0.03
            scene.draw_debug_sphere(pos=pos.tolist(), radius=radius, color=color)
            
        # 画骨架连线 (火柴人)
        lines = [
            ("Hips", "L_Knee"), ("L_Knee", "L_Foot"),
            ("Hips", "R_Knee"), ("R_Knee", "R_Foot"),
            ("Hips", "Chest"),
            ("Chest", "L_Elbow"), ("L_Elbow", "L_Hand"),
            ("Chest", "R_Elbow"), ("R_Elbow", "R_Hand")
        ]
        for p1, p2 in lines:
            if p1 in pts and p2 in pts:
                scene.draw_debug_line(start=pts[p1].tolist(), end=pts[p2].tolist(), radius=0.005, color=(1,1,1))

        scene.step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()