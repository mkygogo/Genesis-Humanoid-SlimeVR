"""
Genesis VMC Full Body FK (8-Point Support)
支持：胸、腰、双手、双大腿、双小腿 (共8点)
特性：
1. 真正的膝盖弯曲 (大腿+小腿双级联动)。
2. 胸部独立驱动，头部跟随。
"""
import time
import numpy as np
import genesis as gs
import threading
from pythonosc import dispatcher, osc_server
from scipy.spatial.transform import Rotation as R

# ================= 配置 =================
VMC_PORT = 39539

# 📐 人体骨骼长度 (单位: 米)
# 请根据你的身高比例微调这些值，效果会更好
LENS = {
    "spine_lower": 0.20, # 腰到胸下沿
    "spine_upper": 0.25, # 胸长度
    "neck":        0.15, # 脖子长 (胸到头)
    
    "shoulder_w":  0.18, # 肩宽的一半
    "arm":         0.28, # 大臂长
    "forearm":     0.25, # 小臂长
    
    "hip_w":       0.08, # 胯宽的一半
    "thigh":       0.42, # 大腿长 (Hip -> Knee)
    "calf":        0.40, # 小腿长 (Knee -> Ankle)
}

# 全局高度修正 (米)
# 也就是 Hips(腰) 在站立时的基准高度，通常是 0.85 ~ 0.95
GLOBAL_Z_OFFSET = 0.85 

# 数据容器: 存储所有骨骼的旋转四元数
rots = {
    "Hips": R.identity(), "Spine": R.identity(), "Chest": R.identity(),
    "Head": R.identity(),
    "LeftUpperArm": R.identity(), "LeftLowerArm": R.identity(),
    "RightUpperArm": R.identity(), "RightLowerArm": R.identity(),
    # 腿部关键点
    "LeftUpLeg": R.identity(),  "LeftLeg": R.identity(), 
    "RightUpLeg": R.identity(), "RightLeg": R.identity(),
}

# Hips 位置 (来自 VMC 的唯一位置源)
hips_raw_pos = np.zeros(3)
data_lock = threading.Lock()

# ================= VMC 数据接收 =================
def vmc_handler(address, *args):
    global hips_raw_pos
    bone_name = args[0]
    
    # 提取位置和旋转
    pos = np.array([args[1], args[2], args[3]])
    q = np.array([args[4], args[5], args[6], args[7]])
    
    # 🔄 旋转修正: VMC(Unity左手系) -> Genesis(右手系)
    # 交换 X/Y 分量通常能修复大部分镜像问题，如果发现反了请调整这里
    # q_fix = [x, y, z, w]
    q_fix = np.array([q[1], q[0], q[2], q[3]]) 
    
    with data_lock:
        # 1. 只有 Hips 记录位置 (并转换坐标系: Unity Z -> Gen X, Unity X -> Gen -Y)
        if bone_name == "Hips":
            # 这里的 x,y,z 是为了适配 Genesis 的 Z-up
            # Unity(x,y,z) -> Gen(z, -x, y)
            hips_raw_pos = np.array([pos[2], -pos[0], pos[1]])

        # 2. 映射骨骼名称 (SlimeVR -> 标准名)
        target = bone_name
        
        # 手臂映射
        if bone_name == "LeftArm": target = "LeftUpperArm"
        if bone_name == "RightArm": target = "RightUpperArm"
        if bone_name == "LeftForeArm" or bone_name == "LeftHand": target = "LeftLowerArm"
        if bone_name == "RightForeArm" or bone_name == "RightHand": target = "RightLowerArm"
        
        # 腿部映射 (关键!)
        # SlimeVR输出: LeftUpLeg(大腿), LeftLeg(小腿), LeftFoot(脚)
        if bone_name == "LeftUpLeg": target = "LeftUpLeg"
        if bone_name == "LeftLeg":   target = "LeftLeg" # 小腿
        if bone_name == "RightUpLeg": target = "RightUpLeg"
        if bone_name == "RightLeg":   target = "RightLeg" # 小腿

        if target in rots:
            rots[target] = R.from_quat(q_fix)

def start_server():
    disp = dispatcher.Dispatcher()
    disp.map("/VMC/Ext/Bone/Pos", vmc_handler)
    try:
        server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", VMC_PORT), disp)
        print(f"🚀 全身FK解算器已启动 (端口 {VMC_PORT})")
        server.serve_forever()
    except OSError:
        print(f"❌ 端口 {VMC_PORT} 被占用！")

# ================= FK 核心算法 =================
def apply_rot(rot_obj, vec):
    """将局部向量应用旋转"""
    return rot_obj.apply(vec)

def solve_full_body_fk():
    """计算全身关键点位置"""
    with data_lock:
        r = rots.copy()
        # 基准点: Hips (加上全局高度补偿)
        root_pos = hips_raw_pos.copy()
        # 如果 VMC 发来的高度太低(比如0)，强制给一个站立高度
        if root_pos[2] < 0.2: 
            root_pos[2] = GLOBAL_Z_OFFSET
    
    pts = {"Hips": root_pos}
    
    # ---------------------------------------------
    # 1. 脊柱链 (Hips -> Chest -> Head)
    # ---------------------------------------------
    # 既然你有一个 Chest Tracker，我们认为 Chest 的旋转是它自己的
    # Hips -> (Spine) -> Chest
    vec_spine = apply_rot(r["Spine"], [0, 0, LENS["spine_lower"]])
    chest_base = root_pos + vec_spine
    
    # Chest 本身的旋转决定了上半身的朝向
    # Chest -> Neck -> Head
    vec_upper_chest = apply_rot(r["Chest"], [0, 0, LENS["spine_upper"]])
    neck_base = chest_base + vec_upper_chest
    pts["Chest"] = neck_base # 这是胸部可视化的位置
    
    vec_head = apply_rot(r["Head"], [0, 0, LENS["neck"]]) # 假设头跟着胸动，或者有单独旋转
    pts["Head"] = neck_base + vec_head

    # ---------------------------------------------
    # 2. 腿部链 (Hips -> Knee -> Ankle) *重点*
    # ---------------------------------------------
    # 左腿
    l_hip_joint = root_pos + [0, LENS["hip_w"], 0] # 胯宽偏移
    # 大腿向量 (由 LeftUpLeg 旋转决定)
    vec_l_thigh = apply_rot(r["LeftUpLeg"], [0, 0, -LENS["thigh"]]) 
    pts["L_Knee"] = l_hip_joint + vec_l_thigh
    
    # 小腿向量 (由 LeftLeg 旋转决定)
    vec_l_calf = apply_rot(r["LeftLeg"], [0, 0, -LENS["calf"]])
    pts["L_Ankle"] = pts["L_Knee"] + vec_l_calf # 这就是我们要的"左脚"位置
    
    # 右腿
    r_hip_joint = root_pos + [0, -LENS["hip_w"], 0]
    vec_r_thigh = apply_rot(r["RightUpLeg"], [0, 0, -LENS["thigh"]])
    pts["R_Knee"] = r_hip_joint + vec_r_thigh
    
    vec_r_calf = apply_rot(r["RightLeg"], [0, 0, -LENS["calf"]])
    pts["R_Ankle"] = pts["R_Knee"] + vec_r_calf # 右脚位置

    # ---------------------------------------------
    # 3. 手臂链 (Chest -> Elbow -> Hand)
    # ---------------------------------------------
    # 左臂
    l_shoulder = neck_base + [0, LENS["shoulder_w"], -0.05]
    vec_l_arm = apply_rot(r["LeftUpperArm"], [0, 0, -LENS["arm"]])
    pts["L_Elbow"] = l_shoulder + vec_l_arm
    vec_l_fore = apply_rot(r["LeftLowerArm"], [0, 0, -LENS["forearm"]])
    pts["L_Hand"] = pts["L_Elbow"] + vec_l_fore
    
    # 右臂
    r_shoulder = neck_base + [0, -LENS["shoulder_w"], -0.05]
    vec_r_arm = apply_rot(r["RightUpperArm"], [0, 0, -LENS["arm"]])
    pts["R_Elbow"] = r_shoulder + vec_r_arm
    vec_r_fore = apply_rot(r["RightLowerArm"], [0, 0, -LENS["forearm"]])
    pts["R_Hand"] = pts["R_Elbow"] + vec_r_fore

    return pts

# ================= 渲染 =================
def main():
    threading.Thread(target=start_server, daemon=True).start()

    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.8),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(dt=0.01),
    )
    plane = scene.add_entity(gs.morphs.Plane())
    scene.build()
    
    print("\nVisualizer Running...")
    
    # 颜色映射
    c_map = {
        "Hips": (0,1,1),     # 青 (核心)
        "Chest": (1,0,1),    # 洋红 (胸)
        "Head": (1,1,0),     # 黄 (头)
        "L_Ankle": (0,0,1),  # 蓝 (左脚)
        "R_Ankle": (1,1,1),  # 白 (右脚)
        "L_Hand": (1,0,0),   # 红 (左手)
        "R_Hand": (0,1,0),   # 绿 (右手)
        # 膝盖和手肘用灰色小球显示，辅助观察
        "L_Knee": (0.5,0.5,0.5), "R_Knee": (0.5,0.5,0.5),
        "L_Elbow": (0.5,0.5,0.5), "R_Elbow": (0.5,0.5,0.5)
    }

    while True:
        try:
            scene.clear_debug_objects()
        except: pass

        # 1. 解算
        pts = solve_full_body_fk()
        
        # 2. 绘制点
        for name, pos in pts.items():
            color = c_map.get(name, (0.3, 0.3, 0.3))
            radius = 0.06 if name in ["Hips", "Chest", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand"] else 0.03
            scene.draw_debug_sphere(pos=pos.tolist(), radius=radius, color=color)
            
        # 3. 绘制骨架线 (看着更像人)
        link_pairs = [
            ("Hips", "Chest"), ("Chest", "Head"),
            ("Hips", "L_Knee"), ("L_Knee", "L_Ankle"),
            ("Hips", "R_Knee"), ("R_Knee", "R_Ankle"),
            ("Chest", "L_Elbow"), ("L_Elbow", "L_Hand"),
            ("Chest", "R_Elbow"), ("R_Elbow", "R_Hand")
        ]
        for p1, p2 in link_pairs:
            scene.draw_debug_line(start=pts[p1].tolist(), end=pts[p2].tolist(), radius=0.005, color=(0.8,0.8,0.8))

        scene.step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()