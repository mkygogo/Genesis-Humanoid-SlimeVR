"""
Genesis VMC FK Final (Fix v2)
修复了 KeyError: 'neck' 的问题。
原理：忽略 SlimeVR 发送的错误位置，利用【旋转数据】+【预设骨骼长度】在 Python 中重建火柴人。
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
# 只要这些长度存在，人就绝不会缩成一团！
# 你可以根据自己的身高微调这些值
LENS = {
    "spine": 0.50,      # 脊柱 (腰->胸)
    "neck": 0.10,       # 脖子 (胸->头)  <-- 补上了这个
    "shoulder_w": 0.18, # 肩宽的一半
    "arm": 0.30,        # 大臂长度
    "forearm": 0.25,    # 小臂长度
    "hip_w": 0.08,      # 胯宽的一半
    "thigh": 0.45,      # 大腿长度
    "calf": 0.42,       # 小腿长度
}

# 全局高度补偿 (米)
# 如果人陷在地里，把这个改大 (比如 0.85)
GLOBAL_OFFSET = np.array([0.0, 0.0, 0.85]) 

# 存储旋转四元数 (默认单位旋转)
rots = {
    "Hips": R.identity(), "Spine": R.identity(), "Chest": R.identity(),
    "Head": R.identity(),
    "LeftUpperArm": R.identity(), "LeftLowerArm": R.identity(),
    "RightUpperArm": R.identity(), "RightLowerArm": R.identity(),
    "LeftUpperLeg": R.identity(), "LeftLowerLeg": R.identity(),
    "RightUpperLeg": R.identity(), "RightLowerLeg": R.identity(),
}
# 存储 Hips 位置 (这是 SlimeVR 发送的唯一可信位置)
hips_raw_pos = np.zeros(3)
data_lock = threading.Lock()

# ================= VMC 接收逻辑 =================
def vmc_handler(address, *args):
    global hips_raw_pos
    bone_name = args[0]
    
    # 提取位置 (x,y,z) 和 旋转 (qx,qy,qz,qw)
    pos = np.array([args[1], args[2], args[3]])
    q = np.array([args[4], args[5], args[6], args[7]])
    
    # 🔄 关键修正：Unity(左手系) -> Genesis(右手系) 旋转转换
    # 交换 X/Y 分量通常能修复大部分镜像/倒置问题
    # 原始: [x, y, z, w] -> 修正: [y, x, z, w] (经验值)
    q_fix = np.array([q[1], q[0], q[2], q[3]])
    
    with data_lock:
        # 1. 处理 Hips 位置
        if bone_name == "Hips":
            # 坐标系转换: Unity(x,y,z) -> Gen(z,-x,y)
            hips_raw_pos = np.array([pos[2], -pos[0], pos[1]])

        # 2. 名称标准化 (SlimeVR -> 标准命名)
        target = bone_name
        if bone_name == "LeftArm": target = "LeftUpperArm"
        if bone_name == "RightArm": target = "RightUpperArm"
        if bone_name == "LeftForeArm" or bone_name == "LeftHand": target = "LeftLowerArm"
        if bone_name == "RightForeArm" or bone_name == "RightHand": target = "RightLowerArm"
        if bone_name == "LeftUpLeg": target = "LeftUpperLeg"
        if bone_name == "RightUpLeg": target = "RightUpperLeg"
        if bone_name == "LeftLeg": target = "LeftLowerLeg"
        if bone_name == "RightLeg": target = "RightLowerLeg"
        
        if target in rots:
            rots[target] = R.from_quat(q_fix)

def start_server():
    disp = dispatcher.Dispatcher()
    disp.map("/VMC/Ext/Bone/Pos", vmc_handler)
    try:
        server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", VMC_PORT), disp)
        print(f"🚀 VMC-FK 接收器就绪 (端口 {VMC_PORT})")
        server.serve_forever()
    except OSError:
        print(f"❌ 端口 {VMC_PORT} 被占用！请关闭其他 Python 窗口。")

# ================= 正向运动学 (FK) 计算 =================
def apply_rot(rot_obj, vec):
    """将局部骨骼向量应用旋转"""
    return rot_obj.apply(vec)

def calc_skeleton():
    """从 Hips 开始，像搭积木一样算出全身坐标"""
    with data_lock:
        r = rots.copy()
        # 根节点：腰
        root = hips_raw_pos.copy() + GLOBAL_OFFSET
    
    # 防止腰部数据错误导致钻地，强制给一个最小高度
    if root[2] < 0.5: root[2] = 0.85
    
    points = {"Hips": root}
    
    # 1. 上半身: Hips -> Spine -> Chest -> Head/Arms
    # 假设人体初始 T-Pose: 脊柱向上(Z), 手臂向两侧(Y/X)
    
    # 脊柱 (向上)
    vec_spine = apply_rot(r["Spine"], [0, 0, LENS["spine"]])
    points["Chest"] = root + vec_spine
    points["Head"] = points["Chest"] + [0, 0, LENS["neck"]]
    
    # 左臂 (左肩 -> 左肘 -> 左手)
    l_shoulder = points["Chest"] + [0, LENS["shoulder_w"], 0] # 左移
    vec_l_arm = apply_rot(r["LeftUpperArm"], [0, 0, -LENS["arm"]]) # 假设初始垂手(向下)
    points["L_Elbow"] = l_shoulder + vec_l_arm
    vec_l_fore = apply_rot(r["LeftLowerArm"], [0, 0, -LENS["forearm"]])
    points["L_Hand"] = points["L_Elbow"] + vec_l_fore
    
    # 右臂
    r_shoulder = points["Chest"] + [0, -LENS["shoulder_w"], 0] # 右移
    vec_r_arm = apply_rot(r["RightUpperArm"], [0, 0, -LENS["arm"]])
    points["R_Elbow"] = r_shoulder + vec_r_arm
    vec_r_fore = apply_rot(r["RightLowerArm"], [0, 0, -LENS["forearm"]])
    points["R_Hand"] = points["R_Elbow"] + vec_r_fore
    
    # 2. 下半身: Hips -> Thigh -> Knee -> Ankle
    # 左腿
    l_hip = root + [0, LENS["hip_w"], 0]
    vec_l_thigh = apply_rot(r["LeftUpperLeg"], [0, 0, -LENS["thigh"]]) # 向下
    points["L_Knee"] = l_hip + vec_l_thigh
    vec_l_calf = apply_rot(r["LeftLowerLeg"], [0, 0, -LENS["calf"]])
    points["L_Foot"] = points["L_Knee"] + vec_l_calf # 这里其实是脚踝

    # 右腿
    r_hip = root + [0, -LENS["hip_w"], 0]
    vec_r_thigh = apply_rot(r["RightUpperLeg"], [0, 0, -LENS["thigh"]])
    points["R_Knee"] = r_hip + vec_r_thigh
    vec_r_calf = apply_rot(r["RightLowerLeg"], [0, 0, -LENS["calf"]])
    points["R_Foot"] = points["R_Knee"] + vec_r_calf
    
    return points

# ================= Genesis 主程序 =================
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
    
    print("\n✨ 终极火柴人模式运行中... 正在根据旋转重建身体")
    
    # 颜色定义
    c_map = {
        "Hips": (0,1,1), "Chest": (1,0,1), "Head": (1,1,0),
        "L_Foot": (0,0,1), "R_Foot": (1,1,1),
        "L_Hand": (1,0,0), "R_Hand": (0,1,0)
    }

    last_print = time.time()

    while True:
        try:
            scene.clear_debug_objects()
        except: pass
        
        # 1. 调用我们的 FK 算法算出所有点的位置
        pts = calc_skeleton()
        
        # 2. 绘制
        for name, pos in pts.items():
            color = c_map.get(name, (0.5, 0.5, 0.5))
            radius = 0.06 if name in c_map else 0.04
            scene.draw_debug_sphere(pos=pos.tolist(), radius=radius, color=color)
            
        # 3. 连线 (画出火柴人)
        lines = [
            ("Hips", "Chest"), ("Chest", "Head"),
            ("Chest", "L_Elbow"), ("L_Elbow", "L_Hand"),
            ("Chest", "R_Elbow"), ("R_Elbow", "R_Hand"),
            ("Hips", "L_Knee"), ("L_Knee", "L_Foot"),
            ("Hips", "R_Knee"), ("R_Knee", "R_Foot")
        ]
        for p1, p2 in lines:
            if p1 in pts and p2 in pts:
                scene.draw_debug_line(start=pts[p1].tolist(), end=pts[p2].tolist(), radius=0.005, color=(0.8,0.8,0.8))

        # 3. 终端心跳包
        if time.time() - last_print > 1.0:
            lh_z = pts["L_Hand"][2]
            head_z = pts["Head"][2]
            print(f"\r[FK状态] 头高: {head_z:.2f}m | 左手高: {lh_z:.2f}m | (若数值变化说明成功!)    ", end="")
            last_print = time.time()

        scene.step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()