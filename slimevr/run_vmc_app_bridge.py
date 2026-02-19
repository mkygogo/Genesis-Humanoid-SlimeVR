"""
Genesis VMC App Bridge (High Precision)
配合 "Virtual Motion Capture" 软件使用。
直接接收软件解算好的【完美坐标】，无需手动FK。
"""
import time
import numpy as np
import genesis as gs
import threading
from pythonosc import dispatcher, osc_server
from scipy.spatial.transform import Rotation as R

# 监听来自 VMC 软件的端口
LISTEN_PORT = 39539

# 坐标系修正：让 Genesis 的小人站得更自然
# 如果人是躺着的，修改这里
def convert_pos(unity_pos):
    # Unity (x, y, z) -> Genesis (z, -x, y)
    return np.array([unity_pos[2], -unity_pos[0], unity_pos[1]])

# 骨骼映射与颜色
BONE_COLORS = {
    "Hips": (0,1,1), "Head": (1,1,0), "Spine": (0.5,0.5,0.5), "Chest": (1,0,1),
    "LeftHand": (1,0,0), "RightHand": (0,1,0),
    "LeftFoot": (0,0,1), "RightFoot": (1,0,1),
    "LeftLowerLeg": (0,0,0.5), "RightLowerLeg": (0.5,0,0.5), # 膝盖
    "LeftLowerArm": (0.5,0,0), "RightLowerArm": (0,0.5,0)    # 手肘
}

trackers = {}
lock = threading.Lock()

def vmc_handler(address, *args):
    bone_name = args[0]
    if bone_name not in BONE_COLORS: return

    # 这里的 Pos 是 VMC 软件算好的世界坐标
    raw_pos = np.array([args[1], args[2], args[3]])
    
    with lock:
        trackers[bone_name] = convert_pos(raw_pos)

def start_server():
    disp = dispatcher.Dispatcher()
    disp.map("/VMC/Ext/Bone/Pos", vmc_handler)
    try:
        server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", LISTEN_PORT), disp)
        print(f"🚀 连接成功！正在监听 VMC 软件数据 (端口 {LISTEN_PORT})")
        server.serve_forever()
    except:
        print(f"❌ 端口 {LISTEN_PORT} 被占用")

def main():
    threading.Thread(target=start_server, daemon=True).start()

    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=True, rigid_options=gs.options.RigidOptions(dt=0.01))
    
    # 地面
    scene.add_entity(gs.morphs.Plane())
    scene.build()
    
    print("\n✅ 系统就绪。")
    print("请确保 'Virtual Motion Capture' 软件已打开并开启了 OSC Sender。")

    while True:
        try: scene.clear_debug_objects()
        except: pass
        
        with lock:
            # 绘制所有骨骼点
            for name, pos in trackers.items():
                # 稍微抬高一点，防止穿模
                final_pos = pos + [0, 0, 0.05] 
                
                # 画点
                scene.draw_debug_sphere(pos=final_pos.tolist(), radius=0.05, color=BONE_COLORS[name])
            
            # 画连线 (简单的火柴人逻辑)
            lines = [
                ("Hips", "Spine"), ("Spine", "Chest"), ("Chest", "Head"),
                ("Chest", "LeftLowerArm"), ("LeftLowerArm", "LeftHand"), # 这里的LowerArm其实是Elbow位置
                ("Chest", "RightLowerArm"), ("RightLowerArm", "RightHand"),
                ("Hips", "LeftLowerLeg"), ("LeftLowerLeg", "LeftFoot"), # LowerLeg其实是Knee位置
                ("Hips", "RightLowerLeg"), ("RightLowerLeg", "RightFoot")
            ]
            
            for p1, p2 in lines:
                if p1 in trackers and p2 in trackers:
                    scene.draw_debug_line(start=trackers[p1].tolist(), end=trackers[p2].tolist(), radius=0.005, color=(1,1,1))

        scene.step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()