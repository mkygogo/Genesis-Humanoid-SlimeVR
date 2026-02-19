"""
SlimeVR VMC Data Auditor (终极版)
用于确认 SlimeVR 发送的所有骨骼名称及其数据活跃度。
"""
import time
import numpy as np
from pythonosc import dispatcher, osc_server
import threading

# 监听端口 (必须与 SlimeVR 设置一致)
LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 39539

# 存储骨骼状态: {name: {'pos': [...], 'rot': [...], 'last_update': time}}
bones = {}
lock = threading.Lock()

def vmc_handler(address, *args):
    # VMC 格式: /VMC/Ext/Bone/Pos (Name, x, y, z, qx, qy, qz, qw)
    try:
        bone_name = args[0]
        pos = np.array(args[1:4])
        rot = np.array(args[4:8])
        
        with lock:
            if bone_name not in bones:
                bones[bone_name] = {
                    'pos': pos, 'rot': rot, 'last_update': time.time(),
                    'pos_changed': False, 'rot_changed': False
                }
            else:
                # 检查数据是否变化 (判断活跃度)
                prev = bones[bone_name]
                if np.linalg.norm(pos - prev['pos']) > 0.001: prev['pos_changed'] = True
                if np.linalg.norm(rot - prev['rot']) > 0.001: prev['rot_changed'] = True
                
                prev['pos'] = pos
                prev['rot'] = rot
                prev['last_update'] = time.time()

    except Exception:
        pass

def print_dashboard():
    print(f"🎧 正在监听 VMC 端口 {LISTEN_PORT} ...")
    print(f"👉 请活动全身，特别是手脚！")
    
    while True:
        time.sleep(0.5) # 每 0.5 秒刷新一次
        
        print("\033[2J\033[H", end="") # 清屏
        print("="*80)
        print(f"📡 VMC 骨骼数据审计 (Ctrl+C 退出)")
        print(f"{'骨骼名称 (Bone Name)':<20} | {'位置 (Pos)':<25} | {'旋转 (Rot)':<25} | {'状态'}")
        print("-" * 80)
        
        with lock:
            # 按名称排序，方便查找
            sorted_bones = sorted(bones.keys())
            
            if not sorted_bones:
                print("⏳ 等待数据中... (如果一直没数据，请检查 SlimeVR 设置 -> VMC -> 端口)")
            
            for name in sorted_bones:
                data = bones[name]
                
                # 格式化数据字符串
                p = data['pos']
                pos_str = f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
                
                # 状态检查
                is_active = (time.time() - data['last_update']) < 1.0
                activity = "🟢 活跃" if is_active else "🔴 离线"
                
                # 变化标记
                p_mark = "⚡" if data['pos_changed'] else "  "
                r_mark = "⚡" if data['rot_changed'] else "  "
                
                # 重置变化标记 (以便下一帧检测新的变化)
                data['pos_changed'] = False
                data['rot_changed'] = False
                
                # 高亮关键部位 (根据你的 8 个 tracker)
                # SlimeVR VMC 标准名称通常是:
                # Hips, Spine, Chest, Head, 
                # LeftShoulder, LeftUpperArm, LeftLowerArm, LeftHand
                # RightShoulder, RightUpperArm, RightLowerArm, RightHand
                # LeftUpperLeg, LeftLowerLeg, LeftFoot, LeftToes
                # RightUpperLeg, RightLowerLeg, RightFoot, RightToes
                
                is_key = name in ["Hips", "Chest", "Spine", "Head", 
                                  "LeftHand", "RightHand", 
                                  "LeftLowerLeg", "RightLowerLeg", "LeftFoot", "RightFoot"]
                
                prefix = ">> " if is_key else "   "
                
                print(f"{prefix}{name:<17} | {pos_str} {p_mark} | 旋转数据 {r_mark}             | {activity}")

        print("="*80)

if __name__ == "__main__":
    try:
        # 启动 OSC
        disp = dispatcher.Dispatcher()
        disp.map("/VMC/Ext/Bone/Pos", vmc_handler)
        
        server = osc_server.ThreadingOSCUDPServer((LISTEN_IP, LISTEN_PORT), disp)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        
        print_dashboard()
    except KeyboardInterrupt:
        print("\n退出。")
    except OSError:
        print(f"❌ 端口 {LISTEN_PORT} 被占用！请关闭其他 Python 脚本。")