"""
SlimeVR Ultimate Debugger
Place this file in: examples/debug_slimevr.py
"""
import time
import socket
import pickle
import threading
import numpy as np
import torch
import genesis as gs

# 复用官方配置
import gs_env.sim.envs as gs_envs
from gs_env.sim.envs.config.registry import EnvArgsRegistry

# ================= 颜色配置 (RGB) =================
COLORS = {
    "head":       (1.0, 1.0, 0.0), # 黄
    "chest":      (1.0, 0.0, 1.0), # 洋红
    "pelvis":     (0.0, 1.0, 1.0), # 青 (核心)
    "left_hand":  (1.0, 0.0, 0.0), # 红
    "right_hand": (0.0, 1.0, 0.0), # 绿
    "left_foot":  (0.0, 0.0, 1.0), # 蓝
    "right_foot": (1.0, 1.0, 1.0)  # 白
}

# ================= SlimeVR 接收器 =================
class SlimeVRReceiver:
    def __init__(self, ip="127.0.0.1", port=8000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)
        self.latest_data = None
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()
        print(f"[SlimeVR] Listening on {ip}:{port}...")

    def _recv_loop(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(40960)
                if data:
                    cmd = pickle.loads(data)
                    with self.lock:
                        self.latest_data = cmd
            except BlockingIOError:
                time.sleep(0.0001)
            except Exception:
                pass

    def get_latest_poses(self):
        with self.lock:
            return self.latest_data

# ================= 辅助函数 =================
def to_list(arr):
    return np.array(arr).flatten().tolist()

def draw_pose(scene, pose_mat, color, label=""):
    """绘制一个带颜色的小球 + RGB 坐标轴"""
    if pose_mat is None: return
    
    pos = pose_mat[:3, 3]
    rot = pose_mat[:3, :3]
    
    # 1. 绘制中心球体 (半径 3cm)
    try:
        scene.draw_debug_sphere(
            pos=to_list(pos), 
            radius=0.03, 
            color=color
        )
    except: pass

    # 2. 绘制坐标轴 (长度 15cm)
    axis_len = 0.15
    # X轴 (红)
    scene.draw_debug_line(
        start=to_list(pos),
        end=to_list(pos + rot[:, 0] * axis_len),
        radius=0.005, color=(1, 0, 0)
    )
    # Y轴 (绿)
    scene.draw_debug_line(
        start=to_list(pos),
        end=to_list(pos + rot[:, 1] * axis_len),
        radius=0.005, color=(0, 1, 0)
    )
    # Z轴 (蓝)
    scene.draw_debug_line(
        start=to_list(pos),
        end=to_list(pos + rot[:, 2] * axis_len),
        radius=0.005, color=(0, 0, 1)
    )

# ================= 主逻辑 =================
def main():
    print("Initializing Debug Environment...")
    env_args = EnvArgsRegistry["g1_motion"]
    env = gs_envs.MotionEnv(
        args=env_args,
        num_envs=1,
        show_viewer=True,
        device=torch.device("cpu"), # 调试模式用 CPU 更稳定
        eval_mode=True,
    )

    receiver = SlimeVRReceiver(port=8000)
    
    print("\n" + "="*60)
    print("🎨 SLIMEVR VISUALIZER")
    print("   🟡 Head   🟣 Chest   🔵 Pelvis")
    print("   🔴 L-Hand 🟢 R-Hand")
    print("   🟦 L-Foot ⚪ R-Foot")
    print("="*60 + "\n")

    try:
        while True:
            # 1. 清除上一帧的绘图 (解决残影的关键!)
            # 注意: 如果你的 Genesis 版本太旧不支持此 API，请注释掉下面这行
            try:
                env.scene.clear_debug_objects()
            except AttributeError:
                pass # 旧版本可能没有这个方法

            # 2. 获取数据
            data = receiver.get_latest_poses()
            
            if data:
                # 3. 遍历所有部位并绘制
                for part_name, color in COLORS.items():
                    if part_name in data:
                        draw_pose(env.scene, data[part_name], color, part_name)
            
            # 4. 刷新环境 (保持 Viewer 活跃)
            dummy_action = torch.zeros((1, env.action_space.shape[0]), device=env.device)
            env.step(dummy_action)
            
            time.sleep(1/60)

    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()