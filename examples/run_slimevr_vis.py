"""
SlimeVR Visualizer based on view_motion.py
Place this file in: examples/run_slimevr_vis.py
"""
import time
import socket
import pickle
import threading
import numpy as np
import torch
import genesis as gs

# 复用官方的 import 路径
import gs_env.sim.envs as gs_envs
from gs_env.sim.envs.config.registry import EnvArgsRegistry

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
                time.sleep(0.001)
            except Exception:
                pass

    def get_latest_poses(self):
        with self.lock:
            return self.latest_data

# ================= 主逻辑 =================
def main():
    # 1. 获取 G1 的标准配置
    env_args = EnvArgsRegistry["g1_motion"]
    
    # 2. 初始化环境
    print("Initializing Genesis Environment...")
    env = gs_envs.MotionEnv(
        args=env_args,
        num_envs=1,
        show_viewer=True,
        device=torch.device("cpu"),
        eval_mode=True,
    )

    receiver = SlimeVRReceiver(port=8000)
    
    print("\n" + "="*50)
    print("🚀 SlimeVR Visualization Running!")
    print("Move your trackers. You should see RGB Axes moving in the scene.")
    print("==================================================" + "\n")

    try:
        while True:
            # 4.1 获取 SlimeVR 数据
            slime_data = receiver.get_latest_poses()
            
            # 4.2 绘图
            if slime_data:
                # 尝试清除上一帧 (注意：Genesis 目前没有公开的 clear_debug_lines API，
                # 如果线太多导致卡顿，可能需要修改底层或减少绘图频率)
                # env.scene.clear_debug_items() 
                
                for part_name, pose_mat in slime_data.items():
                    if isinstance(pose_mat, np.ndarray) and pose_mat.shape == (4, 4):
                        # 提取数据并转为标准 Python List/Float
                        pos = pose_mat[:3, 3]
                        rot_mat = pose_mat[:3, :3]
                        
                        start = pos
                        # 坐标轴长度 0.2m
                        end_x = pos + rot_mat[:, 0] * 0.2
                        end_y = pos + rot_mat[:, 1] * 0.2
                        end_z = pos + rot_mat[:, 2] * 0.2
                        
                        # === 关键修复：强制转换数据类型 ===
                        def clean(arr):
                            return np.array(arr, dtype=np.float32).flatten().tolist()

                        try:
                            # 绘制 RGB 坐标轴
                            env.scene.draw_debug_line(clean(start), clean(end_x), (1.0, 0.0, 0.0)) # Red
                            env.scene.draw_debug_line(clean(start), clean(end_y), (0.0, 1.0, 0.0)) # Green
                            env.scene.draw_debug_line(clean(start), clean(end_z), (0.0, 0.0, 1.0)) # Blue
                        except Exception as e:
                            # 捕获偶尔的绘图错误，防止程序崩退
                            pass
            
            # 4.3 刷新
            dummy_action = torch.zeros((1, env.action_space.shape[0]), device=env.device)
            env.step(dummy_action)
            
            time.sleep(1/60)

    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()