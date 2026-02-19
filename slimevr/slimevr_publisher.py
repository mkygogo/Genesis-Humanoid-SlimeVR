import time
import socket
import pickle
import numpy as np
import threading
from pythonosc import dispatcher, osc_server
from scipy.spatial.transform import Rotation as R

# === 网络配置 ===
SLIME_IP = "127.0.0.1"
SLIME_PORT = 9000
TARGET_IP = "127.0.0.1"
TARGET_PORT = 8000

# === 追踪器 ID 映射 ===
# 根据你的日志和描述设定
TRACKER_MAP = {
    "head": "head",      # SlimeVR 头部
    "1": "chest",        # 胸
    "2": "pelvis",       # 腰/骨盆 (Waist)
    "3": "left_hand",    # 左手
    "4": "right_hand",   # 右手
    "7": "left_foot",    # 左脚/踝 (直接用Ankle驱动脚)
    "8": "right_foot"    # 右脚/踝
}

class SlimeVRPublisher:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (TARGET_IP, TARGET_PORT)
        
        # 初始化数据容器
        # 默认位置设为 T-pose 附近的合理值，避免无数据时飞掉
        self.trackers = {
            "head":       {"pos": np.array([0, 0, 1.7]),   "rot": np.array([0,0,0,1.])},
            "chest":      {"pos": np.array([0, 0, 1.4]),   "rot": np.array([0,0,0,1.])},
            "pelvis":     {"pos": np.array([0, 0, 0.8]),   "rot": np.array([0,0,0,1.])},
            "left_hand":  {"pos": np.array([0, 0.3, 1.0]), "rot": np.array([0,0,0,1.])},
            "right_hand": {"pos": np.array([0,-0.3, 1.0]), "rot": np.array([0,0,0,1.])},
            "left_foot":  {"pos": np.array([0, 0.1, 0.1]), "rot": np.array([0,0,0,1.])},
            "right_foot": {"pos": np.array([0,-0.1, 0.1]), "rot": np.array([0,0,0,1.])}
        }
        
        # 启动 OSC 监听
        disp = dispatcher.Dispatcher()
        disp.map("/tracking/trackers/*", self.osc_handler)
        server = osc_server.ThreadingOSCUDPServer((SLIME_IP, SLIME_PORT), disp)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[SlimeVR] Bridge Started! Receiving data from {SLIME_PORT}...")
        print(f"[SlimeVR] Sending commands to {TARGET_PORT}...")

    def osc_handler(self, address, *args):
        try:
            parts = address.split('/')
            # URL 格式通常为 /tracking/trackers/{ID}/position
            t_id = parts[-2] 
            mode = parts[-1] # position 或 rotation
            
            # 将 SlimeVR 的 ID 转换为我们要的身体部位名称
            body_part = TRACKER_MAP.get(t_id)
            
            if body_part:
                if mode == "position":
                    x, y, z = args
                    # === 坐标系转换 (核心) ===
                    # SlimeVR: X=右, Y=上, Z=后 (或前，取决于校准), 原点=头
                    # Genesis: X=前, Y=左, Z=上, 原点=地面
                    # 转换公式：
                    # Gen_X = -Slime_Z (假设Slime Z是前)
                    # Gen_Y = -Slime_X (X是右，所以负X是左)
                    # Gen_Z = Slime_Y + Offset (Slime Y是相对于头的高度，比如-1.7米)
                    
                    height_offset = 1.70 # 将头坐标(0)抬高到1.7米
                    
                    new_pos = np.array([-z, -x, y + height_offset])
                    self.trackers[body_part]["pos"] = new_pos
                    
                elif mode == "rotation":
                    # SlimeVR 发送的是欧拉角 (度数)
                    if len(args) == 3: 
                        # 将欧拉角转为四元数 (x, y, z, w)
                        r = R.from_euler('xyz', args, degrees=True)
                        self.trackers[body_part]["rot"] = r.as_quat()
                    # 也可以处理四元数
                    elif len(args) == 4:
                        self.trackers[body_part]["rot"] = np.array(args)
        except Exception:
            pass

    def get_matrix(self, part_name):
        """生成 4x4 变换矩阵"""
        data = self.trackers.get(part_name)
        if data is None:
            return np.eye(4)
            
        mat = np.eye(4)
        mat[:3, 3] = data["pos"]
        
        r = R.from_quat(data["rot"])
        # 旋转修正：SlimeVR 的旋转通常需要绕 X 轴转 -90 度才能对齐机器人的 Z-up
        # 这一步可能需要根据你的实际观感微调
        # align_rot = R.from_euler('x', -90, degrees=True)
        # mat[:3, :3] = (r * align_rot).as_matrix()
        mat[:3, :3] = r.as_matrix()
        
        return mat

    def run(self):
        last_print = time.time()
        while True:
            try:
                # 构造 ExtremControl 需要的数据包
                # 键名必须与 run_slimevr_body.py 里的解析代码匹配
                data = {
                    "head":       self.get_matrix("head"),
                    "pelvis":     self.get_matrix("pelvis"),
                    "left_hand":  self.get_matrix("left_hand"),
                    "right_hand": self.get_matrix("right_hand"),
                    "left_foot":  self.get_matrix("left_foot"),
                    "right_foot": self.get_matrix("right_foot")
                }
                
                # 发送数据
                payload = pickle.dumps(data)
                self.sock.sendto(payload, self.dest)
                
                # 每秒打印一次调试信息
                if time.time() - last_print > 1.0:
                    lh = self.trackers['left_hand']['pos']
                    pel = self.trackers['pelvis']['pos']
                    print(f"\r[Debug] Pelvis Z: {pel[2]:.2f}m | L_Hand: {lh} | FPS: 60", end="")
                    last_print = time.time()
                    
                time.sleep(1/60) # 60Hz
            except KeyboardInterrupt:
                print("\nStopped.")
                break

if __name__ == "__main__":
    SlimeVRPublisher().run()