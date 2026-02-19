"""
SlimeVR VMC Publisher (Final Version)
使用 VMC 协议直接驱动 Genesis 机器人，无需配置 ID。
"""
import time
import socket
import pickle
import numpy as np
import threading
from pythonosc import dispatcher, osc_server
from scipy.spatial.transform import Rotation as R

# === 配置 ===
# VMC 监听端口 (SlimeVR 发出的端口)
VMC_IP = "127.0.0.1"
VMC_PORT = 39539 

# Genesis 接收端口
TARGET_IP = "127.0.0.1"
TARGET_PORT = 8000

class VMCPublisher:
    def __init__(self):
        # UDP 发送器 (给 Genesis)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (TARGET_IP, TARGET_PORT)
        
        # 数据容器
        self.trackers = {}
        # 我们需要关注的骨骼名称映射 (VMC标准名 -> Genesis代码用名)
        self.bone_map = {
            "Head": "head",
            "Hips": "pelvis",
            "LeftHand": "left_hand",
            "RightHand": "right_hand",
            "LeftFoot": "left_foot",
            "RightFoot": "right_foot"
        }
        
        # 启动 OSC 服务器 (接收 SlimeVR)
        self.disp = dispatcher.Dispatcher()
        self.disp.map("/VMC/Ext/Bone/Pos", self.vmc_handler)
        
        try:
            self.server = osc_server.ThreadingOSCUDPServer((VMC_IP, VMC_PORT), self.disp)
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            print(f"[VMC Bridge] Listening on port {VMC_PORT}...")
            print(f"[VMC Bridge] Forwarding to Genesis on port {TARGET_PORT}...")
        except OSError:
            print(f"❌ 端口 {VMC_PORT} 被占用！请关闭其他 VMC 测试脚本。")

    def vmc_handler(self, address, *args):
        # VMC 格式: (BoneName, x, y, z, qx, qy, qz, qw)
        try:
            bone_name = args[0]
            if bone_name not in self.bone_map:
                return

            target_name = self.bone_map[bone_name]
            
            # 1. 提取位置 (Unity 坐标系: X右, Y上, Z前)
            pos = np.array([args[1], args[2], args[3]])
            
            # 2. 提取旋转 (四元数)
            quat = np.array([args[4], args[5], args[6], args[7]])
            
            # === 坐标系转换 ===
            # Unity (Y-up) -> Genesis (Z-up)
            # 这里的转换需要根据实际观感微调，通常如下：
            # Gen_X = Unity_Z
            # Gen_Y = -Unity_X
            # Gen_Z = Unity_Y
            
            gen_pos = np.array([pos[2], -pos[0], pos[1]])
            
            # 旋转转换 (Unity -> ROS/Genesis)
            # 先转成 Scipy 旋转对象
            r = R.from_quat(quat)
            # 这一步是最难的，通常需要一个 基准旋转 来对齐
            # 暂时直接透传，如果方向不对，我们在 Genesis 脚本里调
            gen_rot = r.as_quat()

            self.trackers[target_name] = {
                "pos": gen_pos,
                "rot": gen_rot
            }
            
        except Exception as e:
            pass

    def get_matrix(self, part_name):
        """构造 4x4 矩阵"""
        if part_name not in self.trackers:
            return np.eye(4) # 没数据时返回单位矩阵
            
        t = self.trackers[part_name]
        pos = t["pos"]
        rot = t["rot"]
        
        mat = np.eye(4)
        mat[:3, 3] = pos
        mat[:3, :3] = R.from_quat(rot).as_matrix()
        return mat

    def run(self):
        print("🚀 Bridge Running. Load a VRM in SlimeVR to start sending positions!")
        while True:
            try:
                # 检查是否有有效数据 (髋部高度 > 0.1m)
                if "pelvis" in self.trackers and self.trackers["pelvis"]["pos"][2] < 0.1:
                    # 如果位置是 0，说明还没加载 VRM
                    pass 
                
                data = {
                    "head":       self.get_matrix("head"),
                    "pelvis":     self.get_matrix("pelvis"),
                    "left_hand":  self.get_matrix("left_hand"),
                    "right_hand": self.get_matrix("right_hand"),
                    "left_foot":  self.get_matrix("left_foot"),
                    "right_foot": self.get_matrix("right_foot")
                }
                
                payload = pickle.dumps(data)
                self.sock.sendto(payload, self.dest)
                time.sleep(1/60)
                
            except KeyboardInterrupt:
                print("\nStopping...")
                break

if __name__ == "__main__":
    VMCPublisher().run()