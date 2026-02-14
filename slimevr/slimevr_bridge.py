import argparse
import threading
import time
import numpy as np
from pythonosc import dispatcher, osc_server
from scipy.spatial.transform import Rotation as R

class SlimeVRBridge:
    def __init__(self, ip="127.0.0.1", port=9000, scale=1.0):
        """
        :param scale: 缩放比例，如果SlimeVR单位是米，Genesis也是米，则为1.0
        """
        self.ip = ip
        self.port = port
        self.scale = scale
        
        # 存储原始 SlimeVR 数据 (pos: [x,y,z], rot: [x,y,z,w])
        # 对应你提到的8个部位
        self.raw_data = {
            "chest": {"pos": np.zeros(3), "rot": np.array([0,0,0,1.0])},
            "waist": {"pos": np.zeros(3), "rot": np.array([0,0,0,1.0])},
            "left_hand": {"pos": np.zeros(3), "rot": np.array([0,0,0,1.0])},
            "right_hand": {"pos": np.zeros(3), "rot": np.array([0,0,0,1.0])},
            "left_thigh": {"pos": np.zeros(3), "rot": np.array([0,0,0,1.0])}, # 辅助计算
            "right_thigh": {"pos": np.zeros(3), "rot": np.array([0,0,0,1.0])}, # 辅助计算
            "left_calf": {"pos": np.zeros(3), "rot": np.array([0,0,0,1.0])},
            "right_calf": {"pos": np.zeros(3), "rot": np.array([0,0,0,1.0])},
        }

        # OSC ID 映射表 (请根据你的 SlimeVR Server 显示的 ID 修改这里!)
        # 这里只是示例，必须改为你实际的 Tracker ID
        self.id_map = {
            "1": "chest",      # 示例: Tracker 1 是胸
            "2": "waist",      # 示例: Tracker 2 是腰
            "3": "left_hand", 
            "4": "right_hand",
            "5": "left_thigh",
            "6": "right_thigh",
            "7": "left_calf",
            "8": "right_calf"
        }

        # 启动 OSC 服务
        self._start_server()
        print(f"[SlimeVR] Listening on {ip}:{port}...")

    def _start_server(self):
        disp = dispatcher.Dispatcher()
        disp.map("/tracking/trackers/*", self._handle_tracker)
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), disp)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def _handle_tracker(self, address, *args):
        # 解析 OSC 地址: /tracking/trackers/{ID}/position 或 rotation
        try:
            parts = address.split('/')
            t_id = parts[-2]
            data_type = parts[-1]
            
            if t_id in self.id_map:
                body_part = self.id_map[t_id]
                if data_type == "position":
                    # SlimeVR (Y-up) -> Genesis (Z-up) 转换
                    # SlimeVR: x=right, y=up, z=forward
                    # Genesis: x=forward, y=left, z=up (假设)
                    # 这是一个常见的转换矩阵，如果方向不对需调整
                    x, y, z = args
                    # 简单映射: Y(up) -> Z(up), Z(fwd) -> X(fwd), X(right) -> -Y(left)
                    self.raw_data[body_part]["pos"] = np.array([-x, -z, y]) * self.scale
                    
                elif data_type == "rotation":
                    x, y, z, w = args
                    # 四元数旋转坐标系转换略复杂，这里先透传，视具体表现调整
                    # 建议在 SlimeVR Server 里直接重置朝向
                    self.raw_data[body_part]["rot"] = np.array([x, y, z, w])
        except Exception as e:
            pass

    def get_teleop_action(self):
        """
        返回 ExtremControl 论文需要的 6 个关键点 Pose (4x4 Matrix)
        格式: {
            "head": 4x4,
            "pelvis": 4x4,
            "left_hand": 4x4, "right_hand": 4x4,
            "left_foot": 4x4, "right_foot": 4x4
        }
        """
        actions = {}
        
        # 1. Pelvis (直接使用 Waist)
        actions["pelvis"] = self._to_mat(self.raw_data["waist"])
        
        # 2. Head (使用 Chest + Offset)
        # 假设头在胸上方 25cm
        head_data = self.raw_data["chest"].copy()
        head_data["pos"] += np.array([0, 0, 0.25]) 
        actions["head"] = self._to_mat(head_data)
        
        # 3. Hands (直接映射)
        actions["left_hand"] = self._to_mat(self.raw_data["left_hand"])
        actions["right_hand"] = self._to_mat(self.raw_data["right_hand"])
        
        # 4. Feet (使用 Calf + Offset)
        # 假设脚在小腿下方 30cm (具体取决于你绑在脚踝还是小腿肚)
        l_foot_data = self.raw_data["left_calf"].copy()
        l_foot_data["pos"] += np.array([0, 0, -0.30])
        actions["left_foot"] = self._to_mat(l_foot_data)
        
        r_foot_data = self.raw_data["right_calf"].copy()
        r_foot_data["pos"] += np.array([0, 0, -0.30])
        actions["right_foot"] = self._to_mat(r_foot_data)
        
        return actions

    def _to_mat(self, data):
        # 将 pos, rot 转换为 4x4 齐次矩阵
        mat = np.eye(4)
        mat[:3, 3] = data["pos"]
        r = R.from_quat(data["rot"])
        mat[:3, :3] = r.as_matrix()
        # 修正旋转坐标系 (Slime Y-up -> Genesis Z-up)
        # 这里可能需要根据实际观察添加额外的旋转矩阵
        return mat

# 测试代码
if __name__ == "__main__":
    bridge = SlimeVRBridge()
    while True:
        action = bridge.get_teleop_action()
        print(f"Pelvis Z: {action['pelvis'][2,3]:.2f}")
        time.sleep(0.1)