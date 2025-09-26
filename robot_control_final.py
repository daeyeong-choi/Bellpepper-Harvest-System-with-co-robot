import os
import time
import sys
import copy
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init

from od_msg.srv import SrvDepthPosition
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from robot_control.onrobot import RG

package_path = get_package_share_directory("pick_and_place_voice")

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 50, 60
# BUCKET_POS = [445.5, -242.6, 174.4, 156.4, 180.0, -112.5]
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"
DEPTH_OFFSET = -5.0
MIN_DEPTH = 2.0


DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("robot_control_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

########### Gripper Setup. Do not modify this area ############

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


########### Robot Controller ############


class RobotController(Node):
    def __init__(self):
        super().__init__("pick_and_place")
        self.init_robot()

        self.get_position_client = self.create_client(
            SrvDepthPosition, "/get_3d_position"
        )
        while not self.get_position_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for get_depth_position service...")
        self.get_position_request = SrvDepthPosition.Request()

        self.get_keyword_client = self.create_client(Trigger, "/get_keyword")
        while not self.get_keyword_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for get_keyword service...")
        self.get_keyword_request = Trigger.Request()
        self.position_map = {
            'pos1':[345.341, -212.003, 191.381, 120.138, -144.982,-149.638], # 양품 바구니
            "pos2":[575.341, -212.003, 191.381, 120.138, -144.982,-149.638], # 불량품 바구니
        }

        self.scan_section = [
            [613.457, 73.016, 429.452, 3.264, 104.707, 87.343],
            [626.437, -116.906, 441.222, 166.49, -102.713, -92.39],
            [590.615, -101.425, 292.951, 169.561, -110.312, -91.798],
            [615.752, 55.725, 261.135, 3.757, 105.51, 90.992]
        ]
        self.section_num = 0
        self.pimang = 0
        self.rotten_pimang = 0
        self.state = 'normal'


    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        """
        Converts 3D coordinates from the camera coordinate system
        to the robot's base coordinate system.
        """
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)  # Homogeneous coordinate

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)

        # 좌표 변환 (그리퍼 → 베이스)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]
    
    def move_target(self, target_pos):
        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()
    
    def pick_to_put_place(self, put_pos_n,put_pos_r):
        basket_up = [470.866,-98.914,374.232,166.684,-102.157,-92.518]
        normal_basket = put_pos_n # 양품 놓을 위치
        rotten_basket = put_pos_r # 썩은 놈 놓을 위치

        target_pos_rotten = self.get_target_pos('rotten')
        if target_pos_rotten is None:
            target_pos_stem = self.get_target_pos('stem')
            target_pos_stem[0] += 5
            self.move_target(target_pos_stem)
            mwait()
            gripper.close_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.5)
            mwait()
            movel(basket_up, vel=VELOCITY, acc=ACC)
            mwait()
            movel(normal_basket, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.open_gripper()
            mwait()
            self.pimang += 1
        else:
            target_pos_stem = self.get_target_pos('stem')
            target_pos_stem[0] += 5
            self.move_target(target_pos_stem)
            mwait()
            gripper.close_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.5)
            mwait()
            movel(basket_up, vel=VELOCITY, acc=ACC)
            mwait()
            movel(rotten_basket, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.open_gripper()
            mwait()
            self.rotten_pimang += 1

    def pick_to_put_place_r(self, put_pos_r):
        up_basket = [470.866,-98.914,374.232,166.684,-102.157,-92.518]
        rotten_basket = put_pos_r
        target_pos_stem = self.get_target_pos('stem')
        target_pos_stem[0] += 5
        self.move_target(target_pos_stem)
        mwait()
        gripper.close_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)
        mwait()
        movel([-70,0,0,0,0,0], vel=VELOCITY, acc = ACC, mod=1)
        mwait()
        movel(up_basket, vel=VELOCITY, acc = ACC)
        mwait()
        movel(rotten_basket, vel=VELOCITY, acc=ACC)
        mwait()
        gripper.open_gripper()
        mwait()
        self.rotten_pimang += 1
    
    def pick_to_put_place_n(self, put_pos_n):
        up_basket = [470.866,-98.914,374.232,166.684,-102.157,-92.518]
        normal_basket = put_pos_n
        target_pos_stem = self.get_target_pos('stem')
        target_pos_stem[0] += 5
        self.move_target(target_pos_stem)
        mwait()
        gripper.close_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)
        mwait()
        movel([-70,0,0,0,0,0], vel=VELOCITY, acc = ACC, mod=1)
        mwait()
        movel(up_basket, vel=VELOCITY, acc = ACC)
        mwait()
        movel(normal_basket, vel=VELOCITY, acc=ACC)
        mwait()
        gripper.open_gripper()
        mwait()
        self.pimang += 1


    def sterilize(self):
        dif_pos_z = [0,0,90,0,90,0] 
        gripper.close_gripper()
        movej(dif_pos_z, vel=VELOCITY, acc=ACC)
        mwait()
        movel([0,0,-120,0,0,0], vel=VELOCITY, acc=ACC, mod=1)
        movej(dif_pos_z, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()

    def app_target(self, target):
        target[0] -= 100
        target[2] += 30
        return target

    def robot_control(self):
        target_list = []
        self.get_logger().info("call get_keyword service")
        self.get_logger().info("say 'Hello Rokey' and speak what you want to pick up")
        get_keyword_future = self.get_keyword_client.call_async(self.get_keyword_request)
        rclpy.spin_until_future_complete(self, get_keyword_future)
        if get_keyword_future.result().success:
            get_keyword_result = get_keyword_future.result()
            message = get_keyword_result.message    
            if "/" in message:
                object_str, num_str, target_str = message.split("/") ##num_str
                object_list = object_str.strip().split()  # 도구 리스트
                num_list = num_str.strip().split()
                target_list = target_str.strip().split()  # 목적지 리스트

            else:
                object_list = []
                target_list = []
                num_list = []
                print("도구 리스트:", object_list)
                print("숫자 리스트:", num_list)
                print("목적지 리스트:", target_list)
###########################다 따기############################################
            if object_list[0] == 'pimang'  and len(target_list) == 2:
                while self.section_num < 5:

                    if self.state == 'failed': 
                        self.section_num += 1
                        self.state = 'normal'
                        if self.section_num >= 4:
                            self.get_logger().warn("All sections scanned. End program.")
                            break

                    mwait()
                    self.get_logger().warn("change section")
                    movel(self.scan_section[self.section_num], vel=VELOCITY, acc=ACC)

                    target_pos = self.get_target_pos('pimang')
                    if target_pos is None:
                        self.get_logger().warn("rotten pimang dection")
                        target_pos_rotten = self.get_target_pos('rotten')
                        if target_pos_rotten is None:
                            self.state = 'failed'
                            continue 

                        else:                   
                            target_pos_rotten=self.app_target(target_pos_rotten)
                            self.move_target(target_pos_rotten)
                            mwait()
                            self.pick_to_put_place_r(self.position_map[target_list[1]])
                            self.sterilize()
                            continue
        
                    else:
                        target_pos=self.app_target(target_pos)
                        self.get_logger().info(f"target position: {target_pos}")
                        self.move_target(target_pos)
                        mwait()
                        self.pick_to_put_place(self.position_map[target_list[0]],self.position_map[target_list[1]])
                        self.sterilize()
                            
                            
######################병든 것만 따기####################################################################                        
            if object_list[0] == 'rotten'  and target_list[1] == 'pos2':
                while self.section_num < 5:

                    if self.state == 'failed': 
                        self.section_num += 1
                        self.state = 'normal'
                        if self.section_num >= 4:
                            self.get_logger().warn("All sections scanned. End program.")
                            break
                    mwait()
                    movel(self.scan_section[self.section_num], vel=VELOCITY, acc=ACC)

                    target_pos = self.get_target_pos('rotten')
                    if target_pos is None:
                        self.state = 'failed'
                        continue 

                    else:                   
                        target_pos_rotten=self.app_target(target_pos_rotten)
                        self.move_target(target_pos_rotten)
                        mwait()
                        self.pick_to_put_place_r(self.position_map[target_list[1]])
                        self.sterilize()
                        continue

##################양품 지정 개수 따기##############################################################미구현
            if object_list[0] == 'pimang'  and len(num_list) > 0  and target_list[1] == 'pos1':
                while self.section_num < 5:
                    goal = num_list[0]

                    if self.state == 'failed': 
                        self.section_num += 1
                        self.state = 'normal'
                        if self.section_num >= 4:
                            self.get_logger().warn("All sections scanned. End program.")
                            break

                    self.init_robot()
                    mwait()
                    movel(self.scan_section[self.section_num], vel=VELOCITY, acc=ACC)

                    target_pos = self.get_target_pos('rotten')
                    if target_pos is None:
                        target_pos_rotten=self.app_target(target_pos_rotten)
                        self.move_target(target_pos_rotten)
                        mwait()
                        self.pick_to_put_place_r(self.position_map[target_list[1]])
                        self.sterilize()
                        if self.pimang > goal:
                            break
                        continue 

                    else:                   
                        self.state = 'failed'
                        continue                    
                        
            self.init_robot()
            self.get_logger().warn(f"Done")
            self.get_logger().warn(f"pimang:{self.pimang}, rotten pimang:{self.rotten_pimang}")
            return       
        else:
            self.get_logger().warn(f"{get_keyword_result.message}")
            return
        
    def get_target_pos(self, target):
        self.get_position_request.target = target
        self.get_logger().info("call depth position service with s_detection node")
        get_position_future = self.get_position_client.call_async(
            self.get_position_request
        )
        rclpy.spin_until_future_complete(self, get_position_future)

        if get_position_future.result():
            result = get_position_future.result().depth_position.tolist()
            self.get_logger().info(f"Received depth position: {result}")
            if sum(result) == 0:
                print("No target position")
                return None

            gripper2cam_path = os.path.join(
                package_path, "resource", "T_gripper2camera.npy"
            )
            robot_posx = get_current_posx()[0]
            td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)

            if td_coord[2] and sum(td_coord) != 0:
                td_coord[2] += DEPTH_OFFSET  # DEPTH_OFFSET
                td_coord[2] = max(td_coord[2], MIN_DEPTH)  # MIN_DEPTH: float = 2.0

            target_pos = list(td_coord[:3]) + robot_posx[3:]
        return target_pos

    def init_robot(self):
        JReady = [2.296, -29.437, 119.969, -25.004, 6.776, 116.816]
        movej(JReady, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()



def main(args=None):
    node = RobotController()
    while rclpy.ok():
        node.robot_control()
    rclpy.shutdown()
    node.destroy_node()


if __name__ == "__main__":
    main()



