# coding=utf-8
import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from sensor_msgs.msg import CompressedImage
from pymycobot import MyCobot280RDKX5
import time
import cv2
import numpy as np
import threading
import transforms3d as tf3d
class MinimalSubscriber(Node):
    def __init__(self):
        self.mc=MyCobot280RDKX5("/dev/ttyS1",1000000)
        # self.mc.set_fresh_mode(1)
        self.photo_angles=[27.77, 29.79, -116.1, -9.49, 2.72, 165.76]
        self.mc.send_angles(self.photo_angles,50)
        self.wait_done()
        self.mc.set_end_type(0)
        self.mc.set_gripper_state(1,100)
        # self.mc.get_angles()
        print("ok")
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PerceptionTargets,
            '/ai_msg_mono2d_trash_detection',
            self.listener_callback,
            10)
        self.cameraMatrix =np.array( [[639.50350871,   0.    ,     311.3847461 ],
        [  0.     ,    638.50603424 ,219.9975647 ],
        [  0.        ,   0.        ,   1.        ]])

        self.distCoeffs = np.array([[ 1.47614748e-01 ,-1.20671013e+00 , 3.51376303e-03 ,-1.35507970e-03,
   2.57014341e+00]])
        self.cam_coords=None
        # while self.cam_coords is None:
        #     self.cam_coords=self.mc.get_coords()
        #     time.sleep(0.2)
        self.EyesInHand_matrix=np.array([[-0.6939603779673582, -0.7171871341589221, -0.06373074931549055, -7.2912624200794731],
        [0.7196847052210569, -0.6935946078435518, -0.031312059038481804, 43.800953790167206], 
        [-0.021746698192629678, -0.06759537385991699, 0.9974757874507313, -88.38631938563842], 
        [0.0, 0.0, 0.0, 1.0]])
        self.lock = threading.Lock()
        self.flag=0
        self.count=1
    def wait_done(self):
        time.sleep(2)
        while self.mc.is_moving()==1:
            time.sleep(1)

    def calculate_center(self,roi):
        
        x_center = roi.x_offset + roi.width / 2
        y_center = roi.y_offset + roi.height / 2
        return [x_center, y_center]
    
    def pixel_to_camera(self,u, v):
        points = np.array([u, v], dtype=np.float32)
        undistorted = cv2.undistortPoints(points, self.cameraMatrix,self.distCoeffs)
        x_norm = undistorted[0,0,0]
        y_norm = undistorted[0,0,1]
        cam_X = round(x_norm * 0.23*1000, 2)
        cam_Y = round(y_norm * 0.23*1000 , 2)  # 保留原代码的线性调整
        
        return [cam_X,cam_Y]
    

    def listener_callback(self, msg):
        
        for target in msg.targets:
            for rois in target.rois:   
                with self.lock:
                    if  self.count==1:
                        # self.get_logger().info(f'Value: "{rois.type}",rect:"{rois.rect}"')
                        result=self.calculate_center(rois.rect)
                        xy=self.pixel_to_camera(result[0],result[1])
                        self.mc.set_gripper_state(0,100)
                        while self.cam_coords is None:
                            self.cam_coords=self.mc.get_coords()
                            time.sleep(0.05)
                        x, y, z = self.cam_coords[:3]  
                        roll, pitch, yaw = self.cam_coords[3:]
                        R_matrix = tf3d.euler.euler2mat(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw), 'sxyz')         
                        T_tool_to_base = tf3d.affines.compose([x, y, z], R_matrix, [1, 1, 1])
                        target_homogeneous = np.array([xy[0],xy[1],209.79,1])

                        target_base=T_tool_to_base @ self.EyesInHand_matrix @ target_homogeneous
                        
                        for i in range(3):
                            self.cam_coords[i]=target_base[i]
                        
                        self.cam_coords[2]=150
                        self.mc.send_coords(self.cam_coords,50,1)
                        self.wait_done()
                        self.cam_coords[2]=110
                        self.mc.send_coords(self.cam_coords,50,1)
                        self.wait_done()
                        self.mc.set_gripper_state(1,100)
                        time.sleep(2)
                        self.cam_coords[2]=150
                        self.mc.send_coords(self.cam_coords,50,1)
                        self.wait_done()
                        self.mc.send_angles([-70,0,-90,0,0,136.21],50)
                        self.wait_done()
                        
                        self.mc.send_angles([-68.81, -12.65, -122.6, 38.32, 0.7, 136.66],50)
                        self.wait_done()
                        
                        self.mc.set_gripper_state(0,100)
                        self.wait_done()
                        
                        self.mc.send_angles([-70,0,-90,0,0,136.21],50)
                        self.wait_done()
                        self.mc.set_gripper_state(1,100)
                        time.sleep(2)
                        self.cam_coords=None
                        self.mc.send_angles(self.photo_angles,50)
                        self.wait_done()
                        

                    


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()