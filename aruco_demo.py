# coding=utf-8
import cv2
import numpy as np
import stag 
from pymycobot import MyCobot280RDKX5
import json
import time
import transforms3d as tf3d
import Hobot.GPIO as GPIO

K=[[-65.47, -29.53, -71.45, 13.97, -0.35, -18.54],[-65.47, -31.46, -80.68, 26.45, -0.35, -18.54],[-65.65, -43.15, -89.64, 47.54, -0.17, -18.98]]
L=[[-80.06, -31.46, -66.35, 8.7, 1.31, -31.2],[-80.06, -32.78, -76.64, 21.7, 1.31, -31.2], [-80.06, -43.24, -86.48, 42.62, 1.31, -31.28]]

class Camera:
    def __init__(self,index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cameraMatrix =np.array( [[639.50350871,   0.    ,     311.3847461 ],
        [  0.     ,    638.50603424 ,219.9975647 ],
        [  0.        ,   0.        ,   1.        ]])

        self.distCoeffs = np.array([[ 1.47614748e-01 ,-1.20671013e+00 , 3.51376303e-03 ,-1.35507970e-03,
   2.57014341e+00]])
        self.mark_size=30
        self.aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params=cv2.aruco.DetectorParameters()
        if not self.cap.isOpened():
            print("no")
            exit()
        for i in range(3):
            ret, frame =self.cap.read()
            if not ret:
                print("no")
                break
    
    def get_frame(self):
        while True:
            for _ in range(3):
                self.cap.grab()
            ret, frame =self.cap.retrieve()
            if not ret:
                print("no")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict,parameters=self.aruco_params)
            if len(corners) > 0:
                for i in range(len(ids)):
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.mark_size, self.cameraMatrix, self.distCoeffs)
                    ids[i]=int(ids[i].item())
                    
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                euler_angles=  list(cv2.RQDecomp3x3(rotation_matrix)[0])
                xyz=list(tvec.flatten())
                for i in range(3):
                    euler_angles[i]=int(euler_angles[i])
                    xyz[i]=float(round(xyz[i],2))
                yaw=euler_angles[2]
                
                if yaw > 90:
                  yaw=180 -yaw    
                elif yaw < -90:
                  yaw= -180 - yaw   
                if yaw > 45:
                    yaw= 90 - yaw     
                elif yaw < -45:
                    yaw= -90 -yaw
                cv2.aruco.drawDetectedMarkers(frame, corners,ids)
                cv2.imshow("result",frame)
                cv2.waitKey(1500)
                cv2.destroyAllWindows()
                return xyz,yaw,ids[-1]
            else:
                cv2.imshow("result",frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                
            
class Robot(MyCobot280RDKX5):
    def __init__(self):
        super().__init__("/dev/ttyS1")
        self.speed=50
        self.photo_angles = [12.48, -8.78, -60.9, -22.85, -1.75, 59.76]
        self.tmp_angles=[-65.65, 20.47, -102.65, -15.29, 0.26, -16.17]
        self.flag1=1
        self.flag2=1
        self.EyesInHand_matrix=np.array([[-0.6939603779673582, -0.7171871341589221, -0.06373074931549055, -7.2912624200794731],
        [0.7196847052210569, -0.6935946078435518, -0.031312059038481804, 43.800953790167206], 
        [-0.021746698192629678, -0.06759537385991699, 0.9974757874507313, -88.38631938563842], 
        [0.0, 0.0, 0.0, 1.0]])
        self.send_angles(self.photo_angles,self.speed)
        self.wait_done()
        self.cam_coords=None
        self.set_tool_reference([0,0,100,0,0,0])
        self.set_end_type(1)
        
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        self.output_pin1 = 37
        self.output_pin2 = 31 
        GPIO.setup(self.output_pin1, GPIO.OUT)
        GPIO.output(self.output_pin1, GPIO.HIGH)
        GPIO.setup(self.output_pin2, GPIO.OUT)
        GPIO.output(self.output_pin2, GPIO.HIGH)

    def pump_off(self):
        GPIO.output(self.output_pin2, GPIO.LOW)
        time.sleep(0.05)
        GPIO.output(self.output_pin1, GPIO.HIGH)
        time.sleep(0.05)
    
    

    def pump_on(self):
        GPIO.output(self.output_pin2, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(self.output_pin1, GPIO.LOW)

    def wait_done(self):
        time.sleep(2)
        while self.is_moving()==1:
            time.sleep(1)
    def work(self,target_xyz,target_yaw,target_id):
        wait_angle=None
        self.send_angles(self.photo_angles,self.speed)
        self.wait_done()

        while self.cam_coords is None:
            self.cam_coords=self.get_coords()
            time.sleep(0.05)
        x, y, z = self.cam_coords[:3]  
        roll, pitch, yaw = self.cam_coords[3:]
        
    
        R_matrix = tf3d.euler.euler2mat(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw), 'sxyz')         
        T_tool_to_base = tf3d.affines.compose([x, y, z], R_matrix, [1, 1, 1])
        target_homogeneous = np.array([target_xyz[0], target_xyz[1],target_xyz[2],1])

        target_base=T_tool_to_base @ self.EyesInHand_matrix @ target_homogeneous
        for i in range(3):
            self.cam_coords[i]=target_base[i]
        height=self.cam_coords[2]
        self.cam_coords[2]=110
        self.send_coords(self.cam_coords,self.speed,1)
        self.wait_done()

        while wait_angle is None:
            wait_angle=self.get_angles()
        wait_angle[5]+=target_yaw
        self.send_angle(6,wait_angle[-1],100)
        self.wait_done()
        self.cam_coords[-1]=self.get_coords()[-1]
        self.cam_coords[2]=height
        self.send_coords(self.cam_coords,self.speed,1)
        self.wait_done()
        self.pump_on()
        time.sleep(2)
        self.cam_coords[2]=110
        self.send_coords(self.cam_coords,self.speed,1)
        self.wait_done()
        wait_angle[5]-=target_yaw
        self.send_angle(6,wait_angle[5],100)
        self.wait_done()
        self.send_angles(self.tmp_angles,self.speed)
        self.wait_done()
        if target_id==1 or target_id ==2:
           
            if self.flag1 % 2 == 0:
                self.send_angles(K[0],self.speed)
                self.wait_done()
                self.send_angles(K[1],self.speed)
                self.wait_done() 
                self.pump_off()
                time.sleep(2)
                self.send_angles(K[0],self.speed)
                self.wait_done()
            else:
                self.send_angles(K[0],self.speed)
                self.wait_done()
                self.send_angles(K[2],self.speed)
                self.wait_done() 
                self.pump_off()
                time.sleep(2)
                self.send_angles(K[0],self.speed)
                self.wait_done()
            self.flag1+=1

        elif target_id==3 or target_id==4:
           
            if self.flag2 % 2 == 0:
                self.send_angles(L[0],self.speed)
                self.wait_done()
                self.send_angles(L[1],self.speed)
                self.wait_done() 
                self.pump_off()
                time.sleep(2)
                self.send_angles(L[0],self.speed)
                self.wait_done()
            else:
                self.send_angles(L[0],self.speed)
                self.wait_done()
                self.send_angles(L[2],self.speed)
                self.wait_done() 
                self.pump_off()
                time.sleep(2)
                self.send_angles(L[0],self.speed)
                self.wait_done()
            self.flag2+=1
        self.cam_coords=None
        self.send_angles(self.photo_angles,self.speed)
        self.wait_done()



if __name__=="__main__":
    camera=Camera()
    robot=Robot()
    while 1:
        xyz,yaw,id=camera.get_frame()
        print(id)
        if len(xyz)>0:
            robot.work(xyz,yaw,id)
    





        

