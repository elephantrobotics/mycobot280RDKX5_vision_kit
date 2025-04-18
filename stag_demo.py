# coding=utf-8
import cv2
import numpy as np
import stag 
from pymycobot import MyCobot280RDKX5
import json
import time
import transforms3d as tf3d
import Hobot.GPIO as GPIO

A=[[56.51, -50.97, -54.84, 17.66, -1.75, 102.83],[56.33, -54.66, -60.2, 28.12, -1.84, 102.83]]
B=[[-29.17, -53.43, -49.83, 15.99, 2.28, 16.87], [-29.17, -58.53, -55.37, 28.12, 2.28, 17.22]]
C=[[74.0, -24.78, -102.12, 38.05, -3.6, 120.14],[74.35, -33.57, -106.78, 52.29, -4.04, 120.23]]
D=[[-40.42, -30.32, -94.21, 37.52, 3.25, 15.82],[-40.51, -37.7, -97.91, 48.77, 3.25, 15.82]]

class Camera:
    def __init__(self,index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cameraMatrix =np.array( [[639.50350871,   0.    ,     311.3847461 ],
         [  0.     ,    638.50603424 ,219.9975647 ],
         [  0.        ,   0.        ,   1.        ]])

        self.distCoeffs = np.array([[ 1.47614748e-01 ,-1.20671013e+00 , 3.51376303e-03 ,-1.35507970e-03,
   2.57014341e+00]])
        self.mark_size=32
        if not self.cap.isOpened():
            print("no")
            exit()
        for i in range(20):
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

            
            (corners, ids, rejected_corners) = stag.detectMarkers(gray, 11)
            
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
                stag.drawDetectedMarkers(frame, corners, ids)
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
        self.tmp1_angles=[57.48, 4.83, -79.8, -15.11, 0.87, 105.2]
        self.tmp2_angles=[-30.23, -10.72, -60.46, -13.44, 3.33, 15.82]
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
        time.sleep(1)
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
        self.send_coords(self.cam_coords,self.speed)
        self.wait_done()
        self.cam_coords[2]=height
        self.send_coords(self.cam_coords,self.speed)
        self.wait_done()
        self.pump_on()
        time.sleep(2) 
        self.cam_coords[2]=110
        self.send_coords(self.cam_coords,self.speed)
        self.wait_done()
        if target_id==1:
            self.send_angles(self.tmp1_angles,self.speed)
            self.wait_done()
            for i in range(len(A)):
               self.send_angles(A[i],self.speed)
               self.wait_done() 
            self.pump_off()
            time.sleep(2)
            self.send_angles(A[0],self.speed)
            self.wait_done()
        elif target_id==2:
            self.send_angles(self.tmp2_angles,self.speed)
            self.wait_done()
            for i in range(len(B)):
               self.send_angles(B[i],self.speed)
               self.wait_done() 
            self.pump_off()
            time.sleep(2)
            self.send_angles(B[0],self.speed)
            self.wait_done()

        elif target_id==3:
            self.send_angles(self.tmp1_angles,self.speed)
            self.wait_done()
            for i in range(len(C)):
               self.send_angles(C[i],self.speed)
               self.wait_done() 
            self.pump_off()
            time.sleep(2)
            self.send_angles(C[0],self.speed)
            self.wait_done()
        elif target_id==4:
            self.send_angles(self.tmp2_angles,self.speed)
            self.wait_done()
            for i in range(len(D)):
               self.send_angles(D[i],self.speed)
               self.wait_done() 
            self.pump_off()
            time.sleep(2)
            self.send_angles(D[0],self.speed)
            self.wait_done()
        
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
    





        

