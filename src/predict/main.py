import sys
sys.path.append('../')
import numpy as np
from geometry_msgs.msg import PoseStamped,Vector3
from generate_path import Generate_Path
from std_msgs.msg import Int32
from std_msgs.msg import Float32MultiArray
import quadrocoptertrajectory as quadtraj
from commander import Commander
import tensorflow as tf
from commander import Commander
from commander import Image_Capture
from show_gate_pose import Gate
from model import get_DronNet_model
from pathlib import Path
import keras as K
import threading
import time
import cv2
import math
import rospy
import os
class MAV_Jump_Ring:
    def __init__(self):
        self.optimal_path =  None
        self.local_pose = None
        self.update_path =  False
        rospy.init_node("pred_pose_node")
        rate = rospy.Rate(100)
        #self.model = get_DronNet_model(3)
        self.model = K.models.load_model(str(Path("../models/eleven.hdf5")))
        #self.Gate_Handle = Gate()

        self.set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                        'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0,'gate_num':0}
        self.pred_gate_pose_pub = rospy.Publisher('gi/gate_pose_pred/pose', PoseStamped, queue_size=10)
        self.pred_gate_for_path_pub = rospy.Publisher("gi/gate_pose_pred/pose_for_path", PoseStamped, queue_size=10)
        self.gt_gate_pose_pub = rospy.Publisher('gi/gate_pose_gt/pose', PoseStamped, queue_size=10)
        self.gate_num_pub = rospy.Publisher('gi/gate/gate_num', Int32, queue_size=10)

        self.local_pose_sub = rospy.Subscriber("/gi/local_position/pose", PoseStamped, self.local_pose_callback)

    def Obtain_offboard_node(self,**dictArg):
        self.local_pose = dictArg['pose']
    def local_pose_callback(self, msg):
        self.Obtain_offboard_node(pose = msg)

    def publish_gate_pose(self,gate_pose):
        pred_pose_helper = PoseStamped()
        pred_pose_helper.header.stamp = rospy.Time.now()
        pred_pose_helper.header.frame_id = 'pred_gate_pose'
        pred_pose_helper.pose.position.x = gate_pose['p_x']
        pred_pose_helper.pose.position.y = gate_pose['p_y']
        pred_pose_helper.pose.position.z = gate_pose['p_z']
        pred_pose_helper.pose.orientation.x = gate_pose['r_x']
        pred_pose_helper.pose.orientation.y = gate_pose['r_y']
        pred_pose_helper.pose.orientation.z = gate_pose['r_z']
        pred_pose_helper.pose.orientation.w = gate_pose['gate_num']
        self.pred_gate_pose_pub.publish(pred_pose_helper)
        self.pred_gate_for_path_pub.publish(pred_pose_helper)
        #time.sleep(0.01)
        gt_pose_helper = PoseStamped()
        gt_pose_helper.header.stamp = rospy.Time.now()
        gt_pose_helper.header.frame_id = 'gt_gate_pose'
        gt_pose_helper.pose.position.x = gate_pose['p_x_gt']
        gt_pose_helper.pose.position.y = gate_pose['p_y_gt']
        gt_pose_helper.pose.position.z = gate_pose['p_z_gt']
        gt_pose_helper.pose.orientation.x = gate_pose['r_x_gt']
        gt_pose_helper.pose.orientation.y = gate_pose['r_y_gt']
        gt_pose_helper.pose.orientation.z = gate_pose['r_z_gt']
        gt_pose_helper.pose.orientation.w = 0
        self.gt_gate_pose_pub.publish(gt_pose_helper)
        time.sleep(0.01)


    def get_predict(self,image):
        r_max = 10#7
        r_min = 0.1 

        phi_max = 90
        phi_min = -90

        theta_max = 90
        theta_min = 0

        yaw_max = 360 #180
        yaw_min = -360

        pred = self.model.predict(np.expand_dims(image,0))

        r = pred[0] * (r_max-r_min) + r_min
        theta = pred[1] * (theta_max - theta_min) + theta_min
        phi = pred[2] * (phi_max - phi_min) +  phi_min
        yaw = pred[3] * (yaw_max - yaw_min) + yaw_min

        horizen_dis =  r * np.sin(np.deg2rad(theta))
        p_x = horizen_dis * np.cos(np.deg2rad(phi))
        p_y = horizen_dis * np.sin(np.deg2rad(phi))# phi
        p_z = r * np.cos(np.deg2rad(theta))
        #print ("pred_pose:",np.array([r,theta,phi,yaw]))
        self.set_pose['p_x'], self.set_pose['p_y'],self.set_pose['p_z'],self.set_pose['r_z']  = p_x, p_y, p_z, yaw
        #show the gate in openGL
        #self.Gate_Handle.(self.set_pose)
        
        return np.array([p_x,p_y,p_z,yaw]),np.array([r,theta,phi,yaw])
    
    def get_gate_mav_pose(self,gate_num):
        pass
        pos= self.local_pose
        print(pos)
        dict_pos = {} 
        dict_pos['Pos_x'] = pos.pose.position.x
        dict_pos['Pos_y'] = pos.pose.position.y
        dict_pos['Pos_z'] = pos.pose.position.z
        dict_pos['Quaternion_x'] = pos.pose.orientation.x
        dict_pos['Quaternion_y'] = pos.pose.orientation.y
        dict_pos['Quaternion_z'] = pos.pose.orientation.z
        dict_pos['Quaternion_w'] = pos.pose.orientation.w
        q = np.array([dict_pos['Quaternion_w'],dict_pos['Quaternion_x'],dict_pos['Quaternion_y'],dict_pos['Quaternion_z']])
        euler_angle = self.quater_to_euler(q)
        
        #the postion of all gates
        gate_pose_group = np.array([\
            [10.0, 10.0, 1.93, 0, 0, np.rad2deg(0)],\
            [15.5, 11.0, 1.93, 0, 0, np.rad2deg(0.55)],\
            [20.0, 14.0, 1.93, 0, 0, np.rad2deg(0.9)],\
            [22.8, 19.0, 1.93, 0, 0, np.rad2deg(1.6)],\
            [22.0, 25.0, 1.93, 0, 0, np.rad2deg(2.0)],\
            [17.0, 30.0, 1.93, 0, 0, np.rad2deg(2.8)],\
            [11.0, 29.0, 1.93, 0, 0, np.rad2deg(-2.5)],\
            [ 7.5, 25.0, 1.93, 0, 0, np.rad2deg(-1.8)],\
            [ 5.0, 22.3, 1.93, 0, 0, np.rad2deg(-2.3)],\
            [ 4.0, 17.3, 1.93, 0, 0, np.rad2deg(-1.3)],\
            [ 5.5, 13.0, 1.93, 0, 0, np.rad2deg(-0.7)]])
        
        # if len(gate_pose_group) < gate_num :
        #     raise ValueError('Invalid value of gate_num')
        gate_pose = gate_pose_group[0]
        #gate_pose = gate_pose_group[len(gate_pose_group) - gate_num]
        mav_pose =  np.array([dict_pos['Pos_x'],dict_pos['Pos_y'],dict_pos['Pos_z'],\
                            euler_angle[0],euler_angle[1],euler_angle[2]],np.float)
        horizon_dis = np.sqrt(pow(gate_pose[0]-mav_pose[0],2)+pow(gate_pose[1]-mav_pose[1],2))
        sin_phi = (gate_pose[1]-mav_pose[1])/horizon_dis
        phi = math.asin(sin_phi) * 180/math.pi 
        r = np.sqrt(pow(gate_pose[2]-mav_pose[2],2)+pow(horizon_dis,2)) 
        sin_theta = horizon_dis/r
        theta = math.asin(sin_theta) * 180/math.pi 
        yaw_delta = (gate_pose[5]-mav_pose[5])


        # mav's pose
        position = np.array([dict_pos['Pos_x'],dict_pos['Pos_y'],dict_pos['Pos_z']])
        yaw = euler_angle[2]

        return np.array([r,theta,phi,yaw_delta]) , np.append(position,yaw)

    def offset_pos(self,gate_position,mav_position):
        return gate_position - mav_position
        
    '''
    quater to euler angle /degree
    '''
    def quater_to_euler(self,q):
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        phi = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        theta = math.asin(2*(w*y-z*x))
        psi = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))

        Euler_Roll_x = phi*180/math.pi
        Euler_Pitch_y = theta*180/math.pi
        Euler_Yaw_z = psi*180/math.pi

        return (Euler_Roll_x,Euler_Pitch_y,Euler_Yaw_z)

    def pred_gate_pose_handle(self,img):
        global jump_once
        image = img.get_image()
        print 
        if image is not None:
            pred_pose,pred_pose_raw = self.get_predict(image)
            gt_pose,mav_pose = self.get_gate_mav_pose(jump_once)
            '''
            show the real gate pose
            '''
            gt_r = gt_pose[0]
            gt_theta = gt_pose[1] 
            gt_phi = gt_pose[2] 
            gt_yaw = gt_pose[3] 
            gt_horizen_dis =  gt_r * np.sin(np.deg2rad(gt_theta))
            gt_p_x = gt_horizen_dis * np.cos(np.deg2rad(gt_phi))
            gt_p_y = gt_horizen_dis * np.sin(np.deg2rad(gt_phi)) # phi
            gt_p_z = gt_r * np.cos(np.deg2rad(gt_theta))
            self.set_pose['p_x_gt'],self.set_pose['p_y_gt'],self.set_pose['p_z_gt'],self.set_pose['r_z_gt']  = gt_p_x, gt_p_y, gt_p_z, gt_yaw

            print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')
            
            print ("pred_pose:",pred_pose)
            print ("gt_pose:",np.array([gt_p_x,gt_p_y,gt_p_z,gt_yaw]))
            print ("mav_pose:",mav_pose)
            self.publish_gate_pose(self.set_pose)
            
            print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')

           

if __name__== '__main__':
    mav = MAV_Jump_Ring()
    img = Image_Capture()
    jump_once = 0
    while 1:
        
        mav.pred_gate_pose_handle(img)
        if img.get_image() is not None:
            cv2.imshow("Camera", img.get_image())
            cv2.waitKey (1)
                
        
    
