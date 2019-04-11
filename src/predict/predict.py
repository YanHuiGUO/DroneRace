import sys
sys.path.append('../')
import numpy as np
import keras as K
from model import get_DronNet_model
from geometry_msgs.msg import PoseStamped,Vector3
from generate_path import Generate_Path
from parse_data import Parse_helper
from std_msgs.msg import Int32
from commander import Commander
from commander import Image_Capture
from show_gate_pose import Gate
import pandas as pd
from pathlib import Path
from predict_test import Predcit
import time
import cv2
import h5py
import math
import rospy
import os
class MAV_Pred:
    def __init__(self):
        self.optimal_path =  None
        self.local_pose = None
        self.update_path =  False
        self.local_pose = None
        self.b_one_loop_completed = False        
        self.h5_chunk_size = 32 ## 0 is excluded
        self.chunk_id = 0
        self.count = 1
        self.circle_num  = 1 
        self.line_pd_dump = pd.DataFrame(np.zeros((self.h5_chunk_size,7)), columns = ["p_x","p_y","p_z","Quaternion_x","Quaternion_y","Quaternion_z","Quaternion_w"])
        rospy.init_node("pred_pose_node")
        rate = rospy.Rate(100)

        self.set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                        'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0,'gate_num':0}
        self.pred_gate_pose_pub = rospy.Publisher('gi/gate_pose_pred/pose', PoseStamped, queue_size=10)
        self.pred_gate_for_path_pub = rospy.Publisher("gi/gate_pose_pred/pose_for_path", PoseStamped, queue_size=10)
        self.gt_gate_pose_pub = rospy.Publisher('gi/gate_pose_gt/pose', PoseStamped, queue_size=10)
        self.gate_num_pub = rospy.Publisher('gi/gate/gate_num', Int32, queue_size=10)
        self.local_pose_sub = rospy.Subscriber("/gi/local_position/pose", PoseStamped, self.local_pose_callback)
       
        #self.model = get_DronNet_model(3)
        self.model = K.models.load_model(str(Path("../models/eleven.hdf5")))

        self.b_switch_gate = False
        self.start = 0
        self.now_gate = 0
        
        self.center_offset = 0.5 #the offset between the center of gate and the gate coordinate
        #the postion of all gates
        self.gate_pose_group = np.array([\
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
    def get_predict(self,image):
        parse = Parse_helper()
        pred = self.model.predict(np.expand_dims(image,0))
        r = pred[0] * (parse.get_r_max()-parse.get_r_min()) + parse.get_r_min()
        theta = pred[1] * (parse.get_theta_max() - parse.get_theta_min()) + parse.get_theta_min()
        phi = pred[2] * (parse.get_phi_max() - parse.get_phi_min()) +  parse.get_phi_min()
        yaw = pred[3] * (parse.get_yaw_max() -  parse.get_yaw_min()) +parse.get_yaw_min()
        horizen_dis =  r * np.sin(np.deg2rad(theta))
        p_x = horizen_dis * np.cos(np.deg2rad(phi))
        p_y = horizen_dis * np.sin(np.deg2rad(phi)) # phi
        p_z = r * np.cos(np.deg2rad(theta))
        print ("pred_pose:",np.array([r,theta,phi,yaw]))
        #show the gate in openGL
        return np.array([p_x,p_y,p_z,yaw])
    '''
    deal the center offet of gate
    '''
    def tansfer_gate_center(self,gate_pose):
        assert len(gate_pose) == 6
        _yaw = gate_pose[5]
        y0 = gate_pose[1]
        x0 = gate_pose[0]

        if (math.fabs(math.fabs(_yaw)-90)<0.001):
            k= 0
        elif (math.fabs(_yaw - 0) < 1) or (math.fabs(math.fabs(_yaw) - 180) < 1):
            k= 100
        else:
            k = -1/math.tan(np.deg2rad(_yaw))
        
        if _yaw >=0 and _yaw <180:
            x = x0 - math.sqrt(pow(self.center_offset,2)/(k*k+1))
            y = y0 + k*(x-x0) if k != 100 else y0 + self.center_offset
        else:
            x = x0 + math.sqrt(pow(self.center_offset,2)/(k*k+1))
            y = y0 + k*(x-x0) if k != 100 else y0 - self.center_offset
        return np.array([x,y,gate_pose[2],gate_pose[3],gate_pose[4],_yaw])


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
        #time.sleep(0.01)

    
    def get_relavtive_pos(self,image):
        
        pos= self.local_pose
        #print(pos)
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

        gate_pose = self.gate_pose_group[self.now_gate]
        gate_pose = self.tansfer_gate_center(gate_pose)
        mav_pose =  np.array([dict_pos['Pos_x'],dict_pos['Pos_y'],dict_pos['Pos_z'],\
                            euler_angle[0],euler_angle[1],euler_angle[2]],np.float)

        # p_x = gate_pose[0] - mav_pose[0] 
        # p_y = gate_pose[1] - mav_pose[1] 
        # p_z = gate_pose[2] - mav_pose[2]

        ## mav_pose [5] ->  the yaw
        '''
        here deal with the saltation 180<-0->-180
        '''
        # tmp1 = 180 - math.fabs(gate_pose[5])
        # tmp2 = 180 - math.fabs(mav_pose[5])
        # yaw = gate_pose[5] - mav_pose[5] if (tmp1+tmp2 )> math.fabs(gate_pose[5] - mav_pose[5]) else tmp1+tmp2
        
        rela_pos = self.get_predict(image)
        p_x,p_y,p_z,yaw = rela_pos[0],rela_pos[1],rela_pos[2],rela_pos[3]

        '''
        check the mav whether fly though a gate and switch its goal to next one
        '''
        dis = math.sqrt(pow(p_x,2)+pow(p_y,2))
        if(dis <= 0.8 and self.b_switch_gate == False):
            self.b_switch_gate = True
            self.start = time.time()
        
        if(self.b_switch_gate ==  True):
            delta_time = time.time() - self.start  
            if(delta_time > 0.1):
                self.b_switch_gate = False
                self.now_gate = self.now_gate + 1
                if (self.now_gate>len(self.gate_pose_group)-1):
                    self.now_gate = 0
                    self.circle_num = self.circle_num + 1
                    self.count = 1
               
                    
                '''
                redefine the goal
                '''
                pos= self.local_pose
                #print(pos)
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
                gate_pose = self.gate_pose_group[self.now_gate]
                mav_pose =  np.array([dict_pos['Pos_x'],dict_pos['Pos_y'],dict_pos['Pos_z'],\
                                    euler_angle[0],euler_angle[1],euler_angle[2]],np.float)
                # p_x = gate_pose[0] - mav_pose[0]
                # p_y = gate_pose[1] - mav_pose[1]
                # p_z = gate_pose[2] - mav_pose[2]
                # yaw = gate_pose[5] - mav_pose[5]
                rela_pos = self.get_predict(image)
                p_x,p_y,p_z,yaw = rela_pos[0],rela_pos[1],rela_pos[2],rela_pos[3]

        print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')
        print ('dis',dis)
        print ('pred',p_x, p_y, p_z, yaw)
        print ("gt_pose:",gate_pose[0],gate_pose[1],gate_pose[2],gate_pose[5])
        print ('self.b_switch_gate:',self.b_switch_gate)
        self.set_pose['p_x'], self.set_pose['p_y'],self.set_pose['p_z'],self.set_pose['r_z']  = p_x, p_y, p_z, yaw
        self.set_pose['p_x_gt'],self.set_pose['p_y_gt'],self.set_pose['p_z_gt'],self.set_pose['r_z_gt']  = gate_pose[0],gate_pose[1],gate_pose[2],gate_pose[5]
        self.count = self.count + 1
        return np.array([p_x,p_y,p_z,yaw])
        
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
    def run(self,img):
        image = img.get_image()
        
        if image is not None:
            rev_pos = self.get_relavtive_pos(image)
            print ("rev_pos:",rev_pos)
            self.publish_gate_pose(self.set_pose) ##  time.sleep 0.01 delay 0.01s
            print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')

           

if __name__== '__main__':
    mav = MAV_Pred()
    img = Image_Capture()
    
    while 1:
        
        mav.run(img)
        # if img.get_image() is not None:
        #     cv2.imshow("Camera", img.get_image())
        #     cv2.waitKey (1)
                
        
    
