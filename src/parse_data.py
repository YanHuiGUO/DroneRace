import cv2
import os
import h5py
import pandas as pd
import numpy as np
import math
import re
import glob
from show_gate_pose import Gate
import _thread
from pathlib import Path
import random
class Parse_helper:
    
    def __init__(self,file_froup =None,img_file =None):
        pass
        self.file_group = file_froup
        self.file_path_img = img_file#['../image/2019-03-15-16-06-18','']
        #self.file_path_pose = pose_f#['../pose/2019-03-15-16-06-18','']
        self.image = None
        self.pose = None
        
        self.phi_max = 90
        self.phi_min = -90
        
        self.theta_max = 90
        self.theta_min = -90
        
        self.r_max = 7 #(horizon 6m, height 2m)
        self.r_min = 0.1 # for the minimum offset
        
        self.yaw_max = 180 #+-90
        self.yaw_min = -180

        self.gate_num  = 0 
    
    def get_yaw_max(self):
        return self.yaw_max
    def get_yaw_min(self):
        return self.yaw_min
    
    def get_r_max(self):
        return self.r_max
    def get_r_min(self):
        return self.r_min
    
    def get_phi_max(self):
        return self.phi_max
    def get_phi_min(self):
        return self.phi_min
    
    def get_theta_max(self):
        return self.theta_max
    def get_theta_min(self):
        return self.theta_min

    def read_image_paths(self,idx_file = 0):
        pass
        img_paths = glob.glob(self.file_path_img[idx_file]+'/*.bmp')
        #print (img_paths)
        return img_paths
    def get_gate_num(self):
        return self.gate_num
    def read_pair(self):   
        img_path = str(self.file_path_img)
        img_path_tmp = img_path.rstrip('.bmp')
        #print (re.findall(r"\d+_\d+_\d+\.?\d*_\d+\.?\d*",img_path_tmp))
        print (img_path_tmp)
        info = re.findall(r"\d+_\d+_\d+\.?\d*_-?\d+\.*\d*",img_path_tmp)[0].split('_')
        #print('info:',info)
        chunk_id,id_frame,circle_num,gate_num = info[0],info[1],info[2],int(info[3])
        self.gate_num = gate_num
        pose_path = Path(self.file_group+'pose/pose_'+chunk_id+'_'+str(circle_num)+'_'+str(circle_num)+'.h5')
        #print ('pose_path:',str(pose_path))
        pose_data = pd.read_hdf(str(pose_path), 'pose')
        self.pose = pose_data.loc[int(id_frame)]
        self.image = cv2.imread(img_path)
        #self.image = cv2.resize(self.image,(64, 64), interpolation=cv2.INTER_CUBIC)
        
        q = np.array([self.pose['Quaternion_w'],self.pose['Quaternion_x'],self.pose['Quaternion_y'],self.pose['Quaternion_z']])
        #print (q)
        euler_angle = self.quater_to_euler(q)
        #print (euler_angle)
        
        pos = np.array([self.pose['p_x'],self.pose['p_y'],self.pose['p_z']])
        print ('chunk_id:',chunk_id,'id_frame:',id_frame,'circle_num:',circle_num,'gate_num:',gate_num)
        return dict({'image':self.image,'pose':dict({'Roll_x':euler_angle[0],\
                                                'Pitch_y':euler_angle[1],\
                                                'Yaw_z':euler_angle[2],\
                                                'Pos_x':pos[0],\
                                                'Pos_y':pos[1],\
                                                'Pos_z':pos[2]})})
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
#-0.21735269  0.01972332  0.02020593 -0.97568475
        phi = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        theta = math.asin(2*(w*y-z*x))
        psi = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))

        Euler_Roll_x = phi*180/math.pi
        Euler_Pitch_y = theta*180/math.pi
        Euler_Yaw_z = psi*180/math.pi

        return (Euler_Roll_x,Euler_Pitch_y,Euler_Yaw_z)

    def transformaiton_world_to_mav(self,gate,mav):
        yaw = mav[5]
        rotate_angle = yaw if yaw>=0 else 360+yaw
        Rotation = np.array([[np.cos(np.deg2rad(rotate_angle)),np.sin(np.deg2rad(rotate_angle)), 0],\
                             [-np.sin(np.deg2rad(rotate_angle)),np.cos(np.deg2rad(rotate_angle)),0],\
                             [0,0,1 ]])
        Ttranslation = np.array([-mav[0],-mav[1],-mav[2]])
        orig_Corr = np.array([gate[0],gate[1],gate[2]])
        new_Corr = np.dot(Rotation,orig_Corr + Ttranslation) 
        return new_Corr

    def transformaiton_mav_to_world(self,gate,mav):
        '''
        return relative coordiante in the world frame
        '''
        yaw = mav[5]
        rotate_angle = yaw if yaw>=0 else 360+yaw
        Rotation = np.array([[np.cos(np.deg2rad(rotate_angle)),np.sin(np.deg2rad(rotate_angle)), 0],\
                             [-np.sin(np.deg2rad(rotate_angle)),np.cos(np.deg2rad(rotate_angle)),0],\
                             [0,0,1 ]])
        Ttranslation = np.array([-mav[0],-mav[1],-mav[2]])
        orig_Corr = np.array([gate[0],gate[1],gate[2]])
        new_Corr = np.dot(Rotation.transpose(),orig_Corr) 

        return new_Corr

    def generate_train_data(self,gate_pose,mav_pose):
        '''
        data_format: pos_x, pos_y, pos_z, r_x,r_y, r_z
                      x
                      |
                      |
                      |
        y<------------  (phi is the angle with x,theta is the angle with height)
        '''
        
        '''
        here deal with the saltation 180<-0->-180
        ''' 
        tmp1 = 180 - math.fabs(gate_pose[5])
        tmp2 = 180 - math.fabs(mav_pose[5])
        yaw_delta = gate_pose[5] - mav_pose[5] if (tmp1+tmp2 )> math.fabs(gate_pose[5] - mav_pose[5]) else tmp1+tmp2

        gate_corr = self.transformaiton_world_to_mav(gate_pose,mav_pose)

        r = np.sqrt(pow(gate_corr[0],2)+pow(gate_corr[1],2) + pow(gate_corr[2],2))

        sin_theta = gate_corr[2]/r
        theta = math.asin(sin_theta) * 180/math.pi 

        horizon_dis = np.sqrt(pow(gate_corr[0],2)+pow(gate_corr[1],2))
        sin_phi = gate_corr[1]/horizon_dis
        phi = math.asin(sin_phi) * 180/math.pi

        return np.array([[r,theta,phi,yaw_delta],[(r-self.r_min)/(self.r_max - self.r_min), (theta-self.theta_min)/(self.theta_max - self.theta_min), (phi - self.phi_min)/(self.phi_max - self.phi_min),(yaw_delta-self.yaw_min)/(self.yaw_max - self.yaw_min)]])



if __name__ == '__main__':
    test_file = ["../sample_data/"]
    image_paths=(list(Path(test_file[0]+'image/').glob("*.bmp")))
    #print (image_paths)
    Gate_Handle = Gate()
    set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0}
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
    try:
        #thread.start_new_thread(Gate_Handle.set_gate_pose , (set_pose,) )
        _thread.start_new_thread(Gate_Handle.start, () )
    except:
        print ("Error: unable to start thread")
    for image_path in image_paths:
        parse = Parse_helper(test_file[0],image_path)
        pair_data = parse.read_pair()
         
        gate_pose = gate_pose_group(parse.gate_num)
        mav_pose =  np.array([pair_data['pose']['Pos_x'],pair_data['pose']['Pos_y'],pair_data['pose']['Pos_z'],\
                                    pair_data['pose']['Roll_x'],pair_data['pose']['Pitch_y'],pair_data['pose']['Yaw_z']],np.float)
        vectors = parse.generate_train_data(gate_pose,mav_pose)

        norm_vector = vectors[1]

        print ('raw:',vectors[0])
        print ('norm:',norm_vector)
        print ('mav_pose:',mav_pose)

        horizen_dis = vectors[0][0] * vectors[2][0]
        p_x = horizen_dis * np.cos(np.deg2rad(vectors[0][2]))
        p_y = horizen_dis * np.sin(np.deg2rad(vectors[0][2])) # phi
       
        set_pose['p_x'],set_pose['p_y'],set_pose['p_z']  = (p_y)/2, (-p_x)/2, 0
        set_pose['r_x'],set_pose['r_y'],set_pose['r_z'] = pair_data['pose']['Roll_x'],\
                                                          pair_data['pose']['Pitch_y'],\
                                                          -pair_data['pose']['Yaw_z']
        
        #print ('gt:',mav_pose)
        Gate_Handle.set_gate_pose(set_pose)
        cv2.imshow("Image", parse.image)
        cv2.waitKey (0) 
    # while True:
        
    #     cv2.imshow("Image", parse.image)
    #     cv2.waitKey (0)  
    #     pass
        #print (pair_data['pose'])
