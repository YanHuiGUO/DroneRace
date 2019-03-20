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
class Parse_helper:
    
    def __init__(self,img_f,pose_f):
        pass
        self.file_path_img = img_f#['../image/2019-03-15-16-06-18','']
        self.file_path_pose = pose_f#['../pose/2019-03-15-16-06-18','']
        self.image = None
        self.pose = None

    def read_image_paths(self,idx_file = 0):
        pass
        img_paths = glob.glob(self.file_path_img[idx_file]+'/*.bmp')
        #print (img_paths)
        return img_paths
    def read_pair(self,img_path):
        idx_file = 0
        img_path_tmp = img_path.rstrip('.bmp')
        #print (re.findall(r"\d+_\d+_\d+\.?\d*_\d+\.?\d*",img_path_tmp))

        info = re.findall(r"\d+_\d+_\d+\.?\d*_\d+\.?\d*",img_path_tmp)[0].split('_')
        #print('info:',info)
        chunk_id,id_frame,height,radius = info[0],info[1],info[2],info[3]

        pose_path = self.file_path_pose[idx_file]+'/pose_'+chunk_id+'_'+str(height)+'_'+str(radius)+'.h5'
       # print ('image_path:'+img_path,'pose_path:'+pose_path)
        pose_data = pd.read_hdf(pose_path, 'pose')
        self.pose = pose_data.loc[int(id_frame)]
        self.image = cv2.imread(img_path)
        q = np.array([self.pose['Quaternion_w'],self.pose['Quaternion_x'],self.pose['Quaternion_y'],self.pose['Quaternion_z']])
        print (q)
        euler_angle = self.quater_to_euler(q)
        print (euler_angle)
        
        pos = np.array((self.pose['p_x'],self.pose['p_y'],self.pose['p_z']))
        pos = self.offset_pos(np.array((10-0.1,10+0.5,1.931)),pos)
        euler_angle = self.offset_pos(np.array((0,0,0)),euler_angle)
        print ('chunk_id:',chunk_id,'id_frame:',id_frame,'height:',height,'radius:',radius)
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

    def generate_train_data(self,b_label_type):
        pass
        '''
        the center of gate is (10-0.1,10+0.5,1.931)
        ''' 
        if b_label_type == 'Position':
            pass

        elif b_label_type == 'Trajectory':
            pass



if __name__ == '__main__':
    parse = Parse_helper(['../image/2019-03-15-16-06-18',''],['../pose/2019-03-15-16-06-18',''])
    image_paths = parse.read_image_paths()
    Gate_Handle = Gate()
    set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0}
    try:
        #thread.start_new_thread(Gate_Handle.set_gate_pose , (set_pose,) )
        _thread.start_new_thread(Gate_Handle.start, () )
    except:
        print ("Error: unable to start thread")
    for img_path in image_paths:
    
        pair_data = parse.read_pair(img_path)
        set_pose['r_x'],set_pose['r_y'],set_pose['r_z'] = -pair_data['pose']['Pitch_y'],\
                                                        -pair_data['pose']['Roll_x'],\
                                                        pair_data['pose']['Yaw_z']
        Gate_Handle.set_gate_pose(set_pose)
        cv2.imshow("Image", parse.image)
        cv2.waitKey (0) 
    # while True:
        
    #     cv2.imshow("Image", parse.image)
    #     cv2.waitKey (0)  
    #     pass
        #print (pair_data['pose'])
