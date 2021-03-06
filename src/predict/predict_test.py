import sys
sys.path.append('../')
import keras as K
from model import get_DronNet_model
from parse_data import Parse_helper
from generator import TrainImageGenerator, ValGenerator
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import os
import numpy as np
import _thread
from show_gate_pose import Gate
import cv2
import time
class Predcit:
    def __init__(self,model_file = "../models/eleven.hdf5"):
        self.model = get_DronNet_model(3)
        self.model = K.models.load_model(str(Path(model_file)))
        self.Gate_Handle = Gate()
        self.set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0}
        try:
        #thread.start_new_thread(Gate_Handle.set_gate_pose , (set_pose,) )
            _thread.start_new_thread(self.Gate_Handle.start, () )
        except:
            print ("Error: unable to start thread")
    def get_predict(self,image):
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
        self.set_pose['p_x'], self.set_pose['p_y'], self.set_pose['r_z']  = (p_y), (-p_x),yaw
        #show the gate in openGL
        self.Gate_Handle.set_gate_pose(self.set_pose)
        return np.array([p_x,p_y,p_z,yaw])

if __name__== '__main__':
    pass #
    #model = get_DronNet_model(3)
    model = K.models.load_model(str(Path("../models/eleven.hdf5")))
    test_file = ["../../2019-04-04-14-19-16/"]
    image_paths=(list(Path(test_file[0]+'image/').glob("*.bmp")))
    image_paths= sorted(image_paths)
    #print (image_paths)
    image_idx = 1
 
    
    Gate_Handle = Gate()
    set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0}
    try:
        #thread.start_new_thread(Gate_Handle.set_gate_pose , (set_pose,) )
        _thread.start_new_thread(Gate_Handle.start, () )
    except:
        print ("Error: unable to start thread")

    for image_path in image_paths:
        parse = Parse_helper(test_file[0],image_path)
        pair_data = parse.read_pair()
        gate_num = parse.get_gate_num()
        image = pair_data['image']

        start = time.clock()
        pred = model.predict(np.expand_dims(image,0))

        elapsed = (time.clock() - start)
        print("Time used:",elapsed)

        r = pred[0] * (parse.get_r_max()-parse.get_r_min()) + parse.get_r_min()
        theta = pred[1] * (parse.get_theta_max() - parse.get_theta_min()) + parse.get_theta_min()
        phi = pred[2] * (parse.get_phi_max() - parse.get_phi_min()) +  parse.get_phi_min()
        yaw = pred[3] * (parse.get_yaw_max() -  parse.get_yaw_min()) +parse.get_yaw_min()
        horizen_dis =  r * np.sin(np.deg2rad(theta))
        p_x = horizen_dis * np.cos(np.deg2rad(phi))
        p_y = horizen_dis * np.sin(np.deg2rad(phi)) # phi
        set_pose['p_x'],set_pose['p_y'],set_pose['r_z']  = (p_y), (-p_x),yaw
 

        gt =  np.array([pair_data['pose']['Pos_x'],pair_data['pose']['Pos_y'],pair_data['pose']['Pos_z'],\
                        pair_data['pose']['Roll_x'],pair_data['pose']['Pitch_y'],pair_data['pose']['Yaw_z']],np.float)
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
        gate_pose = gate_pose_group[gate_num]
        vectors = parse.generate_train_data(gate_pose,gt)
        gt = vectors[1]
        gt_r= gt[0] * (parse.get_r_max()-parse.get_r_min()) + parse.get_r_min()
        gt_theta = gt[1] * (parse.get_theta_max() - parse.get_theta_min()) + parse.get_theta_min()
        gt_phi = gt[2] * (parse.get_phi_max() - parse.get_phi_min()) +  parse.get_phi_min()
        gt_yaw = gt[3] * (parse.get_yaw_max() -  parse.get_yaw_min()) +parse.get_yaw_min()

        gt_horizen_dis =  gt_r * np.sin(np.deg2rad(gt_theta))
        gt_p_x = gt_horizen_dis * np.cos(np.deg2rad(gt_phi))
        gt_p_y = gt_horizen_dis * np.sin(np.deg2rad(gt_phi)) # phi
        set_pose['p_x_gt'],set_pose['p_y_gt'],set_pose['r_z_gt']  = (gt_p_y)/2, (-gt_p_x)/2,gt_yaw
       

        print ("gt_pose:",gt)
        print ("pred_pose:",np.array([r,theta,phi,yaw]))

        print ("opengl_gt_pose:",np.array([gt_p_x,gt_p_y,gt_yaw]))
        print ("opengl_pred_pose:",np.array([p_x,p_y,yaw]))
        Gate_Handle.set_gate_pose(set_pose)
        cv2.imshow("Image", parse.image)
        cv2.waitKey (0)
