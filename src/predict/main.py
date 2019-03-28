import sys
sys.path.append('../')
import numpy as np
from std_msgs.msg import Int32
import quadrocoptertrajectory as quadtraj
from commander import Commander
import time
import keras as K
import tensorflow as tf
from commander import Commander
from commander import Image_Capture
from show_gate_pose import Gate
from model import get_DronNet_model
from pathlib import Path
import cv2
import math
from generate_path import Generate_Path
import threading
import rospy
from geometry_msgs.msg import PoseStamped,Vector3
import tty
import os
import termios
class MAV_Jump_Ring:
    def __init__(self, init_x,init_y,init_z,init_yaw):
        self.optimal_path =  None
        self.update_path =  False
        self.Duration = 0
        self.init_x = init_x
        self.init_y = init_y
        self.init_z = init_z
        self.init_yaw = init_yaw
        self.model = get_DronNet_model(3)
        self.model = K.models.load_model(str(Path("../models/multigate.hdf5")))
        #self.Gate_Handle = Gate()

        self.set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                        'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0,'gate_num':0}
        self.pred_gate_pose_pub = rospy.Publisher('gi/gate_pose_pred/pose', PoseStamped, queue_size=10)
        self.gt_gate_pose_pub = rospy.Publisher('gi/gate_pose_gt/pose', PoseStamped, queue_size=10)
        self.gate_num_pub = rospy.Publisher('gi/gate/gate_num', Int32, queue_size=10)
        self.path_generation_pub = rospy.Publisher('gi/path/generation', Vector3, queue_size=10)
        self.movement_pub = rospy.Publisher('gi/path/movement', Vector3, queue_size=10)
        # try:
        #     _thread.start_new_thread(self.publish_gate_pose, (self.set_pose) )
        # except:
        #     print ("Error: unable to start thread")
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
        r_max = 6.5#7
        phi_max = 90
        theta_max = 90
        yaw_max = 90 #180


        pred = self.model.predict(np.expand_dims(image,0))[0]

        r = pred[0] * r_max
        theta = pred[1] * theta_max
        phi = pred[2] * phi_max
        yaw = pred[3] * yaw_max

        horizen_dis =  r * np.sin(np.deg2rad(theta))
        p_x = horizen_dis * np.cos(np.deg2rad(phi))
        p_y = horizen_dis * np.sin(np.deg2rad(phi))# phi
        p_z = r * np.cos(np.deg2rad(theta))
        #print ("pred_pose:",np.array([r,theta,phi,yaw]))
        self.set_pose['p_x'], self.set_pose['p_y'], self.set_pose['r_z']  = (p_y), (-p_x),yaw
        #show the gate in openGL
        #self.Gate_Handle.(self.set_pose)
        
        return np.array([p_x,p_y,p_z,yaw]),np.array([r,theta,phi,yaw])
    
    def get_gate_mav_pose(self,con,gate_num):
        pass
        pos= con.get_current_mav_pose()
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
            [10,10.5,1.93,0,0,np.rad2deg(-0.3)],\
            [13,12.5,1.93,0,0,np.rad2deg(0.35)],\
            [16,11.9,1.93,0,0,np.rad2deg(0)],\
            [18.5,11.5,1.93,0,0,np.rad2deg(-0.4)],\
            [21,10.5,1.93,0,0,np.rad2deg(-0.7)],\
            [22.5,8.5,1.93,0,0,np.rad2deg(-1.2)],\
            [10,10.5,1.93,0,0,np.rad2deg(-0.3)],\
            [10,10.5,1.93,0,0,np.rad2deg(-0.3)],\
            [10,10.5,1.93,0,0,np.rad2deg(-0.3)],\
            [10,10.5,1.93,0,0,np.rad2deg(-0.3)]])
        
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
    def move_to_goal(self,con,goal_pose):
        for count in range(0,5):
            con.move(goal_pose[0],goal_pose[1],goal_pose[2],goal_pose[3],False)
            time.sleep(0.01)
    def pass_through(self,path_helper):

        if self.optimal_path == None:
            print (self.optimal_path)
            return 
        time_interval = 0.01
        next_x = 0
        next_y = 0
        next_z = 0
        theta = 0
        # start_yaw = np.rad2deg(np.arccos(vel0[0]))
        # stop_yaw = np.rad2deg(np.arccos(velf[0]))
        # theta = start_yaw 
        
        # delta_yaw = (start_yaw - stop_yaw)/(Duration/time_interval)
        # print ('start_yaw:',start_yaw,'stop_yaw:',stop_yaw,'delta_yaw:',delta_yaw)
        for t in np.arange(0,self.Duration,time_interval):

            '''
            break down the present path and use the replanning path
            '''
            if self.update_path ==True:
                t = 0
                self.update_path = False
                continue
                
            next_x = np.float(self.optimal_path.get_position(t)[0])
            next_y = np.float(self.optimal_path.get_position(t)[1])
            next_z = np.float(self.optimal_path.get_position(t)[2])

            self.path_generation_pub.publish(Vector3(next_x,next_y,next_z))

            pos= con.get_current_mav_pose()
            q = np.array([pos.pose.orientation.w,pos.pose.orientation.x,pos.pose.orientation.y,pos.pose.orientation.z])
            euler_angle = self.quater_to_euler(q)
            
            theta = path_helper.generate_yaw_from_vel(self.optimal_path.get_velocity(t),euler_angle[2])
            #theta = theta + delta_yaw
            #print (next_x,next_y,next_z,theta)
            con.move(next_x,next_y,next_z,theta,False)
            #print (theta)
            #print ('position:',optimal_path.get_position(t))
            #print ('velocity:',optimal_path.get_velocity(t))
            #print ('yaw:',path_helper.generate_yaw_from_vel(optimal_path.get_velocity(t),euler_angle[2]))
            time.sleep(0.02)
        return np.array([next_x,next_y,next_z,theta])
    
    def pred_gate_pose_handle(self,img,path_helper):
        global graph
        global lock
        lock.acquire()
        try:
            image = img.get_image()
            print ('predict thread')
            if image is not None:
                with graph.as_default():
                    pred_pose,pred_pose_raw = self.get_predict(image)
                gt_pose,mav_pose = self.get_gate_mav_pose(con,jump_once)
                #pred_pose = pred_pose_raw = [0,0,0,0]

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
                self.set_pose['p_x_gt'],self.set_pose['p_y_gt'],self.set_pose['r_z_gt']  = (gt_p_y), (-gt_p_x),gt_yaw

                print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')
                
                print ("pred_pose:",pred_pose)
                print ("gt_pose:",np.array([gt_p_x,gt_p_y,gt_p_z,gt_yaw]))
                print ("mav_pose:",mav_pose)

                self.publish_gate_pose(self.set_pose)
                goal_pose = np.zeros(4)
                goal_pose[0] = mav_pose[0] + pred_pose[0]
                goal_pose[1] = mav_pose[1] + pred_pose[1]
                goal_pose[2] = np.clip(mav_pose[2] + pred_pose[2],1,2.2)
                goal_pose[3] = mav_pose[3] + pred_pose[3]

                #offset = [0.3*np.cos(np.deg2rad(goal_pose[3])),0.3*np.sin(np.deg2rad(goal_pose[3]))]
                #goal_pose[0] = goal_pose[0] + offset[0]
                # goal_pose[1] = goal_pose[1] + offset[1]
                
                pos0 = [mav_pose[0], mav_pose[1], mav_pose[2]] #position
                vel0 = [-np.cos(np.deg2rad(mav_pose[3])), np.sin(np.deg2rad(mav_pose[3])),0] #velocity
                acc0 = [0, 0, 0] #acceleration

                
                posf = [goal_pose[0],goal_pose[1], goal_pose[2]]  # position
                velf = [-np.cos(np.deg2rad(goal_pose[3])), np.sin(np.deg2rad(goal_pose[3])), 0]  # velocity
                accf = [0, 0, 0]  # acceleration
                self.Duration = pred_pose_raw[0]*1.1
                self.optimal_path = path_helper.get_paths_list(pos0,vel0,acc0,posf,velf,accf,self.Duration)

                self.update_path = True
                print ('update_path:',self.update_path)
                print ('jump_goal:',goal_pose)
                print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')
        finally:
            lock.release()

    def run(self,con,img,path_helper):    
        global jump_once

        '''
        execute the generated path 
        '''
        if (jump_once >0):
                

            print ('main thread')
            now_point = self.pass_through(path_helper)
            # con.move(now_point[0],now_point[1],now_point[2],pred_pose[3],False)
            # gt_pose,mav_pose = self.get_gate_mav_pose(con,jump_once)
            # offset = [0.2*np.cos(np.deg2rad(mav_pose[3])),0.2*np.sin(np.deg2rad(mav_pose[3]))]
            # con.move(mav_pose[0]+offset[0],mav_pose[1]+offset[1],mav_pose[2],mav_pose[3],False)
            time.sleep(1)
            
            
            #jump_once = jump_once -1
            self.gate_num_pub.publish(jump_once)

                

if __name__== '__main__':

   

    # Define the input limits:
    fmin = 0.1  #[m/s**2]
    fmax = 2 #[m/s**2]
    wmax = 0.79 #[rad/s]
    minTimeSec = 0.02 #[s]

    # Define how gravity lies:
    gravity = [0,0,-9.81]
    path_handle = Generate_Path(fmin,fmax,wmax, minTimeSec,gravity)
    con = Commander()
    img = Image_Capture()
    jump_once = 1
    theta = 0
    r = 2
    
    c_x,c_y = 10,10
    bias_x,bias_y = -0.1,0.5
    start_x = c_x -r +bias_x
    start_y = c_y + bias_y
    sin_theta = np.sin(np.deg2rad(theta))
    cos_theta = np.cos(np.deg2rad(theta))
    print (sin_theta,cos_theta)
    next_x = r-r * cos_theta + start_x
    next_y = -r * sin_theta + start_y
   
    mav = MAV_Jump_Ring(next_x,next_y,2,theta)
    '''
    init the mav position 
    '''
    for i in range(10):
        con.move(mav.init_x,mav.init_y,mav.init_z,mav.init_yaw,False)
        time.sleep(0.02)
    time.sleep(10)
    lock = threading.Lock()
    graph = tf.get_default_graph()
    '''
    start the pose prediction thread
    '''
    try:
        thread_predic = threading.Thread(target=mav.pred_gate_pose_handle,args=(img,path_handle))
        thread_predic.start()
        thread_predic.join()
    except:
        print ("Error: unable to start thread")

    
    while 1:
        mav.run(con, img,path_handle)
        # fd = sys.stdin.fileno()
        # tty.setraw( fd )
        # old_settings = termios.tcgetattr(fd)
        # ch = sys.stdin.read( 1 )
        # if ch == 'g':
        #     jump_once = 0
           # print ('jump_once:',jump_once)