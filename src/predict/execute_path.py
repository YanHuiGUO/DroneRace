
import sys
sys.path.append('../')
from commander import Commander
from generate_path import Generate_Path
import quadrocoptertrajectory as quadtraj
from geometry_msgs.msg import PoseStamped,Vector3
import numpy as np
import queue
import time
import math
import rospy
import threading
import tty
import os
import termios
class Execute_Class:
    def __init__(self):
        self.pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0}
        self.pred_r =0 # r parameter
        self.optimal_path = None
        self.Duration = 3
        self.update_path = False
        self.con  = Commander()
        self.path_queue_size = 100
        self.goal_pose = np.zeros(4)
        self.mav_pose = np.zeros(6)
        self.path_queue = queue.Queue(self.path_queue_size)
       # rospy.init_node("execute_path_node")
        rate = rospy.Rate(100)
        self.path_generation_pub = rospy.Publisher('gi/path/generation', Vector3, queue_size=10)
        self.movement_pub = rospy.Publisher('gi/path/movement', Vector3, queue_size=10)
        self.pred_pose_sub = rospy.Subscriber("gi/gate_pose_pred/pose_for_path", PoseStamped, self.pred_pose_callback)

        # Define the input limits:
        fmin = 5  #[m/s**2]
        fmax = 25 #[m/s**2]
        wmax = 20 #[rad/s]
        minTimeSec = 0.02 #[s]
        # Define how gravity lies:
        gravity = [0,0,-9.81]
        self.path_handle = Generate_Path(fmin,fmax,wmax, minTimeSec,gravity)
    
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

    def generate_path(self):
        cur_pos= self.con.get_current_mav_pose()

        dict_pos = {} 
        dict_pos['Pos_x'] = cur_pos.pose.position.x
        dict_pos['Pos_y'] = cur_pos.pose.position.y
        dict_pos['Pos_z'] = cur_pos.pose.position.z
        dict_pos['Quaternion_x'] = cur_pos.pose.orientation.x
        dict_pos['Quaternion_y'] = cur_pos.pose.orientation.y
        dict_pos['Quaternion_z'] = cur_pos.pose.orientation.z
        dict_pos['Quaternion_w'] = cur_pos.pose.orientation.w
        q = np.array([dict_pos['Quaternion_w'],dict_pos['Quaternion_x'],dict_pos['Quaternion_y'],dict_pos['Quaternion_z']])
        euler_angle = self.quater_to_euler(q)

        self.mav_pose  =  np.array([dict_pos['Pos_x'],dict_pos['Pos_y'],dict_pos['Pos_z'],\
                            euler_angle[0],euler_angle[1],euler_angle[2]],np.float)
       # print ('mav_pose:',mav_pose)
        '''
        the coordinate between the path planner and gazebo is different
        '''
        pos0 = [self.mav_pose [0], self.mav_pose [1], self.mav_pose [2]] #position
        vel0 = self.path_handle.generate_vel_from_yaw(self.mav_pose[5])
        acc0 = [0, 0, 0] #acceleration

        
        self.goal_pose [0] = self.mav_pose [0] + self.pose['p_x']
        self.goal_pose [1] = self.mav_pose [1] + self.pose['p_y']
        self.goal_pose [2] = np.clip(self.mav_pose[2] + self.pose['p_z'],1.9,2.5)
        self.goal_pose [3] = self.mav_pose [5] + self.pose['r_z']

        self.goal_pose [3] = -360 + self.goal_pose [3] if self.goal_pose [3] > 180 else self.goal_pose [3]
        self.goal_pose [3] =  360 + self.goal_pose [3] if self.goal_pose [3] <-180 else self.goal_pose [3]

        posf = [self.goal_pose [0],self.goal_pose [1], self.goal_pose [2]]  # position

        velf = self.path_handle.generate_vel_from_yaw(self.goal_pose[3])  # velocity
        accf = [0, 0, 0]  # acceleration
        #self.Duration = self.pred_r*1.1
        self.optimal_path = self.path_handle.get_paths_list(pos0,vel0,acc0,posf,velf,accf,self.Duration)
        print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')
        print ('jump_goal:',self.goal_pose)
        #return  self.optimal_path

    def pred_pose_callback(self,msg):
        
        self.pose ['p_x'] = msg.pose.position.x
        self.pose ['p_y'] = msg.pose.position.y
        self.pose ['p_z'] = msg.pose.position.z
        self.pose ['r_x'] = msg.pose.orientation.x
        self.pose ['r_y'] = msg.pose.orientation.y
        self.pose ['r_z'] = msg.pose.orientation.z
        self.pred_r = np.sqrt (pow(self.pose ['p_x'],2)+pow(self.pose ['p_y'],2)+pow(self.pose ['p_z'],2))
        
        if (self.update_path ==  False):
            self.generate_path()
            self.update_path =  True
        #print ('raw_pose',self.pose)
       

    def pass_through(self):
       
        if self.optimal_path is not None:   
            
            time_interval = 0.02
            next_x = 0
            next_y = 0
            next_z = 0
            theta = 0
            #print ('update_path:',self.update_path)
            
            
            #print (self.path_queue)
            while self.path_queue.empty() == False:
                path_tmp = self.path_queue.get()
                print ('next_piont:',path_tmp)
                print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')
                self.con.move(path_tmp[0],path_tmp[1],path_tmp[2],path_tmp[3],False)
                time.sleep(0.05)

            if(self.path_queue.empty()== True):
                self.generate_path() 
            start_t = self.Duration - self.path_queue_size*time_interval
            stop_t  = self.Duration
            #yaw_delta = self.pose['r_z']/self.path_queue_size
            yaw_a = self.pose['r_z']/(np.exp(stop_t - time_interval)-np.exp(start_t)) #[),so stop_t - time_interval
            yaw_b = self.mav_pose[5] - yaw_a * np.exp(start_t)


            theta = self.mav_pose[5]
            for t in np.arange(start_t, stop_t,time_interval):

                '''
                break down the present path and use the replanning path
                '''
                if self.path_queue.full() == True:                         
                    break
                    
                next_x = np.float(self.optimal_path.get_position(t)[0])
                next_y = np.float(self.optimal_path.get_position(t)[1])
                next_z = 1.93#np.float(self.optimal_path.get_position(t)[2])

                self.path_generation_pub.publish(Vector3(next_x,next_y,next_z))
                #print ('get_normal_vector(t):',self.optimal_path.get_normal_vector(t),self.path_handle.generate_yaw_from_vel(self.optimal_path.get_normal_vector(t)))
                #print ('get_velocity(t):',self.optimal_path.get_velocity(t),self.path_handle.generate_yaw_from_vel(self.optimal_path.get_velocity(t)))
                #theta =  self.path_handle.generate_yaw_from_vel(self.optimal_path.get_normal_vector(t),self.mav_pose[5])
                theta = yaw_a * np.exp(t) + yaw_b#theta + yaw_delta

                '''
                deal with the point of -180->180
                '''
                theta = -360 + theta if theta > 180 else theta
                theta =  360 + theta if theta <-180 else theta
                
                self.path_queue.put(np.array([next_x,next_y,next_z,theta]))
               # self.con.move(next_x,next_y,next_z,theta,False)
            
                #time.sleep(0.02)
            return np.array([next_x,next_y,next_z,theta])
    def run(self):    
        global jump_once

        '''
        execute the generated path 
        '''
        if (jump_once >0):
            now_point = self.pass_through()
            #jump_once = jump_once -1
            time.sleep(0.1)

            #self.gate_num_pub.publish(jump_once)
if __name__== '__main__':   
    
    jump_once = 1
    theta = 0
    r = 3
    
    c_x,c_y = 10,10
    bias_x,bias_y = -0.1,0.5
    start_x = c_x -r +bias_x
    start_y = c_y + bias_y
    sin_theta = np.sin(np.deg2rad(theta))
    cos_theta = np.cos(np.deg2rad(theta))
    print (sin_theta,cos_theta)
    next_x = r-r * cos_theta + start_x
    next_y = -r * sin_theta + start_y
   
    mav = Execute_Class()
    '''
    init the mav position 
    '''
    for i in range(10):
        mav.con.move(next_x,next_y,1.93,0,False)
        time.sleep(0.02)
    time.sleep(20)

    while 1:
        mav.run()
