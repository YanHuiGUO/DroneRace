import sys
sys.path.append('../')
import numpy as np
import math
import quadrocoptertrajectory as quadtraj
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from commander import Commander
import time
'''
notice the coordinate of generate x<----------
path is rigth-hand                           |
                                             |
                                             |
                                             |
                                            y
'''
class Generate_Path:
    '''
    input are the lmitation of the dynamic parameters of MAV (f-> thrust [m/s**2], w->body rate[rad/s], 
    minTimeSec->time interval,gravity->the vector of gravity)
    '''
    def __init__(self, fmin = 0.1,fmax = 2,wmax = 0.79,minTimeSec = 0.02, gravity=[0,0,-9.81]):
        self.fmin = fmin
        self.fmax = fmax
        self.wmax = wmax
        self.minTimeSec = minTimeSec
        self.gravity = gravity
    '''
    only set the velocity a unit, and tranform the yaw to athe unit vel in order to ahieve final correct yaw
    the velocity is based on the world frame. In simulation it uses the frame of gazebo z->yaw-> anticlockwise->+
    input /degree
    '''
    def generate_vel_from_yaw(self, yaw):
        if yaw<-180 or yaw>180:
             raise RuntimeError("yaw is over the normal range")
        return np.array([np.sin(np.deg2rad(yaw)),-np.cos(np.deg2rad(yaw)),0])
        
        
        

    '''
    return /degree
    '''
    def generate_yaw_from_vel(self, vel):
        x = vel[0]
        y = vel[1]
        abs_yaw = 0
        if (x==0 and y==0):
            return 0 
        if (x>=0 and y>0):
            if(x ==0 ):
                return -180
            abs_yaw = np.rad2deg(math.atan(math.fabs (y) / math.fabs(x)))
            return abs_yaw + 90
        elif (x<=0 and y>0):
            if(x ==0 ):
                return -180
            abs_yaw = np.rad2deg(math.atan(math.fabs (y) / math.fabs(x)))
            return -90 - abs_yaw
        elif (x<0 and y<=0):
            if(y ==0 ):
                return -90
            abs_yaw = np.rad2deg(math.atan(math.fabs (y) / math.fabs(x)))
            return -(90 - abs_yaw)
        elif (x>0 and y<=0):
            if(y ==0 ):
                return 90
            abs_yaw = np.rad2deg(math.atan(math.fabs (y) / math.fabs(x)))
            return 90 - abs_yaw
        return abs_yaw

    '''
    input should be six lists
    the start position, velocity, accelebrate ; and the goal position, velocity, accelebrate 
    return: the object of trajectory including two function .get_position(time interval) and .get_velocity(time interval)
    '''
    def get_paths_list(self, pos_s,vec_s,acc_s, pos_g,vec_g,acc_g = [0,0,0],T_duration = 0):
        if T_duration == 0 :
            print ('Wrong duration time!!')
            return
        traj = quadtraj.RapidTrajectory(pos_s, vec_s, acc_s, self.gravity)
        traj.set_goal_position(pos_g)
        traj.set_goal_velocity(vec_g)
        traj.set_goal_acceleration(acc_g)
        traj.generate(T_duration)
        floorPoint  = [0,0,0]  # a point on the floor
        floorNormal = [0,0,1]  # we want to be in this direction of the point (upwards)
        positionFeasible = traj.check_position_feasibility(floorPoint, floorNormal)
        #print("Position feasibility result: ", quadtraj.StateFeasibilityResult.to_string(positionFeasible), "(", positionFeasible, ")")
        return traj

if __name__== '__main__':
    gt_r = 2.56047563
    gt_theta = 80.62601745
    gt_phi = -6.37174979
    gt_yaw = -1.3644157
    gt_horizen_dis =  gt_r * np.sin(np.deg2rad(gt_theta))
    gt_p_x = gt_horizen_dis * np.cos(np.deg2rad(gt_phi))
    gt_p_y = gt_horizen_dis * np.sin(np.deg2rad(gt_phi)) # phi
    gt_p_z = gt_r * np.cos(np.deg2rad(gt_theta))
    # Define the trajectory starting state:
    pos0 = [0, 0, 2] #position
    vel0 = [0, 0, 0] #velocity
    acc0 = [0, 0, 0] #acceleration
    # Define the goal state:
    posf = [gt_p_x,gt_p_y, gt_p_z]  # position
    print ('posf:',posf)
    velf = [np.cos(np.deg2rad(gt_yaw)), np.sin(np.deg2rad(gt_yaw)), 0]  # velocity
    print ('velf:',velf)
    accf = [0, 0, 0]  # acceleration

    # Define the duration:
    Tf = 10

    # Define the input limits:
    fmin = 0.1  #[m/s**2]
    fmax = 2 #[m/s**2]
    wmax = 0.79 #[rad/s]
    minTimeSec = 0.02 #[s]

    # Define how gravity lies:
    gravity = [0,0,-9.81]
    p = Generate_Path(fmin,fmax,wmax, minTimeSec,gravity)
    optimal_path = p.get_paths_list(pos0,vel0,acc0,posf,velf,accf,Tf)
    
    con = Commander()

    for i in range(10):
        con.move(0,0,1,0,False)
        time.sleep(1)
    time_interval = 0.01
    next_x = 0
    next_y = 0
    next_z = 1
    theta = 0
    plot_arr = [[] for i in range(3)]
    for t in np.arange(0,Tf+time_interval,time_interval):
        next_x = np.float(optimal_path.get_position(t)[0])
        next_y = 10*np.float(optimal_path.get_position(t)[1])
        next_z = np.float(optimal_path.get_position(t)[2])
        theta = p.generate_yaw_from_vel(optimal_path.get_velocity(t))
        print (next_x,next_y,next_z,theta)
        con.move(next_x,next_y,next_z,theta,False)
        for idx in range(3):
            plot_arr[idx].append(optimal_path.get_position(t)[idx])
        print ('position:',optimal_path.get_position(t))
        print ('velocity:',optimal_path.get_velocity(t))
        print ('yaw:',p.generate_yaw_from_vel(optimal_path.get_velocity(t)))
        time.sleep(time_interval)
    print (plot_arr)
    fig=plt.figure(1)
    ax=fig.add_subplot(1,1,1,projection='3d')
    x = plot_arr[0]
    y = plot_arr[1]
    z = plot_arr[2]
    ax.scatter(x[0], y[0], z[0], c='y')
    ax.scatter(x[len(x)-1], y[len(x)-1], z[len(x)-1], c='r')
    ax.plot(x,y,z,label='path')
    ax.legend()
    plt.show()

