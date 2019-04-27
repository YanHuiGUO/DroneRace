import rospy
from control import Px4Controller
from mavros_msgs.msg import GlobalPositionTarget, State,PositionTarget
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32, String
from pyquaternion import Quaternion
from sensor_msgs.msg import Image
import time
import math
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import h5py
import pandas as pd
import numpy as np

class Image_Capture:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_id = 0
        self.image_raw = None
        image_topic = "gi/image_raw"
        rospy.Subscriber(image_topic, Image, self.image_callback)

    def image_callback(self,msg):
        #print("Received an image!")
        self.image_raw  = self.bridge.imgmsg_to_cv2(msg, "bgr8")




class Commander:
    def __init__(self):
        
        rospy.init_node("commander_node")
        rate = rospy.Rate(100)
        self.local_pose = None
        self.b_one_loop_completed = False
        self.h5_chunk_size = 32 ## 0 is elimilated]
        self.chunk_id = 0
        self.line_pd_dump = pd.DataFrame(np.zeros((self.h5_chunk_size,7)), columns = ["p_x","p_y","p_z","Quaternion_x","Quaternion_y","Quaternion_z","Quaternion_w"])
        self.position_target_pub = rospy.Publisher('gi/set_pose/position', PositionTarget, queue_size=10)
        self.yaw_target_pub = rospy.Publisher('gi/set_pose/orientation', Float32, queue_size=10)
        self.custom_activity_pub = rospy.Publisher('gi/set_activity/type', String, queue_size=10)
        '''
        suv mav postion pos 
        ''' 
        self.local_pose_sub = rospy.Subscriber("/gi/local_position/pose", PoseStamped, self.local_pose_callback)

    def Obtain_offboard_node(self,**dictArg):
        self.local_pose = dictArg['pose']

    def local_pose_callback(self, msg):
        self.Obtain_offboard_node(pose = msg)

    def move(self, x, y, z,yaw_degree = 0, BODY_OFF_SET_ENU=True):
        self.position_target_pub.publish(self.set_pose(x, y, z,yaw_degree,BODY_OFF_SET_ENU))


    def turn(self, yaw_degree):
        self.yaw_target_pub.publish(yaw_degree)

    def take_off(self):
        self.custom_activity_pub.publish(String("TAKE_OFF"))

    # land in current position
    def land(self):
        self.custom_activity_pub.publish(String("LAND"))


    # hover at current position
    def hover(self):
        self.custom_activity_pub.publish(String("HOVER"))


    def set_pose(self, x=0, y=0, z=2,yaw = 0,BODY_OFF_SET_ENU = True):
        pose = PositionTarget()
        pose.header.stamp = rospy.Time.now()

        if BODY_OFF_SET_ENU:
            pose.header.frame_id = 'frame.body'
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.yaw = yaw
        else:
            pose.header.frame_id = 'frame.local_enu'
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z 
            pose.yaw = yaw
   

        return pose
    def debug_infor(self):
        pass

    def trigger_data_collection(self,chunk_id,img,count):
        pass
        cv2.imwrite('../image/'+'camera_image_'+str(chunk_id)+'_'+str(count)+'.bmp',img.image_raw )
    
    def collect_data(self,pos,img,count,h,r):
        dict_dump = {} #  map_dict = {'p_x':0,'p_y':1,'p_z':2,'Quaternion_x':3,'Quaternion_y':4,'Quaternion_z':5,'Quaternion_w':6} 
        #same with the xyzw quaternion, needed to be tranformed to polar coodinates or Euler Angles
        dict_dump['p_x'] = pos.pose.position.x
        dict_dump['p_y'] = pos.pose.position.y
        dict_dump['p_z'] = pos.pose.position.z
        dict_dump['Quaternion_x'] = pos.pose.orientation.x
        dict_dump['Quaternion_y'] = pos.pose.orientation.y
        dict_dump['Quaternion_z'] = pos.pose.orientation.z
        dict_dump['Quaternion_w'] = pos.pose.orientation.w

        self.line_pd_dump.loc[count] =dict_dump
        print ("chunk_id:", self.chunk_id,"count:",count)
        global image_fname
        cv2.imwrite('../'+image_fname + '/image/' + 'camera_image_'+str(self.chunk_id)+'_'+str(count)+'_'+str(h)+'_'+str(r)+'.bmp',img.image_raw)
        #print (self.line_pd_dump.loc[count])

        if self.b_one_loop_completed == True:
            path = '../'+pose_fname + '/pose/' +'pose_'+str(self.chunk_id)+'_'+str(h)+'_'+str(r)+'.h5'
            self.line_pd_dump.to_hdf(path,key = 'pose',mode='w')

    '''
    control drone to complete a circle movement
    '''
    def circle_move(self,c_x,c_y,c_z,h,r,direction_flag = 'Anticlockwise'):
        global img,con
        self.b_one_loop_completed = False
        self.chunk_id = self.chunk_id + 1
        '''
        center of gate is (10-0.1,10+0.5,1.931) , 0.325 and 0.5 are the biases of the coordinate of gate. The unit is /m
        '''
        bias_x,bias_y = -0.1,0.5
        if r>c_x: 
            print ("The radius is too big!!! please input a number < "+str(c_x))
            return 

        '''
        move to the begin point
        '''
        start_x = c_x -r +bias_x
        start_y = c_y + bias_y
        for i in range(10):
            con.move(start_x,start_y,h,0,False)
            time.sleep(0.5)
        print ('start:',(start_x,start_y))
        time.sleep(10)
        time_step = 0.2
        count = 1

        if direction_flag == 'Anticlockwise':
            '''
            0-90 degrees
            ''' 
            for theta in range(1,90):
                sin_theta = np.sin(np.deg2rad(theta))
                cos_theta = np.cos(np.deg2rad(theta))
                print (sin_theta,cos_theta)
                next_x = r-r * cos_theta + start_x
                next_y = -r * sin_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1

            '''
            90-180 degrees
            ''' 
            for theta in range(90,180):
                _theta = theta-90
                sin_theta = np.sin(np.deg2rad(_theta))
                cos_theta = np.cos(np.deg2rad(_theta))
                print (sin_theta,cos_theta)
                next_x = r+r * sin_theta + start_x
                next_y = -r * cos_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1
            
            '''
            180-270 degrees
            ''' 
            for theta in range(-180,-90):
                _theta = theta + 180
                sin_theta = np.sin(np.deg2rad(_theta))
                cos_theta = np.cos(np.deg2rad(_theta))
                print (sin_theta,cos_theta)
                next_x = r+r * cos_theta + start_x
                next_y = r * sin_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1
            
            '''
            270-360 degrees
            ''' 
            for theta in range(-90,0):
                _theta = theta + 90
                sin_theta = np.sin(np.deg2rad(_theta))
                cos_theta = np.cos(np.deg2rad(_theta))
                print (sin_theta,cos_theta)
                next_x = r-r * sin_theta + start_x
                next_y = r * cos_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1
                self.b_one_loop_completed = True
                con.collect_data(con.local_pose,img,count,h,r)
            time.sleep(5)

        elif direction_flag == 'Clockwise' :
            '''
            0 - -90 degrees
            ''' 
            for theta in range(-1,-90,-1):
                _theta = theta + 90
                sin_theta = np.sin(np.deg2rad(_theta))
                cos_theta = np.cos(np.deg2rad(_theta))
                print (sin_theta,cos_theta)
                next_x = r-r * sin_theta + start_x
                next_y = r * cos_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1

            '''
            -90- -180 degrees
            ''' 
            for theta in range(-90,-180,-1):
                _theta = theta + 180
                sin_theta = np.sin(np.deg2rad(_theta))
                cos_theta = np.cos(np.deg2rad(_theta))
                print (sin_theta,cos_theta)
                next_x = r+r * cos_theta + start_x
                next_y = r * sin_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1
            
            '''
            180-270 degrees
            ''' 
            for theta in range(180,90,-1):
                _theta = theta-90
                sin_theta = np.sin(np.deg2rad(_theta))
                cos_theta = np.cos(np.deg2rad(_theta))
                print (sin_theta,cos_theta)
                next_x = r+r * sin_theta + start_x
                next_y = -r * cos_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1
            
            '''
            270-360 degrees
            ''' 
            for theta in range(90,-1,-1):
                sin_theta = np.sin(np.deg2rad(theta))
                cos_theta = np.cos(np.deg2rad(theta))
                print (sin_theta,cos_theta)
                next_x = r-r * cos_theta + start_x
                next_y = -r * sin_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1
                self.b_one_loop_completed = True
                con.collect_data(con.local_pose,img,count,h,r)
        time.sleep(5)

    def scan_gate(self,c_x,c_y,c_z,h,r):
        global img,con
        self.b_one_loop_completed = False
        self.chunk_id = self.chunk_id + 1
        '''
        center of gate is (10-0.1,10+0.5,1.931) , 0.325 and 0.5 are the biases of the coordinate of gate. The unit is /m
        '''
        bias_x,bias_y = -0.1,0.5
        if r>c_x: 
            print ("The radius is too big!!! please input a number < "+str(c_x))
            return 

        '''
        move to the begin point
        '''
        start_x = c_x -r +bias_x
        start_y = c_y + bias_y
        for i in range(10):
            con.move(start_x,start_y,h,0,False)
            time.sleep(0.5)
        print ('start:',(start_x,start_y,h))
        time.sleep(5)
        time_step = 0.3
        count = 1
        for theta in range(1,60):
                sin_theta = np.sin(np.deg2rad(theta))
                cos_theta = np.cos(np.deg2rad(theta))
                print (sin_theta,cos_theta)
                next_x = r-r * cos_theta + start_x
                next_y = -r * sin_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1
        for theta in range(60,0,-1):
                sin_theta = np.sin(np.deg2rad(theta))
                cos_theta = np.cos(np.deg2rad(theta))
                print (sin_theta,cos_theta)
                next_x = r-r * cos_theta + start_x
                next_y = -r * sin_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1
        for theta in range(0,-60,-1):
                _theta = theta + 90
                sin_theta = np.sin(np.deg2rad(_theta))
                cos_theta = np.cos(np.deg2rad(_theta))
                print (sin_theta,cos_theta)
                next_x = r-r * sin_theta + start_x
                next_y = r * cos_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1
        for theta in range(-60,0):
                _theta = theta + 90
                sin_theta = np.sin(np.deg2rad(_theta))
                cos_theta = np.cos(np.deg2rad(_theta))
                print (sin_theta,cos_theta)
                next_x = r-r * sin_theta + start_x
                next_y = r * cos_theta + start_y
                print ('----',next_x,next_y,'----')
                con.move(next_x,next_y,h,theta,False)
                time.sleep(time_step)
                con.collect_data(con.local_pose,img,count,h,r)
                count = count+1
                
        self.b_one_loop_completed = True
        con.collect_data(con.local_pose,img,count,h,r)
        time.sleep(5)


    '''
    control drone to pass through the ring
    size of ring:
    1500X1500x20
    1000 r= 20
    1000x650x0.181
    '''
    def pass_through_ring(self,x,y,z):
        pass
            

        

if __name__ == "__main__":
    pose_fname = ''
    image_fname = ''
    con = Commander()
    img = Image_Capture()
    debug = False
    if debug == False:
        pose_fname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        image_fname = pose_fname
        os.makedirs ('../'+pose_fname +'/pose/')
        os.makedirs ('../'+image_fname + '/image/')

    # con.circle_move(10,10,1.931,1.931,5,'Clockwise')
    # con.circle_move(10,10,1.931,1.931,4.5,'Clockwise')
    # con.circle_move(10,10,1.931,1.931,4,'Clockwise')
    # con.circle_move(10,10,1.931,1.931,3.5,'Clockwise')
    # con.circle_move(10,10,1.931,1.931,3,'Clockwise')
    # con.circle_move(10,10,1.931,1.931,2.5,'Clockwise')
    # con.circle_move(10,10,1.931,1.931,2,'Clockwise')

    # con.circle_move(10,10,1.931,1.931,1.3,'Clockwise')
    # con.circle_move(10,10,1.931,1.931,1.5,'Clockwise')
    # con.circle_move(10,10,1.931,1.931,1.7,'Clockwise')

    # con.circle_move(10,10,1.931,1,2,'Clockwise')
    # con.circle_move(10,10,1.931,1.5,2,'Clockwise')
    # con.circle_move(10,10,1.931,2,2,'Clockwise')
    # con.circle_move(10,10,1.931,2.5,2,'Clockwise')

    for height in np.arange(1.5,2.5,0.1):
        for radius in np.arange(1.5,5.1,0.2):
            con.scan_gate(10,10,1.931,height,radius)
            time.sleep(1)

    time.sleep(0.5)


    '''
    read h5  checked
    ''' 
    #print (pd.read_hdf('../pose/'+pose_fname+'/pose_'+str(1)+'.h5', 'pose'))




    # read the entire file into a python array

      
        
      