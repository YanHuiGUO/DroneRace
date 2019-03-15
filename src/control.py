import rospy
from mavros_msgs.msg import GlobalPositionTarget, State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Imu, NavSatFix
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float64, String
import time
from pyquaternion import Quaternion
import math
import threading


class Px4Controller:

    def __init__(self):

        self.imu = None
        self.gps = None
        self.image_raw = None
        self.local_pose = None
        self.current_state = None
        self.current_heading = None
        self.takeoff_height = 3
        self.initial_heading = 0

        self.cur_target_pose = None
        self.global_target = None

        self.received_new_task = False
        self.arm_state = False
        self.offboard_state = False
        self.received_imu = False
        self.frame = "BODY"

        self.state = None

        '''
        ros subscribers
        '''
        self.local_pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_pose_callback)
        self.mavros_sub = rospy.Subscriber("/mavros/state", State, self.mavros_state_callback)
        self.gps_sub = rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.gps_callback)
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)
        self.camera_sub = rospy.Subscriber("/mavlink/image_raw",Image,self.image_callback)

        self.set_target_position_sub = rospy.Subscriber("gi/set_pose/position", PositionTarget, self.set_target_position_callback)
        self.set_target_yaw_sub = rospy.Subscriber("gi/set_pose/orientation", Float32, self.set_target_yaw_callback)
        self.custom_activity_sub = rospy.Subscriber("gi/set_activity/type", String, self.custom_activity_callback)


        '''
        ros publishers
        '''
        self.local_target_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)

        self.local_infor_pos_pub = rospy.Publisher("gi/local_position/pose", PoseStamped, queue_size=10)

        self.local_infor_image_pub = rospy.Publisher("gi/image_raw", Image, queue_size=10)
        '''
        ros services
        '''
        self.armService = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.flightModeService = rospy.ServiceProxy('/mavros/set_mode', SetMode)


        print("Px4 Controller Initialized!")

    def take_off (self):
        self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

        #print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))

        for i in range(10):
            self.local_target_pub.publish(self.cur_target_pose)
            self.local_infor_pos_pub.publish(self.local_pose)
            #print (self.local_pose)
            self.arm_state = self.arm()
            self.offboard_state = self.offboard()


        # while self.offboard_state is False:
        #     self.local_target_pub.publish(self.cur_target_pose)
        #     self.arm_state = self.arm()
        #     self.offboard_state = self.offboard()


        # self.initial_heading = self.q2yaw(self.imu.orientation)

        self.initial_heading = math.pi / 2
        print("Initial heading set to: ", self.initial_heading)
        print("Vehicle Took Off!")

    def start(self):

        rospy.init_node("offboard_node","commander_node")
        rate = rospy.Rate(100)
        self.take_off()

        '''
        main ROS thread
        '''
        while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):


            self.local_target_pub.publish(self.cur_target_pose)
            self.local_infor_pos_pub.publish(self.local_pose)
            self.local_infor_image_pub.publish(self.image_raw)
            if (self.state is "LAND") and (self.local_pose.pose.position.z < 0.15):

                if(self.disarm()):

                    self.state = "DISARMED"


            time.sleep(0.1)


    def construct_target(self, x, y, z, yaw, yaw_rate = 1):
        target_raw_pose = PositionTarget()
        target_raw_pose.header.stamp = rospy.Time.now()

        target_raw_pose.coordinate_frame = 9

        target_raw_pose.position.x = x
        target_raw_pose.position.y = y
        target_raw_pose.position.z = z

        target_raw_pose.type_mask = PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
                                    + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ \
                                    + PositionTarget.FORCE

        target_raw_pose.yaw = yaw
        target_raw_pose.yaw_rate = yaw_rate

        return target_raw_pose



    '''
    cur_p : poseStamped
    target_p: positionTarget
    '''
    def position_distance(self, cur_p, target_p, threshold=0.1):
        delta_x = math.fabs(cur_p.pose.position.x - target_p.position.x)
        delta_y = math.fabs(cur_p.pose.position.y - target_p.position.y)
        delta_z = math.fabs(cur_p.pose.position.z - target_p.position.z)

        if (delta_x + delta_y + delta_z < threshold):
            return True
        else:
            return False

    def image_callback(self,msg):
        self.image_raw = msg


    def local_pose_callback(self, msg):
        self.local_pose = msg


    def mavros_state_callback(self, msg):
        self.mavros_state = msg.mode


    def imu_callback(self, msg):
        global global_imu, current_heading
        self.imu = msg

        self.current_heading = self.q2yaw(self.imu.orientation)

        self.received_imu = True


    def gps_callback(self, msg):
        self.gps = msg


    def body2enu(self, body_target_x, body_target_y, body_target_z):

        heading_delta = self.initial_heading - self.current_heading

        ENU_y = body_target_y * math.cos(heading_delta) - body_target_x * math.sin(heading_delta)
        ENU_x = body_target_y * math.sin(heading_delta) + body_target_x * math.cos(heading_delta)
        ENU_z = body_target_z

        return ENU_x, ENU_y, ENU_z


    '''
    Receive A Custom Activity
    '''
    def custom_activity_callback(self, msg):

        print("Received Custom Activity:", msg.data)

        if msg.data == "LAND":
            print("LANDING!")
            self.state = "LAND"
            self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                         self.local_pose.pose.position.y,
                                                         0.1,
                                                         self.current_heading)
        if msg.data == "TAKE_OFF":
            self.state = "TAKE_OFF"
            self.take_off()

        if msg.data == "HOVER":
            print("HOVERING!")
            self.state = "HOVER"
            self.hover()


        else:
            print("Received Custom Activity:", msg.data, "not supported yet!")



    def set_target_position_callback(self, msg):
        print("Received New Position Task!")

        '''
        BODY_OFFSET_ENU
        '''
        yaw_deg = msg.yaw * math.pi / 180.0
        if self.frame is "BODY" and msg.header.frame_id=='frame.body':
            new_x, new_y, new_z = self.body2enu(msg.position.x, msg.position.y, msg.position.z)
            print(new_x, new_y, new_z)
            ENU_x = new_x + self.local_pose.pose.position.x
            ENU_y = new_y + self.local_pose.pose.position.y
            ENU_z = new_z + self.local_pose.pose.position.z

            self.cur_target_pose = self.construct_target(ENU_x, ENU_y, ENU_z, yaw_deg)

        else:
            print("LOCAL ENU")
            '''
            LOCAL_ENU
            '''
            
            self.cur_target_pose = self.construct_target(msg.position.x,
                                                         msg.position.y,
                                                         msg.position.z,
                                                         yaw_deg)


    def set_target_yaw_callback(self, msg):
        print("Received New Yaw Task!")

        yaw_deg = msg.data * math.pi / 180.0
        self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                     self.local_pose.pose.position.y,
                                                     self.local_pose.pose.position.z,
                                                     yaw_deg)

    '''
    return yaw from current IMU
    '''
    def q2yaw(self, q):
        if isinstance(q, Quaternion):
            rotate_z_rad = q.yaw_pitch_roll[0]
        else:
            q_ = Quaternion(q.w, q.x, q.y, q.z)
            rotate_z_rad = q_.yaw_pitch_roll[0]

        return rotate_z_rad


    def arm(self):
        if self.armService(True):
            return True
        else:
            print("Vehicle arming failed!")
            return False

    def disarm(self):
        if self.armService(False):
            return True
        else:
            print("Vehicle disarming failed!")
            return False


    def offboard(self):
        if self.flightModeService(custom_mode='OFFBOARD'):
            return True
        else:
            print("Vechile Offboard failed")
            return False


    def hover(self):

        self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
                                                     self.local_pose.pose.position.y,
                                                     self.local_pose.pose.position.z,
                                                     self.current_heading)


if __name__ == '__main__':

    con = Px4Controller()
    con.start()
