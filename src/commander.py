import rospy
from control import Px4Controller
from mavros_msgs.msg import GlobalPositionTarget, State
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
class Image_Capture:
    def __init__(self):
        self.bridge = CvBridge()
        image_topic = "gi/image_raw"
        rospy.Subscriber(image_topic, Image, self.image_callback)

    def image_callback(self,msg):
        print("Received an image!")
        cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite('image/camera_image.png', cv2_img)


class Commander:
    def __init__(self):
        
        rospy.init_node("commander_node")
        rate = rospy.Rate(20)
        self.local_pose = None

        self.position_target_pub = rospy.Publisher('gi/set_pose/position', PoseStamped, queue_size=10)
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

    def move(self, x, y, z, BODY_OFF_SET_ENU=True):
        self.position_target_pub.publish(self.set_pose(x, y, z, BODY_OFF_SET_ENU))

    def turn(self, yaw_degree):
        self.yaw_target_pub.publish(yaw_degree)

    
    # land in current position
    def land(self):
        self.custom_activity_pub.publish(String("LAND"))


    # hover at current position
    def hover(self):
        self.custom_activity_pub.publish(String("HOVER"))


    def set_pose(self, x=0, y=0, z=2, BODY_OFF_SET_ENU = True):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()

        if BODY_OFF_SET_ENU:
            pose.header.frame_id = 'frame.body'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
        else:
            pose.header.frame_id = 'frame.local_enu'
            pose.pose.position.x = y
            pose.pose.position.y = -x
            pose.pose.position.z = z       

        return pose
    def debug_infor(self):
        print(self.local_pose)

if __name__ == "__main__":

    con = Commander()
    img = Image_Capture()
    con.move(0,0,1,False)
    con.debug_infor()
    # time.sleep(1)
    # con.land()

