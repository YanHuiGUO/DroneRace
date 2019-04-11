from commander import Image_Capture
from geometry_msgs.msg import PoseStamped
import OpenGL.GLUT as GLUT
import OpenGL.GL  as GL
import OpenGL.GLU  as GLU
import numpy as np
import time
import rospy
import threading
import cv2
WIN_W, WIN_H = 800, 600  

class Gate:
    '''
    The coordinates between gazebo and openGl are different
    OpenGL r_x + anticlockwise, r_y + anticlockwise, r_z + anticlockwise
    Gzebo  r_x + clockwise, r_y + clockwise, r_z + anticlockwise
    '''
    def __init__(self):
        self.f_position_x = 0.0
        self.f_position_y = 0.0
        self.f_position_z = 0.0
        self.f_rotate_x = 0.0
        self.f_rotate_y = 0.0
        self.f_rotate_z = 0.0
        self.f_position_x_gt = 0.0
        self.f_position_y_gt = 0.0
        self.f_position_z_gt = 0.0
        self.f_rotate_x_gt = 0.0
        self.f_rotate_y_gt = 0.0
        self.f_rotate_z_gt = 0.0
        self.depth = -5
        self.EYE = np.array([1, 1, 1])                     
        self.LOOK_AT = np.array([0.0, 0.0, 0.0])                 
        self.EYE_UP = np.array([-1.0, -1.0, 0]) 
        self.SCALE_K = np.array([0.1,0.1,0.1])
        self.pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
            'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0,'gate_num':0}
        GLUT.glutInit()                           
        GLUT.glutInitDisplayMode(GLUT.GLUT_DOUBLE | GLUT.GLUT_RGBA | GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(WIN_W, WIN_H)
        GLUT.glutCreateWindow('Gate') # 
        GL.glClearColor(0.0, 0.0, 0.0, 1.0) 
        GL.glEnable(GL.GL_DEPTH_TEST)          
        GL.glEnable(GL.GL_DITHER)
        GL.glShadeModel(GL.GL_SMOOTH) 
   
        GLUT.glutReshapeFunc(self.reshape)
        # Gate_Handle = Gate(10,10,1)
        # GLUT.glutDisplayFunc(self.draw)    
        # GLUT.glutReshapeFunc(self.reshape)    
        #GLUT.glutMainLoop()   
        rospy.init_node("show_pose_node")
        rate = rospy.Rate(100)
        self.pred_pose_show_sub = rospy.Subscriber("gi/gate_pose_pred/pose_show", PoseStamped, self.pred_pose_callback)
        self.gt_pose_sub = rospy.Subscriber("gi/gate_pose_gt/pose", PoseStamped, self.gt_pose_callback)
    

    def draw(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glPushMatrix()
        GLU.gluLookAt(self.EYE[0], self.EYE[1], self.EYE[2], \
        self.LOOK_AT[0], self.LOOK_AT[1], self.LOOK_AT[2],\
        self.EYE_UP[0], self.EYE_UP[1], self.EYE_UP[2])

       # print (self.f_rotate_x,self.f_rotate_y,self.f_rotate_z)
        # ---------------------------------------------------------------
        '''
        Draw the coordinate axis
        '''
  
      
        GL.glBegin(GL.GL_LINES)     
        # x axis
        GL.glColor4f(1.0, 0.0, 0.0, 1.0)        
        GL.glVertex3f(0, 0.0, 0.0)         
        GL.glVertex3f(-0.8, 0.0, 0.0)           

        # y axis
        GL.glColor4f(0.0, 1.0, 0.0, 1.0)
        GL.glVertex3f(0.0, 0, 0.0)        
        GL.glVertex3f(0.0, -0.8, 0.0)          

        # z axis
        GL.glColor4f(0.0, 0.0, 1.0, 1.0)     
        GL.glVertex3f(0.0, 0.0, 0)       
        GL.glVertex3f(0.0, 0.0, 0.8)         
        GL.glEnd()                       
        # ---------------------------------------------------------------


        GL.glPushMatrix()
        GL.glScale(self.SCALE_K[0], self.SCALE_K[1], self.SCALE_K[2])

        GL.glTranslatef(self.f_position_x_gt, self.f_position_y_gt, self.f_position_z_gt)
        GL.glRotatef(self.f_rotate_x_gt, 1, 0, 0)
        GL.glRotatef(self.f_rotate_y_gt, 0, 1, 0)
        GL.glRotatef(self.f_rotate_z_gt, 0, 0, 1)

        GL.glBegin(GL.GL_QUADS) 
        self.up_part_gate_gt(0.6,0.6,0.2)
        GL.glEnd() 
        GL.glPopMatrix()

        GL.glPushMatrix()
        GL.glScale(self.SCALE_K[0], self.SCALE_K[1], self.SCALE_K[2])
        GL.glTranslatef(self.f_position_x, self.f_position_y, self.f_position_z)
        GL.glRotatef(self.f_rotate_x, 1, 0, 0)
        GL.glRotatef(self.f_rotate_y, 0, 1, 0)
        GL.glRotatef(self.f_rotate_z, 0, 0, 1)

        GL.glBegin(GL.GL_QUADS) 
        self.up_part_gate_pred(0.6,0.6,0.2)
        GL.glEnd()
        GL.glPopMatrix()

        
        GL.glPopMatrix()

        GLUT.glutSwapBuffers()
    def draw_face1 (self, x,y,z,height,width,thickness):
        # White
        GL.glColor3f(1, 1, 1)
        GL.glVertex3f(x,y,z)
        # Yellow
        GL.glColor3f(1, 1, 0)
        GL.glVertex3f(x,y-thickness,z)
        # Red
        GL.glColor3f(1, 0, 0)
        GL.glVertex3f(x-width,y-thickness,z)
        # Blue
        GL.glColor3f(0, 0, 1)
        GL.glVertex3f(x-width,y,z)

        # White
        GL.glColor3f(1, 1, 1)
        GL.glVertex3f(x,y,z-height)
        # Yellow
        GL.glColor3f(1, 1, 0)
        GL.glVertex3f(x,y-thickness,z-height)
        # Red
        GL.glColor3f(1, 0, 0)
        GL.glVertex3f(x-width,y-thickness,z-height)
        # Blue
        GL.glColor3f(0, 0, 1)
        GL.glVertex3f(x-width,y,z-height)
    def draw_face2 (self, x,y,z,height,width,thickness):
        # White
        GL.glColor3f(1, 1, 1)
        GL.glVertex3f(x,y,z)
        # Yellow
        GL.glColor3f(1, 1, 0)
        GL.glVertex3f(x,y,z-height)
        # Red
        GL.glColor3f(1, 0, 0)
        GL.glVertex3f(x,y-thickness,z-height)
        # Blue
        GL.glColor3f(0, 0, 1)
        GL.glVertex3f(x,y-thickness,z)

       # White
        GL.glColor3f(1, 1, 1)
        GL.glVertex3f(x-width,y,z)
        # Yellow
        GL.glColor3f(1, 1, 0)
        GL.glVertex3f(x-width,y,z-height)
        # Red
        GL.glColor3f(1, 0, 0)
        GL.glVertex3f(x-width,y-thickness,z-height)
        # Blue
        GL.glColor3f(0, 0, 1)
        GL.glVertex3f(x-width,y-thickness,z)

    def up_part_gate_pred(self,height,width,thickness):
        
        self.draw_face1(0.6,0,0.6,height,width,thickness) ##*100
        self.draw_face2(0.6,0,0.6,height,width,thickness) ##*100
    def up_part_gate_gt(self,height,width,thickness):
        
        x,y,z = 0.6,0,0.6
        # White
        GL.glColor3f(0, 1, 0)
        GL.glVertex3f(x,y,z)
        GL.glVertex3f(x,y-thickness,z)
        GL.glVertex3f(x-width,y-thickness,z)
        GL.glVertex3f(x-width,y,z)
        GL.glVertex3f(x,y,z-height)
        GL.glVertex3f(x,y-thickness,z-height)
        GL.glVertex3f(x-width,y-thickness,z-height)
        GL.glVertex3f(x-width,y,z-height)
        GL.glVertex3f(x,y,z)
        GL.glVertex3f(x,y,z-height)
        GL.glVertex3f(x,y-thickness,z-height)
        GL.glVertex3f(x,y-thickness,z)
        GL.glVertex3f(x-width,y,z)
        GL.glVertex3f(x-width,y,z-height)
        GL.glVertex3f(x-width,y-thickness,z-height)
        GL.glVertex3f(x-width,y-thickness,z)
         
        

    def reshape(self,width,height):
        
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        fAspect = width / height
        GLU.gluPerspective(25, fAspect, 1.0, 100000)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glTranslatef(0,0,0)
        GLUT.glutPostRedisplay()
        print (self.f_rotate_x,self.f_rotate_y,self.f_rotate_z)

    def set_gate_pose(self,pose):
        self.f_rotate_x = pose['r_x']
        self.f_rotate_y = pose['r_y']
        self.f_rotate_z = pose['r_z']  
        self.f_position_x = pose['p_x']
        self.f_position_y = pose['p_y']
        self.f_position_z = pose['p_z']  

        self.f_rotate_x_gt = pose['r_x_gt']
        self.f_rotate_y_gt = pose['r_y_gt']
        self.f_rotate_z_gt = pose['r_z_gt']  
        self.f_position_x_gt = pose['p_x_gt']
        self.f_position_y_gt = pose['p_y_gt']
        self.f_position_z_gt = pose['p_z_gt']  
        #self.depth = np.square(pow(self.f_position_x,2)+pow(self.f_position_y,2)+pow(self.f_position_z,2))
        #print (self.depth)
        GLUT.glutPostRedisplay() 
        self.draw()
        print ("pose",pose )

    def start(self):
        GLUT.glutMainLoop()

    def pred_pose_callback(self,msg):
        
        self.pose ['p_x'] = msg.pose.position.y
        self.pose ['p_y'] = -msg.pose.position.x
        self.pose ['p_z'] = msg.pose.position.z
        self.pose ['r_x'] = msg.pose.orientation.x
        self.pose ['r_y'] = msg.pose.orientation.y
        self.pose['r_z'] = msg.pose.orientation.z

        #print ('pred_pose_callback')

    def gt_pose_callback(self,msg):
       
        
        self.pose ['p_x_gt'] = msg.pose.position.y
        self.pose ['p_y_gt'] = -msg.pose.position.x
        self.pose ['p_z_gt'] = msg.pose.position.z
        self.pose ['r_x_gt'] = msg.pose.orientation.x
        self.pose ['r_y_gt'] = msg.pose.orientation.y
        self.pose ['r_z_gt'] = msg.pose.orientation.z
        #print ('gt_pose_callback')

def test_pose_change(Handle):
    for num in range(0,360):
        Handle.f_rotate_x = num
        Handle.f_rotate_y = num
        Handle.f_rotate_z = num  
        # GL.glRotatef(Handle.f_rotate_x, 1, 0, 0)
        # GL.glRotatef(Handle.f_rotate_y, 0, 1, 0)
        # GL.glRotatef(Handle.f_rotate_z, 0, 0, 1)
        print (Handle.f_rotate_x,Handle.f_rotate_y,Handle.f_rotate_z)
        GLUT.glutPostRedisplay() 
        Handle.draw()
        time.sleep(0.1)


if __name__ == "__main__":
             
    # GLUT.glutInit()                           
    # GLUT.glutInitDisplayMode(GLUT.GLUT_DOUBLE | GLUT.GLUT_RGBA | GLUT.GLUT_DEPTH)
    # GLUT.glutInitWindowSize(WIN_W, WIN_H)
    # GLUT.glutCreateWindow('Gate') # 
    # GL.glClearColor(0.0, 0.0, 0.0, 1.0) 
    # GL.glEnable(GL.GL_DEPTH_TEST)          
    # GL.glEnable(GL.GL_DITHER)
    #GL.glShadeModel(GL.GL_SMOOTH) 
  
    Gate_Handle = Gate()
    threading.Thread(target=Gate_Handle.start).start()
    pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
            'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0,'gate_num':0}
   
    '''
    used to get single image in gazebo
    '''
    # img = Image_Capture()
    # count = 1
    # while 1:
    #     image = img.get_image()
    #     if image is not None:
    #         Gate_Handle.set_gate_pose(pose)
    #         cv2.imshow("Camera", image)
    #         cv2.imwrite('image_from_main' + '/camera_image_'+'1'+ '_' +str(count)+'_1.5_1.5'+'.bmp',img.image_raw)
    #         count = count +1
    #         cv2.waitKey (0)

    #img = Image_Capture()
    while 1:
        #image = img.get_image()
        #if image is not None:
        Gate_Handle.set_gate_pose(Gate_Handle.pose)
            #cv2.imshow("Camera", image)
            #cv2.waitKey (1)
      

