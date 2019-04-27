import OpenGL.GLUT as GLUT
import OpenGL.GL  as GL
import OpenGL.GLU  as GLU
import numpy as np
import time
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
        self.depth = 0
        self.EYE = np.array([1, 1, 1])                     
        self.LOOK_AT = np.array([0.0, 0.0, 0.0])                 
        self.EYE_UP = np.array([-1.0, -1.0, 0]) 
        self.SCALE_K = np.array([0.1,0.1,0.1])  
        GLUT.glutInit()                           
        GLUT.glutInitDisplayMode(GLUT.GLUT_DOUBLE | GLUT.GLUT_RGBA | GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(WIN_W, WIN_H)
        GLUT.glutCreateWindow('Gate') # 
        GL.glClearColor(0.0, 0.0, 0.0, 1.0) 
        GL.glEnable(GL.GL_DEPTH_TEST)          
        GL.glEnable(GL.GL_DITHER)
        GL.glShadeModel(GL.GL_SMOOTH) 
        #GLUT.glutDisplayFunc(self.draw)    
        GLUT.glutReshapeFunc(self.reshape)
        # Gate_Handle = Gate(10,10,1)
        # GLUT.glutDisplayFunc(self.draw)    
        # GLUT.glutReshapeFunc(self.reshape)    
        # GLUT.glutMainLoop()   

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
        GL.glScale(self.SCALE_K[0], self.SCALE_K[1], self.SCALE_K[2])
        GL.glTranslatef(self.f_position_x, self.f_position_y, self.f_position_z)
        GL.glRotatef(self.f_rotate_x, 1, 0, 0)
        GL.glRotatef(self.f_rotate_y, 0, 1, 0)
        GL.glRotatef(self.f_rotate_z, 0, 0, 1)
        self.up_part_gate(0.6,0.6,0.2)


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

    def up_part_gate(self,height,width,thickness):
        GL.glBegin(GL.GL_QUADS) 
        self.draw_face1(0.6,0,0.6,height,width,thickness) ##*100
        self.draw_face2(0.6,0,0.6,height,width,thickness) ##*100

         
        # GL.glVertex3f(50,50,50)
        # GL.glVertex3f(50.0,-50.0,50.0)
        # GL.glVertex3f(-50.0,-50.0,50.0)
        # GL.glVertex3f(-50.0,50.0,50.0)

        # GL.glColor3f(1, 0, 0)
        # GL.glVertex3f(50.0,50.0,-50.0)
        # GL.glVertex3f(50.0,-50.0,-50.0)
        # GL.glVertex3f(-50.0,-50.0,-50.0)
        # GL.glVertex3f(-50.0,50.0,-50.0)


        # GL.glColor3f(0, 1, 0)
        # GL.glVertex3f(50.0,50.0,-50.0) 
        # GL.glVertex3f(50.0,50.0,50.0)
        # GL.glVertex3f(-50.0,50.0,50.0)
        # GL.glVertex3f(-50.0,50.0,-50.0)

        # GL.glVertex3f(50.0,-50.0,-50.0)
        # GL.glVertex3f(50.0,-50.0,50.0)
        # GL.glVertex3f(-50.0,-50.0,50.0)
        # GL.glVertex3f(-50.0,-50.0,-50.0)
        
        
        # GL.glColor3f(0, 0, 1)
        # GL.glVertex3f(50.0,50.0,50.0)
        # GL.glVertex3f(50.0,50.0,-50.0)
        # GL.glVertex3f(50.0,-50.0,-50.0)
        # GL.glVertex3f(50.0,-50.0,50.0)
        # GL.glVertex3f(-50.0,50.0,50.0)
        # GL.glVertex3f(-50.0,50.0,-50.0)
        # GL.glVertex3f(-50.0,-50.0,-50.0)
        # GL.glVertex3f(-50.0,-50.0,50.0)

        GL.glEnd() 

    def reshape(self,width,height):
        
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        fAspect = width / height
        GLU.gluPerspective(20, fAspect, 1.0, 100000)
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
        #self.depth = np.square(pow(self.f_position_x,2)+pow(self.f_position_y,2)+pow(self.f_position_z,2))
        print (self.depth)
        GLUT.glutPostRedisplay() 
        self.draw()

    def start(self):
        GLUT.glutMainLoop()

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
    GLUT.glutDisplayFunc(Gate_Handle.draw)  
    pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':30,'r_y':0,'r_z':0}
    #Gate_Handle.draw()
    try:
        thread.start_new_thread(Gate_Handle.set_gate_pose , (pose,) )
        thread.start_new_thread(Gate_Handle.start, () )
    except:
        print ("Error: unable to start thread")
    
    while 1:
        pass
      

