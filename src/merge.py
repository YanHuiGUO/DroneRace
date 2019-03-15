import os
import cv2

img_root = '/home/gyhui/catkin_ws/src/drone_simulate/src/images/'
fps = 24    

#ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('camera.avi',fourcc,fps,(752,480))

for i in range(100,600):
    frame = cv2.imread(img_root+str(i)+'.png')
    videoWriter.write(frame)
videoWriter.release()
