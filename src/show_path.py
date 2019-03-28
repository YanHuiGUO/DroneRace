from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from std_msgs.msg import Int32
from geometry_msgs.msg import Vector3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import rospy
import time


def path_geneation_callback(msg):
    
    if msg is not None:
        plot_temp[0].append(msg.x)
        plot_temp[1].append(msg.y)
        plot_temp[2].append(msg.z)
        #print(msg)


    
   # print (generation_vec)
def gate_num_callback(msg):
    global gate_num
    gate_num = msg.data

def movement_callback(msg):
    global movement_vec
    movement_vec = msg

# is_ipython = 'inline' in matplotlib.get_backend()

# if is_ipython:
#     from IPython import display

# plot_temp = [[] for i in range(3)]

# def plot_durations(vec):
#     ax=plt.subplot(projection='3d')
#    # plt.clf()
#     #ax.scatter(vec.x, vec.y, yvec.z, c='y')
#     plot_temp[0].append(vec[0])
#     plot_temp[1].append(vec[1])
#     plot_temp[2].append(vec[2])
#     #ax.scatter(vec[0], vec[1], vec[2], c='y')
#     ax.scatter(plot_temp[0], plot_temp[1], plot_temp[2],label='path')
#     ax.legend()
#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())


# plt.ion()
def show_path(plot_arr):
        fig=plt.figure(1)
        ax=fig.add_subplot(1,1,1,projection='3d')
        x = plot_arr[0]
        y = plot_arr[1]
        z = plot_arr[2]
        ax.scatter(x[0], y[0], z[0], c='y')
        ax.scatter(x[len(x)-1], y[len(x)-1], z[len(x)-1], c='r')
        ax.plot(x,y,z,label='path')
        ax.legend()
        # print ('show_path:',x,y,z)
        plt.show()

plot_temp = [[] for i in range(3)]
movement_vec = []
rospy.init_node("show_path_node")
rate = rospy.Rate(100)
rospy.Subscriber("gi/path/generation", Vector3, path_geneation_callback)
rospy.Subscriber("gi/path/movement", Vector3, movement_callback)
rospy.Subscriber("gi/gate/gate_num", Int32, gate_num_callback)
gate_num = 1000
while 1:
    print (gate_num)
    time.sleep(1)
    if gate_num < 1 :
       show_path(plot_temp)
       