#This file is to the motion control of the robot when he's in the global path
import numpy as np
import math
from tdmclient import ClientAsync, aw


#Conversion coefficients from Thymio to radians per second
C_conv_toThymio_right = 67.60908181
C_conv_toThymio_left = 67.82946137

threshold_for_convergence = 50 #mm
L = 104 #[mm]
R = 20 #[mm]
           
############################################################################################################################################

def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

############################################################################################################################################

def stop_motors(node):
    aw(node.set_variables(motors(0,0)))
    
############################################################################################################################################

def set_motors(left,right,node):
    aw(node.set_variables(motors(left,right)))
    """ d = {
                    "motor.left.target": [left],
                    "motor.right.target": [right],
                }
    aw(node.set_variable(d)) """

############################################################################################################################################

def read_motors_speed(node,client):
    aw(node.wait_for_variables({"motor.left.speed","motor.right.speed"}))
    aw(client.sleep(0.01))
    speed=[node.v.motor.left.speed, node.v.motor.right.speed]
    return speed

############################################################################################################################################
def convert_velocity2vw(vr, vl,C_conv_toThymio_right, C_conv_toThymio_left, L, R):
    # convert right and left wheel velocities in Thymio units to v and w in mm/s and rad/s
    vr_rads= vr/C_conv_toThymio_right
    vl_rads= vl/C_conv_toThymio_left
    v=R* (vr_rads+vl_rads)/2   
    w=R*(vr_rads-vl_rads)/L
    return v, w

############################################################################################################################################
def convert_velocity2RL(v,w,C_conv_toThymio_right, C_conv_toThymio_left,L,R):
    # convert v in mm/s and w in rad/s to right and left wheel velocities in Thymio units
    vr_rads= (2*v+w*L)/(2*R)
    vl_rads= (2*v-w*L)/(2*R)
    vr=vr_rads*C_conv_toThymio_right
    vl=vl_rads*C_conv_toThymio_left
    return vr, vl
    
############################################################################################################################################
def getB(yaw, deltak):
    B = np.array([[np.cos(yaw) * deltak, 0],
                  [np.sin(yaw) * deltak, 0],
                  [0, deltak]])
    return B

############################################################################################################################################
def control_law(state_estimate_k, x_goal, y_goal,  constant_speed = 100, Kp_alpha = 2):

     """
     -->alpha is the angle to the goal relative to the heading of the robot
     -->beta is the angle between the robot's position and the goal position plus the goal angle
     -->Kp_rho*rho and Kp_alpha*alpha drive the robot along a line towards the goal
     -->Kp_beta*beta rotates the line so that it is parallel to the goal angle
    
     """
     x = state_estimate_k[0]
     y = state_estimate_k[1]
     theta = state_estimate_k[2]

     x_diff = x_goal - x
     y_diff = y_goal - y

     alpha = (math.atan2(y_diff, x_diff) - theta + math.pi) % (2 * math.pi) - math.pi

     v = constant_speed
     w = Kp_alpha * alpha

     return v,w
            
        
        
      
            
