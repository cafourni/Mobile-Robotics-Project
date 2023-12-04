#This file is to the motion control of the robot when he's in the global path
import numpy as np
import math
from tdmclient import ClientAsync, aw


#Conversion coefficients from Thymio to radians per second
C_conv_toThymio_right = 67.60908181
C_conv_toThymio_left = 67.82946137

threshold_for_convergence = 50 #mm
L = 95 #[mm]
R = 23 #[mm]
           
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
#def control_law(state_estimate_k, x_goal, y_goal,  Kv = 30, Kp_alpha = 0.6 , Kp_beta = 0):
def control_law(state_estimate_k, x_goal, y_goal, speed0, speedGain):
     
     orient_goal = math.atan2(y_goal - state_estimate_k[1], x_goal - state_estimate_k[0])
     print("orient_goal", orient_goal)
     delta_angle = orient_goal - state_estimate_k[2]
     print("delta_angle", delta_angle)

     if abs(delta_angle) > 0.8:
         vr = int(speedGain * delta_angle)
         vl = int(-speedGain * delta_angle)
     else:
        vr = int(speed0 + speedGain * delta_angle)
        vl = int(speed0 - speedGain * delta_angle)
     return vr, vl 
     """ x = state_estimate_k[0]
     y = state_estimate_k[1]
     theta = state_estimate_k[2]
     theta_grad = math.degrees(theta)

     x_diff = x_goal - x
     y_diff = y_goal - y

     angle_to_goal = math.atan2(y_diff, x_diff)
     angle_to_goal_grad = math.degrees(angle_to_goal)
     alpha_grad = (angle_to_goal_grad - theta_grad + 180) % (360) - 180
     #beta = (theta_goal - theta - math.radians(alpha_grad) + math.pi) % (2 * math.pi) - math.pi

     #if alpha_grad > 10:
     w_grad = Kp_alpha * alpha_grad
     w = math.radians(w_grad) + 0 #Kp_beta * beta
  #   else: 
       # w= 0
     v = Kv
    

     return v,w, alpha_grad """
            
        
        
      
            
