#This file is to the motion control of the robot when he's in the global path
import numpy as np
           
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
    
############################################################################################################################################

def read_motors_speed(node,client):
    aw(node.wait_for_variables({"motor.left.speed","motor.right.speed"}))
    aw(client.sleep(0.01))
    speed=[node.v.motor.left.speed, node.v.motor.right.speed]
    return speed

############################################################################################################################################


            
            
        
        
      
            
