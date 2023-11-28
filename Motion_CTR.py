#This file is to the motion control of the robot when he's in the global path
import numpy as np

def moving_robot(node,path)

    new_path = path.T
    for colonne in new_path:
        tar_x = col[0]
        tar_y = col[1]
        pos_x, pos_y, orientation_angle, no_cam = "From Computer Vision" # the cam give the robot position, the angle and a bool cam that determine if the cam is obstructed or not.
        tol = 0.5 #tolerance
        tol_angle = np.pi/180 #angle tolerance of 1 degree
        a = abs(tar_x - pos_x)
        b = abs(tar_y - pos_y) 
        distance_to_move = sqrt(a**2 + b**2)
        angle_to_move = np.arctan2(b,a)
        if angle_to_move < 0:
            angle_to_move += 2*np.pi
            
        while distance_to_move > tolerance:
            motor_global_speed_l = 100
            motor_global_speed_r = 100
            diff_angle = angle_to_move - orientation_angle
            if abs(diff_angle) > tol_angle:
                if diff_angle > 0:
                    motor_global_speed_r += 50
                else :
                    motor_global_speed_l += 50
        #To verify , not sure
        if no_cam == true :
            pos_x += motor_l_target * 0.1 * np.cos(orientation_angle)
            pos_y += motor_r_target * 0.1 * np.sin(orientation_angle)
            orientation_angle += angle_difference
        motor_global = [motor_global_speed_l, motor_global_speed_r]
        return motor_global
           
         
                
            
          
            
            
        
        
      
            
