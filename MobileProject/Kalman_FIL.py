
import numpy as np
# A matrix
A_k_minus_1 = np.array([[1.0, 0, 0],
                        [0, 1.0, 0],
                        [0, 0, 1.0]])

# Process noise
process_noise_v_k_minus_1 = np.array([0.01, 0.01, 0.003]) #

# State model noise covariance matrix Q_k
Q_k = np.array([[0.0013661, 0, 0],
              [0 ,0.0013661, 0],
              [0, 0, 0.005]])

# Measurement matrix H_k
H_k = np.array([[1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0]])

# Sensor measurement noise covariance matrix R_k
R_k= np.array([[0.1, 0, 0],
               [0, 0.1, 0],
               [0, 0, 0.1]]) #TO BE CHANGED AFTER MEASUREMENTS

# Sensor noise
sensor_noise_w_k = np.array([0, 0, 0]) #zero mean noise

# Initial state covariance matrix
P_k_minus_1 = np.array([[0.1, 0, 0],
                        [0, 0.1, 0],
                        
                        [0, 0, 0.1]])

######################################################################################################################
#Function to get B matrix
def getB(yaw, deltak):
    B = np.array([[np.cos(yaw) * deltak, 0],
                  [np.sin(yaw) * deltak, 0],
                  [0, deltak]])
    return B
######################################################################################################################
#Kalman Filter

def ekf(z_k_observation_vector, state_estimate_k_minus_1, 
        control_vector_k_minus_1, P_k_minus_1, dk, camera_obstructed):
    # PREDICT
    
    state_estimate_k_priori = A_k_minus_1 @ state_estimate_k_minus_1 + \
                              getB(state_estimate_k_minus_1[2], dk) @ control_vector_k_minus_1 + \
                              process_noise_v_k_minus_1
    state_estimate_k_priori[2] = state_estimate_k_priori[2]% (2*np.pi )
             
    # Predict state covariance
    P_k = A_k_minus_1 @ P_k_minus_1 @ A_k_minus_1.T + Q_k
         
    # MEASUREMENT residual
    measurement_residual_y_k = z_k_observation_vector - (H_k @ state_estimate_k_priori + sensor_noise_w_k)
             
    # Measurement residual covariance
    S_k = H_k @ P_k @ H_k.T + R_k
         
    # Kalman gain
    if camera_obstructed == 1:
        K_k = np.zeros((3,3))
    else:
        K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)
         
    # UPDATE state estimate
    state_estimate_k = state_estimate_k_priori + (K_k @ measurement_residual_y_k)
     
    # Update state covariance
    P_k = P_k - (K_k @ H_k @ P_k)
     
    return state_estimate_k, P_k