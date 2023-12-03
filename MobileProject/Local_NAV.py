from tdmclient import ClientAsync, aw

obstThrL = 2500      
obstThrH = 3700
spRight = 0
spLeft = 0
speed_L_R = [0,0]
obstSpeedGain = [6, 8, -15, -8, -6] #version de Gab
#obstSpeedGain =  [4, 2, -15, -4, -6] #version le cours
speed0 = 80


############################################################################################################################################

def update_state(state,obst,client):
    if state == 1: #State = 0 -> Global path
        if (obst[0] > obstThrH):
            state = 2
        if (obst[1] > obstThrH):
            state = 2
        if (obst[2] > obstThrH):
            state = 2
        if (obst[3] > obstThrH):
            state = 2
        if (obst[4] > obstThrH):
            state = 2
    elif state == 2: #State = 1 -> Normal local navigation -> obstacles pop far from the thymio -> normal avoidance
        if obst[0] < obstThrL:
            if obst[1] < obstThrL:
                if obst[2] < obstThrL:
                    if obst[3] < obstThrL:
                        if obst[4] < obstThrL:
                            state = 1
    return state
    
############################################################################################################################################    
def local_navigation(obst):
    global spLeft, spRight, speed_L_R, obstSpeedGain,speed0
    spLeft = speed0
    spRight = speed0
    for i in range(5):
        spLeft += obst[i] * obstSpeedGain[i] // 300
        spRight += obst[i] * obstSpeedGain[4 - i] // 300
    speed_L_R = [spLeft,spRight]
    return speed_L_R

############################################################################################################################################

def read_prox_sensors(node,client):
    aw(node.wait_for_variables({"prox.horizontal"}))
    aw(client.sleep(0.01))
    prox=node.v.prox.horizontal
    return prox

############################################################################################################################################
