
from tdmclient import ClientAsync, aw

def read_acc_sensors(node,client):
    aw(node.wait_for_variables({"acc"}))
    aw(client.sleep(0.01))
    acc=list(node.v.acc)
    acc_vert = acc[2]
    return acc_vert