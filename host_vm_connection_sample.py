import HostMachine
import numpy as np


def decide_action_based_on_depth(depth):
    decision = np.zeros(3)
    decision[0] = np.average(depth[:, 2:6])  # Middle
    decision[1] = np.average(depth[:, 0:2])  # Left
    decision[2] = np.average(depth[:, 6:8])  # Right

    return np.argmax(decision)


server = HostMachine.HostServer()
connector = HostMachine.HostConnector()
server.listen()

while True:
    depth = server.receive_data()
    action = str(decide_action_based_on_depth(depth))
    print("------------------\n")
    print(action)
    connector.send_data([action])