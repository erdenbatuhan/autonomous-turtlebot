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
    state = server.receive_data()  # [Distance, Depth, TimePassed]
    print("Distance")
    print(state[0])
    print("Depth")
    print(state[1])
    print("Time")
    print(state[2])
    depth = state[1]
    action = str(decide_action_based_on_depth(depth))
    print("------------------\n")
    print(action)
    connector.send_data([action])