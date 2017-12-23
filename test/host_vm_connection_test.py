import numpy as np
import host


def get_action_using(depth):
    decision = np.zeros(3)

    decision[0] = np.average(depth[:, 2:6])  # Middle
    decision[1] = np.average(depth[:, 0:2])  # Left
    decision[2] = np.average(depth[:, 6:8])  # Right

    return np.argmax(decision)


def main():
    server = host.HostServer()
    connector = host.HostConnector()
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
        action = str(get_action_using(depth))

        print("------------------\n")
        print(action)

        connector.send_data([action])


if __name__ == '__main__':
    main()

