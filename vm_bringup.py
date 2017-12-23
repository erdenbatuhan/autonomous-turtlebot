from random import randint
from matplotlib import pyplot as plt
from environment import Environment
import vm


BASE_NAME = "mobile_base"
DESTINATION_NAME = "unit_sphere_3"
EPOCH = 100000


def plot_learning_curve(episodes):
    plt.plot(episodes)
    plt.xlabel("Episode")
    plt.ylabel("Length of episode")
    plt.show()


def main():
    env = Environment(base_name=BASE_NAME, destination_name=DESTINATION_NAME)
    env.reset_base()

    # conn = vm.VMConnector()
    # server = vm.VMServer()
    # server.listen()

    state = env.get_state()
    # conn.send_data(state)
    # action = server.receive_data()

    # Act randomly
    for _ in range(EPOCH):
        action = randint(0, 2)
        next_state, reward, terminal = env.act(action)

        # conn.send_data(next_state)
        # action = server.receive_data()

        if terminal:
            print "Destination Reached!"
            break


if __name__ == '__main__':
    main()

