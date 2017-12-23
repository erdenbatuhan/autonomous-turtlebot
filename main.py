from random import randint
from environment import Environment
import vm


BASE_NAME = "mobile_base"
DESTINATION_NAME = "unit_sphere_3"
EPOCH = 1000


def main():
    env = Environment(base_name=BASE_NAME, destination_name=DESTINATION_NAME)
    env.reset_base()

    conn = vm.VMConnector()
    server = vm.VMServer()
    server.listen()

    state = env.get_state()
    conn.send_data(state)

    # Act randomly
    for _ in range(EPOCH):
        random_action = randint(0, 2)
        next_state, reward, terminal = env.act(random_action)

        conn.send_data(next_state)
        action = server.receive_data()

        env.act(int(action[0]))

        if terminal:
            print("Destination Reached!")
            break


if __name__ == '__main__':
    main()

