import vm
from environment import Environment


def main():
    env = Environment()
    env.reset_base()

    state, _ = env.observe()  # Initial state

    connector = vm.VMConnector()
    server = vm.VMServer()
    server.listen()

    connector.send_data(state)
    action = server.receive_data()

    while action != -2:
        if action == -1:
            env.reset_base()

            state, _ = env.observe()  # Initial state
            connector.send_data(state)
        else:
            next_state, reward, terminal = env.act(action)
            connector.send_data((next_state, reward, terminal))

        action = server.receive_data()


if __name__ == '__main__':
    main()

