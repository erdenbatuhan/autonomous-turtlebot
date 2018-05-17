import vm
from environment import Environment


def main():
    env = Environment()
    env.reset_base()

    state = env.get_state()  # Initial state

    connector = vm.VMConnector()
    server = vm.VMServer()
    server.listen()

    connector.send_data(state)
    action = server.receive_data()

    while action != -4:
        if action == -3:
            env.reset_base()

            state = env.get_state()  # Initial state
            connector.send_data(state)
        else:
            next_state, reward, terminal, crashed = env.act(action)
            connector.send_data((next_state, reward, terminal, crashed))

        action = server.receive_data()


if __name__ == '__main__':
    main()

