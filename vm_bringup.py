from environment import Environment
import vm


BASE_NAME = "mobile_base"
DESTINATION_NAME = "unit_sphere_3"


def main():
    env = Environment(base_name=BASE_NAME, destination_name=DESTINATION_NAME)
    env.reset_base()

    connector = vm.VMConnector()
    server = vm.VMServer()
    server.listen()

    state = env.flatten(env.get_state())  # Initial state
    connector.send_data(state)
    action = server.receive_data()

    while action != -2:
        if action == -1:
            env.reset_base()

            state = env.flatten(env.get_state())  # Initial state
            connector.send_data(state)
        else:
            next_state, reward, terminal, crashed = env.act(action)
            connector.send_data((next_state, reward, terminal, crashed))

        action = server.receive_data()


if __name__ == '__main__':
    main()

