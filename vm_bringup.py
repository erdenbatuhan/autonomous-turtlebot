from environment import Environment
import vm


BASE_NAME = "mobile_base"
DESTINATION_NAME = "bowl"


def main():
    env = Environment(base_name=BASE_NAME, destination_name=DESTINATION_NAME)
    env.reset_base()

    connector = vm.VMConnector()
    server = vm.VMServer()
    server.listen()

    state = env.get_state()  # Initial state

    if state["greedy"] != 1.:
        print("Expected initial distance (1.), got (%.2f).. Please reset the simulation!!" % state["greedy"])

        connector.send_data(None)  # Stop training
        return  # Stop simulation

    connector.send_data(state)
    action = server.receive_data()

    while action != -2:
        if action == -1:
            env.reset_base()

            state = env.get_state()  # Initial state
            connector.send_data(state)
        else:
            next_state, reward, terminal, crashed = env.act(action)
            connector.send_data((next_state, reward, terminal, crashed))

        action = server.receive_data()


if __name__ == '__main__':
    main()

