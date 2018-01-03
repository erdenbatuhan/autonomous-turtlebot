from environment import Environment
import util
import vm


BASE_NAME = "mobile_base"
DESTINATION = {"x": 4.93, "y": 1.00}


def main():
    env = Environment(base_name=BASE_NAME, destination=DESTINATION)
    env.reset_base()

    state = env.get_state()  # Initial state

    if state["greedy"] != 1.:
        print("Expected initial distance (1.), got (%f). Re-running the simulation.." % state["greedy"])
        return main()  # Stop simulation

    connector = vm.VMConnector()
    server = vm.VMServer()
    server.listen()

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

    return None


if __name__ == '__main__':
    main()

