import vm
import util
from environment import Environment


BASE_NAME = "mobile_base"
DESTINATION = {"x": 3., "y": 1.}


def main():
    env = Environment(base_name=BASE_NAME, destination=DESTINATION)
    env.reset_base()

    state = env.get_state()  # Initial state

    c = util.c(state[0], state[1])
    error_rate = util.to_precision(1. - c / env.initial_distance, 2)

    if error_rate != 0:
        print("Expected initial distance (1.), got (%f). Re-running the simulation.." % (1. - error_rate))
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


if __name__ == '__main__':
    main()

