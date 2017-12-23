import numpy as np
from random import randint
from environment import Environment


def get_action_using(depth):
    decision = np.zeros(3)

    decision[0] = np.average(depth[:, 0:2])  # Left
    decision[1] = np.average(depth[:, 2:6])  # Middle
    decision[2] = np.average(depth[:, 6:8])  # Right

    return np.argmax(decision)


if __name__ == '__main__':
    env = Environment(base_name="mobile_base", destination_name="unit_sphere_3")
    env.reset_base()

    destination_point = env.destination
    #  Act randomly
    while destination_point == env.destination:
        random_action = randint(0, 2)
        env.act(random_action)
        print(env.depth_image_raw)
        print("---")
        print("Random Action", random_action)
        print("Position", env.position)
        print("Destination", env.destination)
        print("---")

    """
    #  Act according to depth values
    random_action = randint(0, 2)
    env.act(0)
    while destination_point == env.destination:
        state = env.get_state()
        action = get_action_using(state[1])
        
        env.act(action)
        print(action)
        print("-------------------------------------------")
    """

    print("Destination reached!")
