from random import randint
from environment import Environment


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
        depth = env.depth_image_raw
        action = env.decide_action_based_on_depth(depth)
        env.act(action)
        print(action)
        print("-------------------------------------------")
    """

    print("Destination reached!")
