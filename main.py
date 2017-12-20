from random import randint
from environment import Environment


if __name__ == '__main__':
    env = Environment(base_name="mobile_base", destination_name="unit_sphere_3")
    env.reset_base()

    destination_point = env.destination
    while destination_point == env.destination:
        random_action = randint(0, 2)
        env.act(random_action)

        print("---")
        print("Random Action", random_action)
        print("Position", env.position)
        print("Destination", env.destination)
        print("---")

    print("Destination reached!")

