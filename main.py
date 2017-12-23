from random import randint
from environment import Environment
import VM

if __name__ == '__main__':
    env = Environment(base_name="mobile_base", destination_name="unit_sphere_3")
    env.reset_base()
    con = VM.VMConnector()
    server = VM.VMServer()
    server.listen()
    destination_point = env.destination
    random_action = randint(0, 2)
    env.act(0)

    while destination_point == env.destination:
        depth = env.depth_image_raw
        con.send_data(depth)
        action = server.receive_data()
        env.act(int(action[0]))
        print("-------------------------------------------")
    """
    
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
    print("Destination Reached !")