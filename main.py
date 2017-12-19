import rospy
from scripts.environment import Environment


if __name__ == '__main__':
    try:
        env = Environment()

        env.act(0)
        env.act(0)

        env.act(1)
        env.act(1)

        env.act(0)
        env.act(0)

        env.act(2)
        env.act(2)

        env.act(0)
        env.act(0)
        env.act(0)
        env.act(0)
    except:
        rospy.loginfo("Terminated!")

