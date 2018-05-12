import math
import util
import rospy
import cv2
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


class Environment:

    ROSPY_FREQUENCY = 10

    def __init__(self):
        rospy.init_node("Environment", anonymous=False)
        rospy.loginfo("CTRL + C to terminate..")

        rospy.on_shutdown(self.shutdown)

        self.vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
        self.rate = rospy.Rate(self.ROSPY_FREQUENCY)

    def reset_base(self):
        self.shutdown()

    def shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.vel_pub.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

    @staticmethod
    def observe():
        terminal = False

        image = util.capture_image()
        state = util.process_image(image)

        if math.fabs(state) == image.shape[1] / 2:
            terminal = True

        return np.array([state]), terminal, (image.shape[1] / 2)

    @staticmethod
    def get_reward(state, terminal, radius):
        if terminal:
            return -1

        return 1 - (math.fabs(state) / radius)

    def act(self, action, v1=0.3):
        vel_cmd = Twist()

        if action == 0:
            vel_cmd.angular.z = v1
        elif action == 1:
            vel_cmd.angular.z = v1 / 2
        elif action == 2:
            vel_cmd.angular.z = -v1 / 2
        elif action == 3:
            vel_cmd.angular.z = -v1

        if rospy.is_shutdown():
            return

        self.vel_pub.publish(vel_cmd)
        self.rate.sleep()

        state, terminal, radius = self.observe()
        reward = self.get_reward(state, terminal, radius)

        print("State {} | Reward {} | Act {}".format(state, reward, action))
        return state, reward, terminal

