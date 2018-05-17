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

        self.image = None
        # self.subscribe_rgb_image_raw()
        # self.wait_for_image()

    def reset_base(self):
        self.shutdown()

    def shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.vel_pub.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

    def wait_for_image(self):
        while self.image is None:
            pass

    def rgb_image_raw_callback(self, rgb_image_raw):
        self.image = None

        try:
            self.image = CvBridge().imgmsg_to_cv2(rgb_image_raw, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.image = np.array(self.image, dtype=np.uint8)

    def subscribe_rgb_image_raw(self):
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_image_raw_callback)

    @staticmethod
    def observe():
        # self.wait_for_image()
        # try:
        #     image = self.image
        # except cv2.error:
        #    image = np.zeros((80, 80))

        terminal = False

        try:
            image = util.capture_image()
        except cv2.error as e:
            print(e)
            image = np.zeros((80, 80))

        state = util.process_image(image)

        if math.fabs(state) == image.shape[1] / 2:
            terminal = True

        return np.array([state]), terminal

    @staticmethod
    def get_reward(state, terminal):
        if terminal:
            return -1
        elif -20 <= state <= 20:
            if -5 <= state <= 5:
                return 0.75
            return 0.25

        return 0

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
        else:
            vel_cmd.linear.x = (v1 - v2)  # Forward

        if rospy.is_shutdown():
            return

        self.vel_pub.publish(vel_cmd)
        self.rate.sleep()

        state, terminal = self.observe()
        reward = self.get_reward(state, terminal)

        print("State {} | Reward {} | Act {}".format(state, reward, action))
        return state, reward, terminal

