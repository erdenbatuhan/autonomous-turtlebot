import time
import util
import rospy
import cv2
import numpy as np
import image_preprocessor as ipp

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Environment:

    ROSPY_FREQUENCY = 10
    NUM_OF_SUBSCRIPTIONS = 1

    def __init__(self):
        rospy.init_node("Environment", anonymous=False)
        rospy.loginfo("CTRL + C to terminate..")

        rospy.on_shutdown(self.shutdown)

        self.vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
        self.rate = rospy.Rate(self.ROSPY_FREQUENCY)

        self.bridge = CvBridge()
        self.initial_time = time.time()

        self.depth_image_raw = None
        self.terminal = False
        self.crashed = False

        self.subscriptions_ready = np.zeros(self.NUM_OF_SUBSCRIPTIONS)
        self.subscribe_depth_image_raw()

        self.wait_for_subscriptions()

    def shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.vel_pub.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

    def wait_for_subscriptions(self):
        while np.sum(self.subscriptions_ready) < self.NUM_OF_SUBSCRIPTIONS:
            pass

    def depth_image_raw_callback(self, depth_image_raw):
        try:
            self.depth_image_raw = self.bridge.imgmsg_to_cv2(depth_image_raw, "passthrough")
            self.depth_image_raw = np.array(self.depth_image_raw, dtype=np.float32)

            cv2.normalize(self.depth_image_raw, self.depth_image_raw, 0, 1, cv2.NORM_MINMAX)
            self.subscriptions_ready[0] = 1
        except CvBridgeError as e:
            print(e)

    def subscribe_depth_image_raw(self):
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_image_raw_callback)

    def get_state(self):
        # Check if crashed
        depth = ipp.preprocess_image(self.depth_image_raw)
        
        if np.average(np.array(depth[0][0])) <= 0.05 / 255:
            self.crashed = True

        return depth

    def get_reward(self):
        if self.crashed:
            return -1

        return 0.1

    def act(self, action, v1=0.3, v2=0.05):
        vel_cmd = Twist()

        if action == 0:  # LEFT
            vel_cmd.linear.x = v1 - v2
            vel_cmd.angular.z = 2. * v1
        elif action == 1:  # FORWARD - LEFT
            vel_cmd.linear.x = v1 - v2
            vel_cmd.angular.z = v1
        elif action == 2:  # FORWARD - AHEAD
            vel_cmd.linear.x = 2. * (v1 - v2)
            vel_cmd.angular.z = 0.
        elif action == 3:  # FORWARD - RIGHT
            vel_cmd.linear.x = v1 - v2
            vel_cmd.angular.z = -v1
        elif action == 4:  # RIGHT
            vel_cmd.linear.x = v1 - v2
            vel_cmd.angular.z = -2. * v1
        elif action == 5:  # BACK
            vel_cmd.linear.x = -5. * (v1 - v2)
            vel_cmd.angular.z = 0.

        if rospy.is_shutdown():
            return

        self.vel_pub.publish(vel_cmd)
        self.rate.sleep()

        self.subscriptions_ready = np.zeros(self.NUM_OF_SUBSCRIPTIONS)
        self.wait_for_subscriptions()

        state = self.get_state()
        reward = self.get_reward()

        print("State {} | Reward {} | Act {}".format(state, reward, (action - 2)))
        return state, reward, self.terminal, self.crashed

    def reset_base(self):
        self.shutdown()

        self.terminal = False
        self.crashed = False

        self.subscriptions_ready = np.zeros(self.NUM_OF_SUBSCRIPTIONS)
        self.wait_for_subscriptions()

        self.initial_time = time.time()

