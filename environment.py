import math
import util
import rospy
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


class Environment:

    ROSPY_FREQUENCY = 10

    def __init__(self, base_name):
        rospy.init_node("Environment", anonymous=False)
        rospy.loginfo("CTRL + C to terminate..")
        rospy.on_shutdown(self.shutdown)

        self.vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
        self.rate = rospy.Rate(self.ROSPY_FREQUENCY)

        self.base_name = base_name
        self.terminal = False

        self.image = None

        self.subscribe_rgb_image_raw()
        self.wait_for_image()

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

    def observe(self):
        state = util.process_image(self.image)

        if math.fabs(state) == self.image.shape[1] / 2:
            self.terminal = True

        return np.array([util.process_image(self.image)])

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

        self.image = None
        self.wait_for_image()

        state = self.observe()
        reward = -1 if self.terminal else 0

        print("State {} | Reward {} | Act {}".format(state, reward, action))
        return state, reward, self.terminal

    @staticmethod
    def reset_model_state(model_state):
        model_state.pose.position.x = 0.
        model_state.pose.position.y = 0.
        model_state.pose.position.z = 0.
        model_state.pose.orientation.x = 0.
        model_state.pose.orientation.y = 0.
        model_state.pose.orientation.z = 0.
        model_state.pose.orientation.w = 0.
        model_state.twist.linear.x = 0.
        model_state.twist.linear.y = 0.
        model_state.twist.linear.z = 0.
        model_state.twist.angular.x = 0.
        model_state.twist.angular.y = 0.
        model_state.twist.angular.z = 0.

    def reset_base(self):
        rospy.wait_for_service("/gazebo/set_model_state")
        set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        model_state = ModelState()
        model_state.model_name = self.base_name

        self.reset_model_state(model_state)
        set_model_state(model_state)

        self.shutdown()
        self.terminal = False

        self.wait_for_image()

