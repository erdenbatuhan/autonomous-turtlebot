import time
import util
import rospy
import cv2
import numpy as np
import image_preprocessor as ipp

import message_filters
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from kobuki_msgs.msg import BumperEvent
from cv_bridge import CvBridge, CvBridgeError


class Environment:

    ROSPY_FREQUENCY = 10
    MAX_CRASH_COUNT = 10

    def __init__(self, base_name):
        rospy.init_node("Environment", anonymous=False)
        rospy.loginfo("CTRL + C to terminate..")
        rospy.on_shutdown(self.shutdown)

        self.vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
        self.rate = rospy.Rate(self.ROSPY_FREQUENCY)
        self.initial_time = time.time()
        self.base_name = base_name
        
        # Create the cv_bridge objects
        self.rgb_bridge = CvBridge()
        self.depth_bridge = CvBridge()

        self.image = None
        self.depth_image = None

        self.has_green = False
        self.terminal = False
        self.crashed = False

        self.crash_counter = 0
        self.history = np.zeros(self.MAX_CRASH_COUNT)

        # self.subscribe()

        self.subscribe_rgb_image_raw()
        # self.subscribe_depth_image_raw()
        self.subscribe_bumper_event()

        self.wait_for_image()

    def shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.vel_pub.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

    def wait_for_image(self):
        while self.image is None and self.depth_image is None:
            pass

    def callback(self, image_raw, depth_image_raw):
    	_image, _depth_image = None, None
    	self.image, self.depth_image = None, None

        try:
            _image = self.rgb_bridge.imgmsg_to_cv2(image_raw, "bgr8")
            _depth_image = self.depth_bridge.imgmsg_to_cv2(depth_image_raw, "32FC1")

            if np.average(_image) > 1.:
            	self.image = np.array(_image, dtype=np.uint8)

            if np.average(_depth_image) < 1.:
            	self.depth_image = np.array(_depth_image, dtype=np.float32) / 255.0
        except CvBridgeError as e:
            print(e)

    def subscribe(self):
    	image_raw = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        depth_image_raw = message_filters.Subscriber("/camera/depth/image_raw", Image)

        ts = message_filters.TimeSynchronizer([image_raw, depth_image_raw], 100)
        ts.registerCallback(self.callback)

    def rgb_image_raw_callback(self, rgb_image_raw):
        self.image = None

        try:
            self.image = self.rgb_bridge.imgmsg_to_cv2(rgb_image_raw, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.image = np.array(self.image, dtype=np.uint8)

    def subscribe_rgb_image_raw(self):
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_image_raw_callback)

    def depth_image_raw_callback(self, depth_image_raw):
        self.depth_image = None

        try:
            self.depth_image = self.depth_bridge.imgmsg_to_cv2(depth_image_raw, "32FC1")
        except CvBridgeError as e:
            print(e)

        self.depth_image = np.array(self.depth_image, dtype=np.float32)
        self.depth_image = self.depth_image / 255.0

    def subscribe_depth_image_raw(self):
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_image_raw_callback)

    def bumper_event_callback(self, bumper_event):
        self.crashed = True if bumper_event.bumper == 1 else False

    def subscribe_bumper_event(self):
        rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.bumper_event_callback)

    def get_state(self):
        try:
            curr_image = self.image
            
            frame = ipp.preprocess_image(curr_image)
            mask, self.has_green, self.terminal = ipp.get_greens(curr_image)

            if self.crash_counter < self.MAX_CRASH_COUNT:
                self.history[self.crash_counter] = np.average(frame)
                self.crash_counter = self.crash_counter + 1
            else:
                self.crash_counter = 0
                if -0.01 <= np.average(self.history) - np.average(frame) <= 0.01:
                    self.crashed = True
            
            state = np.array([np.array([frame, mask])], dtype=np.uint8)
            return state
        except cv2.error as e:
            return np.zeros((1, 2, 80, 80), dtype=uint8)

    def get_reward(self):
    	if self.terminal:
    		return 100
        elif self.crashed:
            return -1
        elif self.has_green:
        	return 0.1

        return 0

    def act(self, action, v1=0.3, v2=0.05):
        vel_cmd = Twist()

        if action == 0:  # LEFT
            vel_cmd.linear.x = v1 - v2
            vel_cmd.angular.z = 2. * v1
        elif action == 1:  # FORWARD - LEFT
            vel_cmd.linear.x = 1.5 * (v1 - v2)
            vel_cmd.angular.z = v1
        elif action == 2:  # FORWARD - AHEAD
            vel_cmd.linear.x = 2.5 * (v1 - v2)
            vel_cmd.angular.z = 0.
        elif action == 3:  # FORWARD - RIGHT
            vel_cmd.linear.x = 1.5 * (v1 - v2)
            vel_cmd.angular.z = -v1
        elif action == 4:  # RIGHT
            vel_cmd.linear.x = v1 - v2
            vel_cmd.angular.z = -2. * v1

        if rospy.is_shutdown():
            return

        self.vel_pub.publish(vel_cmd)
        self.rate.sleep()

        self.image = None
        self.wait_for_image()

        state = self.get_state()
        reward = self.get_reward()

        print("Has Green? {} | Terminal {} | Crashed {} | Reward {} | Act {}".
        	format(self.has_green, self.terminal, self.crashed, reward, (action - 2)))
        return state, reward, self.terminal, self.crashed

    @staticmethod
    def reset_model_state(model_state):
        model_state.pose.position.x = 3.
        model_state.pose.position.y = -3.
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

        self.has_green = False
        self.terminal = False
        self.crashed = False

        self.image = None
        self.wait_for_image()

        self.initial_time = time.time()

