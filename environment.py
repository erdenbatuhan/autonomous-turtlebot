import time
import util
import rospy
import cv2
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from create_node.msg import TurtlebotSensorState
from sensor_msgs.msg import PointCloud2
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

        self.points = None
        self.image = None
        self.depth_image_raw = None
        self.terminal = False
        self.crashed = False

        self.subscriptions_ready = np.zeros(self.NUM_OF_SUBSCRIPTIONS)
        #self.subscribe_rgb_image_raw()
        self.subscribe_depth_image_raw()
        self.subscribe_core_sensors()
        #self.subscribe_point_cloud()

        self.wait_for_subscriptions()

    def shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.vel_pub.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

    def wait_for_subscriptions(self):
        while np.sum(self.subscriptions_ready) < self.NUM_OF_SUBSCRIPTIONS:
            pass

    def rgb_image_raw_callback(self, rgb_image_raw):
        self.image = None

        try:
            self.image = self.bridge.imgmsg_to_cv2(rgb_image_raw, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.image = np.array(self.image, dtype=np.uint8)
        self.subscriptions_ready[0] = 1

    def subscribe_rgb_image_raw(self):
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_image_raw_callback)

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

    def core_sensor_callback(self, sensor_state):
        bumper_state = sensor_state.bumps_wheeldrops
        self.crashed = True if bumper_state is not 0 else False

    def subscribe_core_sensors(self):
        rospy.Subscriber("/mobile_base/sensors/core", TurtlebotSensorState, self.core_sensor_callback)

    def point_cloud_callback(self, cloud_msg):
        dtype_list = [(f.name, np.float32) for f in cloud_msg.fields]
        self.points = np.fromstring(cloud_msg.data, dtype_list)
        self.points = np.reshape(self.points, (640, 640))
        self.points = np.resize(self.points, (80, 80))

        for i in range(0, 80):
            for j in range(0, 80):
                if np.isnan(self.points[0][0][i][j]):
                    self.points[0][0][i][j] = 0

        self.subscriptions_ready[0] = 1

    def subscribe_point_cloud(self):
        rospy.Subscriber("/camera/depth/points", PointCloud2, self.point_cloud_callback)

    def get_state(self):
        '''
        image = np.zeros((80, 80))

        try:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            image = util.preprocess_image(image)
        except cv2.error as e:
            print(e)

        return np.array([np.array([image])])


        image = np.zeros((80, 80))

        try:
            image = util.capture_image()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = util.preprocess_image(image)
        except cv2.error as e:
            print(e)
        '''

        depth = util.preprocess_image(self.depth_image_raw)
        state = np.array([np.array([depth])])

        return state

        #return np.array([np.array([self.points])])

    def get_reward(self):
        if self.crashed:
            return -1

        return 0

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
            vel_cmd.linear.x = -30. * (v1 - v2)
            vel_cmd.angular.z = -5. * v1

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

