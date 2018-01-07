import time
import math
import util
import rospy
import numpy as np

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Environment:

    __FREQUENCY = 10

    def __init__(self, base_name, destination):
        rospy.init_node("Environment", anonymous=False)
        rospy.loginfo("CTRL + C to terminate..")
        rospy.on_shutdown(self.__shutdown)

        self.__vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
        self.__rate = rospy.Rate(self.__FREQUENCY)
        self.__bridge = CvBridge()
        self.__initial_time = time.time()
        self.__base_name = base_name
        self.__position = {"x": 0., "y": 0.}
        self.__destination = destination
        self.__depth_image_raw = None
        self.__terminal = False
        self.__crashed = False

        self.__subscriptions_ready = np.zeros(2)
        self.__subscribe_model_states()
        self.__subscribe_depth_image_raw()

        self.__wait_for_subscriptions()
        self.__initial_distance, _, _ = util.get_distance_between(self.__position, self.__destination)

    def __shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.__vel_pub.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

    def __wait_for_subscriptions(self):
        while np.sum(self.__subscriptions_ready) < 2:
            pass

    def __model_states_callback(self, model_states):
        base_ind = util.get_index_of(model_states.name, self.__base_name)
        position = model_states.pose[base_ind].position

        # destination_ind = util.get_index_of(model_states.name, self.__destination_name)
        # destination = model_states.pose[destination_ind].position

        self.__position["x"] = position.x
        self.__position["y"] = position.y

        self.__subscriptions_ready[0] = 1

    def __depth_image_raw_callback(self, depth_image_raw):
        try:
            self.__depth_image_raw = self.__bridge.imgmsg_to_cv2(depth_image_raw, "32FC1")
        except CvBridgeError as e:
            print(e)

        self.__depth_image_raw = np.array(self.__depth_image_raw, dtype=np.float32)
        self.__subscriptions_ready[1] = 1

    def __subscribe_model_states(self):
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.__model_states_callback)

    def __subscribe_depth_image_raw(self):
        rospy.Subscriber("/camera/depth/image_raw", Image, self.__depth_image_raw_callback)

    @staticmethod
    def __get_depth_minimized(image):
        depth = np.zeros((8, 8), dtype=np.float)
        depth_minimized = np.zeros(3, dtype=np.float)

        for i in range(0, 8):
            for j in range(0, 8):
                x = i * 60
                y = j * 80

                temp_array = image[x:x + 60, y:y + 80]
                depth[i][j] = np.average(temp_array)

                if np.isnan(depth[i][j]) or (not -100. < depth[i][j] < 100.):
                    depth[i][j] = 0

        try:
            depth_minimized[0] = util.to_precision(np.average(depth[:, 0:2]), 2)
            depth_minimized[1] = util.to_precision(np.average(depth[:, 2:6]), 2)
            depth_minimized[2] = util.to_precision(np.average(depth[:, 6:8]), 2)
        except OverflowError:
            depth_minimized = np.zeros(3, dtype=np.float)

        return depth_minimized

    @staticmethod
    def __get_depth_modified(depth):
        count_special = sum([1 if d == .12 else 0 for d in depth])
        if count_special == 1:
            depth = [10. if d == .12 else d for d in depth]

        powers = [1, 1, 1]
        depth_modified = [10 if d > .75 else d for d in depth]

        return [p * d for p, d in zip(powers, depth_modified)]

    def get_state(self):
        # S(t) = (distance(t), depth(t))
        distance, _, self.__terminal = util.get_distance_between(self.__position, self.__destination)
        depth = self.__get_depth_minimized(self.__depth_image_raw)

        # Get distance as percentage
        for i in range(len(distance)):
            distance[i] /= self.__initial_distance[i]

        # Get depth modified
        depth = self.__get_depth_modified(depth)

        # Check if crashed
        if np.min(depth) <= .05:
            self.__crashed = True

        return {
            "greedy": np.array(util.to_precision_all(distance, 2)).reshape((1, -1)),
            "safe": np.array(util.to_precision_all(depth, 2)).reshape((1, -1))
        }

    def get_reward(self, state):
        # c is a value between (0, 1]
        av = np.average(state["safe"])

        return {
            "greedy": 1000 if self.__terminal else 0,
            "safe": -100 if self.__crashed else 0
        }

    def act(self, action, v1=.3, v2=.05):
        vel_cmd = Twist()

        if action == 0:  # LEFT
            vel_cmd.linear.x = v1 - v2
            vel_cmd.angular.z = 2 * v1
        elif action == 1:  # FORWARD
            vel_cmd.linear.x = 3 * (v1 - v2)
            vel_cmd.angular.z = 0.
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = v1 - v2
            vel_cmd.angular.z = -2 * v1

        if rospy.is_shutdown():
            return

        self.__vel_pub.publish(vel_cmd)
        self.__rate.sleep()

        state = self.get_state()
        reward = self.get_reward(state)

        return state, reward, self.__terminal, self.__crashed

    @staticmethod
    def __reset_model_state(model_state):
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
        model_state.model_name = self.__base_name

        self.__reset_model_state(model_state)
        set_model_state(model_state)

        self.__shutdown()

        self.__terminal = False
        self.__crashed = False

        self.__subscriptions_ready = np.zeros(2)
        self.__wait_for_subscriptions()

        self.__initial_time = time.time()

