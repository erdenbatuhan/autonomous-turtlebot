import time
import math
import rospy
import numpy as np

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from kobuki_msgs.msg import BumperEvent
from cv_bridge import CvBridge, CvBridgeError


class Environment:

    __STATE_DIM = 1 + 8 * 8 + 1
    __FREQUENCY = 10
    __TIME_LIMIT = 10 ** 5
    
    def __init__(self, base_name, destination_name):
        rospy.init_node("Environment", anonymous=False)
        rospy.loginfo("CTRL + C to terminate..")   
        rospy.on_shutdown(self.__shutdown)

        self.__vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
        self.__rate = rospy.Rate(self.__FREQUENCY)
        self.__bridge = CvBridge()
        self.__initial_time = time.time()
        self.__base_name = base_name
        self.__destination_name = destination_name
        self.__position = {"x": 0., "y": 0.}
        self.__destination = {"x": -999., "y": -999.}
        self.__depth_image_raw = None
        self.__terminal = False
        self.__crashed = False

        self.__subscriptions_ready = np.zeros(2)
        self.__subscribe_model_states()
        self.__subscribe_depth_image_raw()
        self.__subscribe_bumper_event()

        # Wait for the first image to be taken
        while np.sum(self.__subscriptions_ready) < 2:
            pass

        self.__initial_distance = self.__get_distance_between(self.__position, self.__destination)

    @staticmethod
    def __get_angle_between(p1, p2):
        y = p2["y"] - p1["y"]
        x = p2["x"] - p1["x"]

        return math.atan2(y, x)

    @staticmethod
    def __get_index_of(arr, item):
        for i in range(len(arr)):
            if arr[i] == item:
                return i

        return -1

    @staticmethod
    def __minimize(image):
        mini_depth = np.zeros((8, 8))

        for i in range(0, 8):
            for j in range(0, 8):
                x = i * 60
                y = j * 80

                temp_array = image[x:x + 60, y:y + 80]
                mini_depth[i][j] = np.average(temp_array)

                if np.isnan(mini_depth[i][j]):
                    mini_depth[i][j] = 10.

        return mini_depth

    def __shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.__vel_pub.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

    def __model_states_callback(self, model_states):
        base_ind = self.__get_index_of(model_states.name, self.__base_name)
        destination_ind = self.__get_index_of(model_states.name, self.__destination_name)

        position = model_states.pose[base_ind].position
        destination = model_states.pose[destination_ind].position

        self.__position["x"] = position.x
        self.__position["y"] = position.y

        self.__destination["x"] = destination.x
        self.__destination["y"] = destination.y

        self.__subscriptions_ready[0] = 1

    def __depth_image_raw_callback(self, depth_image_raw):
        try:
            self.__depth_image_raw = self.__bridge.imgmsg_to_cv2(depth_image_raw, "32FC1")
        except CvBridgeError as e:
            print(e)

        self.__depth_image_raw = np.array(self.__depth_image_raw, dtype=np.float32)
        self.__subscriptions_ready[1] = 1

    def __bumper_event_callback(self, bumper_event):
        self.__crashed = True if bumper_event.bumper == 1 else False

    def __subscribe_model_states(self):
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.__model_states_callback)

    def __subscribe_depth_image_raw(self):
        rospy.Subscriber("/camera/depth/image_raw", Image, self.__depth_image_raw_callback)

    def __subscribe_bumper_event(self):
        rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.__bumper_event_callback)

    def __get_distance_between(self, p1, p2):
        a, b = p2["x"] - p1["x"], p2["y"] - p1["y"]
        c = math.sqrt(a ** 2 + b ** 2)

        if c < .1:
            self.__terminal = True

        return c

    def get_state(self):
        # S(t) = (distance(t), depth(t), time_passed(t))
        state = np.zeros(self.__STATE_DIM)
        last_element = len(state) - 1

        distance = self.__get_distance_between(self.__position, self.__destination) / self.__initial_distance
        depth = self.__minimize(self.__depth_image_raw)
        time_passed = time.time() - self.__initial_time

        state[0] = distance
        state[1:last_element] = depth.reshape(1, -1)
        state[last_element] = time_passed

        return state  # Shape = (66, )

    def __get_direct_reward(self, state):
        if self.__terminal:
            return 200
        elif self.__crashed or state[2] > self.__TIME_LIMIT:
            return -100

        return 0

    def get_reward(self, state):
        last_element = len(state) - 1
        state = state[0], state[1:last_element], state[last_element]

        c = [-20, 3, -1]  # coefficients for each state element (distance, depth, time_passed)
        direct_reward = self.__get_direct_reward(state)

        if direct_reward != 0:
            self.reset_base()
            return direct_reward

        reward = sum([state[i] * c[i] for i in range(len(state))])  # 8x8 Reward
        mini_reward = np.array(3, dtype=np.float32)

        mini_reward[0] = np.average(reward[:, 0:2])  # LEFT
        mini_reward[1] = np.average(reward[:, 2:6])  # FORWARD
        mini_reward[2] = np.average(reward[:, 6:8])  # RIGHT

        return mini_reward  # 1x3 Reward

    def act(self, action, v1=.3, v2=.05, duration=10):
        vel_cmd = Twist()

        if action == 0:  # LEFT
            vel_cmd.linear.x = v2
            vel_cmd.angular.z = v1
        elif action == 1:  # FORWARD
            vel_cmd.linear.x = v1 - v2
            vel_cmd.angular.z = 0.
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = v2
            vel_cmd.angular.z = -v1

        for _ in range(duration):
            if rospy.is_shutdown():
                return

            self.__vel_pub.publish(vel_cmd)
            self.__rate.sleep()

            if self.__crashed:
                break

        state = self.get_state()
        reward = self.get_reward(state)

        return state, reward, self.__terminal

    def reset_base(self):
        rospy.wait_for_service("/gazebo/set_model_state")
        set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        model_state = ModelState()
        model_state.model_name = self.__base_name

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

        set_model_state(model_state)

        self.__initial_time = time.time()
        self.__crashed = False

