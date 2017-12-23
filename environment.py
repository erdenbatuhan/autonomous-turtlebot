import time
import math
import numpy as np
import cv2
import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Environment:
    
    def __init__(self, base_name, destination_name, frequency=10):
        rospy.init_node("Environment", anonymous=False)

        rospy.loginfo("CTRL + C to terminate..")   
        rospy.on_shutdown(self.__shutdown)

        self.__base_name = base_name
        self.__destination_name = destination_name

        self.__vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
        self.__rate = rospy.Rate(frequency)

        self.__position = {"x": 0., "y": 0.}
        self.__destination = {"x": -999., "y": -999.}

        self.__bridge = CvBridge()
        self.__depth_image_raw = None

        self.__subscribe_model_states()
        self.__subscribe_depth_image_raw()

        self.__initial_time = time.time()

    @staticmethod
    def __get_distance_between(p1, p2):
        a, b = p2["x"] - p1["x"], p2["y"] - p1["y"]
        c = math.sqrt(a ** 2 + b ** 2)

        return c

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
    def __compress(image):
        depth_avg = np.zeros((8, 8))

        for i in range(0, 8):
            for j in range(0, 8):
                x = i * 60
                y = j * 80

                temp_array = image[x:x + 60, y:y + 80]
                depth_avg[i][j] = np.average(temp_array)

                if np.isnan(depth_avg[i][j]):
                    depth_avg[i][j] = 10.0

        return depth_avg

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

    def __depth_image_raw_callback(self, depth_image_raw):
        try:
            self.__depth_image_raw = self.__bridge.imgmsg_to_cv2(depth_image_raw, "32FC1")
        except CvBridgeError as e:
            print(e)

        self.__depth_image_raw = np.array(self.__depth_image_raw, dtype=np.float32)

    def __subscribe_model_states(self):
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.__model_states_callback)
        # rospy.sleep(1)

    def __subscribe_depth_image_raw(self):
        rospy.Subscriber("/camera/depth/image_raw", Image, self.__depth_image_raw_callback)
        # rospy.sleep(1)

    def get_state(self):
        # S(t) = (distance(t), depth(t), time_passed(t))

        distance = self.__get_distance_between(self.__position, self.__destination)
        depth = self.__compress(self.__depth_image_raw)
        time_passed = time.time() - self.__initial_time

        return distance, depth, time_passed

    def get_reward(self, state):
        # Importance hierarchy: Depth > Distance > Time Passed

        if self.__get_distance_between(self.__position, self.__destination) < .1:  # Destination reached!
            return 1000
        # elif self.crashed:
        #   return -1500
        # elif state[2] > self.time_limit:
        #   return -1000

        return state[0] * 2 + state[1] * 3 - state[2]

    def act(self, action, v1=.3, v2=.05, num_iterations=10):
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

        for _ in range(num_iterations):
            if rospy.is_shutdown():
                return

            self.__vel_pub.publish(vel_cmd)
            self.__rate.sleep()

        state = self.get_state()
        reward = self.get_reward(state)

        return state, reward, False

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

