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
        rospy.on_shutdown(self.shutdown)

        self.base_name = base_name
        self.destination_name = destination_name

        self.vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
        self.rate = rospy.Rate(frequency)

        self.position = {"x": 0., "y": 0.}
        self.destination = {"x": -999., "y": -999.}

        self.bridge = CvBridge()
        self.depth_image_raw = None

        self.subscribe_model_states()
        self.subscribe_depth_image_raw()

    @staticmethod
    def get_angle_between_points(p1, p2):
        # It returns the angle in radians between 2 points
        return math.atan2(p2["y"] - p1["y"], p2["x"] - p1["x"])

    @staticmethod
    def get_index_of(arr, item):
        for i in range(len(arr)):
            if arr[i] == item:
                return i

        return -1

    def model_states_callback(self, model_states):
        base_ind = self.get_index_of(model_states.name, self.base_name)
        destination_ind = self.get_index_of(model_states.name, self.destination_name)

        position = model_states.pose[base_ind].position
        destination = model_states.pose[destination_ind].position

        self.position["x"] = position.x
        self.position["y"] = position.y

        self.destination["x"] = destination.x
        self.destination["y"] = destination.y

    def depth_image_raw_callback(self, depth_image_raw):
        try:
            self.depth_image_raw = self.bridge.imgmsg_to_cv2(depth_image_raw, "32FC1")
        except CvBridgeError as e:
            print(e)

        self.depth_image_raw = np.array(self.depth_image_raw, dtype=np.float32)
        # cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)  # Normalize the depth array to 0-1

    def subscribe_model_states(self):
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback)
        # rospy.sleep(1)

    def subscribe_depth_image_raw(self):
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_image_raw_callback)
        # rospy.sleep(1)

    def get_state(self):
        distance_vector = {"x": self.destination["x"] - self.position["x"],
                           "y": self.destination["y"] - self.position["y"],
                           "a": self.get_angle_between_points(self.position, self.destination)}

        # TODO: S(t) = (distance_vector(t), depth_image(t), time_passed(t))
        return distance_vector, self.depth_image_raw, None

    def act(self, action, v1=.3, v2=.05, num_iterations=10):
        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = v1 - v2
            vel_cmd.angular.z = 0.
        elif action == 1:  # LEFT
            vel_cmd.linear.x = v2
            vel_cmd.angular.z = v1
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = v2
            vel_cmd.angular.z = -v1

        for _ in range(num_iterations):
            if rospy.is_shutdown():
                return

            self.vel_pub.publish(vel_cmd)
            self.rate.sleep()

    def reset_base(self):
        rospy.wait_for_service("/gazebo/set_model_state")
        set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        model_state = ModelState()
        model_state.model_name = self.base_name

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

    def shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.vel_pub.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

