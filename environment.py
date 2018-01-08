import time
import util
import rospy
import numpy as np

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Environment:

    ROSPY_FREQUENCY = 10

    def __init__(self, base_name, destination):
        rospy.init_node("Environment", anonymous=False)
        rospy.loginfo("CTRL + C to terminate..")
        rospy.on_shutdown(self.shutdown)

        self.vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
        self.rate = rospy.Rate(self.ROSPY_FREQUENCY)
        self.bridge = CvBridge()
        self.initial_time = time.time()
        self.base_name = base_name
        self.position = {"x": 0., "y": 0.}
        self.destination = destination
        self.depth_image_raw = None
        self.terminal = False
        self.crashed = False

        self.subscriptions_ready = np.zeros(2)
        self.subscribe_model_states()
        self.subscribe_depth_image_raw()

        self.wait_for_subscriptions()
        _, self.initial_distance, _ = util.get_distance_between(self.position, self.destination)

    def shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.vel_pub.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

    def wait_for_subscriptions(self):
        while np.sum(self.subscriptions_ready) < 2:
            pass

    def model_states_callback(self, model_states):
        base_ind = util.get_index_of(model_states.name, self.base_name)
        position = model_states.pose[base_ind].position

        # destination_ind = util.get_index_of(model_states.name, self.__destination_name)
        # destination = model_states.pose[destination_ind].position

        self.position["x"] = position.x
        self.position["y"] = position.y

        self.subscriptions_ready[0] = 1

    def depth_image_raw_callback(self, depth_image_raw):
        try:
            self.depth_image_raw = self.bridge.imgmsg_to_cv2(depth_image_raw, "32FC1")
        except CvBridgeError as e:
            print(e)

        self.depth_image_raw = np.array(self.depth_image_raw, dtype=np.float32)
        self.subscriptions_ready[1] = 1

    def subscribe_model_states(self):
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback)

    def subscribe_depth_image_raw(self):
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_image_raw_callback)

    @staticmethod
    def get_depth_minimized(image):
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
    def get_depth_modified(depth):
        count_special = sum([1 if d == .12 else 0 for d in depth])
        if count_special == 1:
            depth = [1. if d == .12 else d for d in depth]

        powers = [1, 1, 1]
        depth_modified = [1. if d > .75 else d for d in depth]

        return [p * d for p, d in zip(powers, depth_modified)]

    def get_state(self):
        # S(t) = (distance(t), depth(t))
        _, distance, self.terminal = util.get_distance_between(self.position, self.destination)
        depth = self.get_depth_minimized(self.depth_image_raw)

        # Get distance as percentage
        distance /= self.initial_distance

        # Get depth modified
        depth = self.get_depth_modified(depth)

        # Check if crashed
        if np.min(depth) <= 0.05:
            self.crashed = True

        return {
            "greedy": np.array(util.to_precision(distance, 2)).reshape((1, -1)),
            "safe": np.array(util.to_precision_all(depth, 2)).reshape((1, -1))
        }

    def get_reward(self, state):
        reward = {
            "greedy": 20 if self.terminal else (1 - state["greedy"][0][0]),
            "safe": 0 if self.crashed else np.average(state["safe"][0])
        }

        print("State {} | Reward {}".format(state, reward))
        return reward

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

        self.vel_pub.publish(vel_cmd)
        self.rate.sleep()

        self.subscriptions_ready = np.zeros(2)
        self.wait_for_subscriptions()

        state = self.get_state()
        reward = self.get_reward(state)

        return state, reward, self.terminal, self.crashed

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
        self.crashed = False

        self.subscriptions_ready = np.zeros(2)
        self.wait_for_subscriptions()

        self.initial_time = time.time()

