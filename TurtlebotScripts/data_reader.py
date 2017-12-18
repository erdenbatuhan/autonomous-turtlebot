from __future__ import print_function
import rospy
import cv2
import numpy as np
import kobuki_msgs.msg
import BumperEvent
import geometry_msgs
import sensor_msgs
import std_msgs
from cv_bridge import CvBridge, CvBridgeError


class PhotoTaker:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_received = False

    def capture_rgb(self):
        raw_image_topic = "/camera/rgb/image_raw"
        rospy.Subscriber(raw_image_topic, sensor_msgs.msg.Image, self.rgb_callback)
        rospy.sleep(1)

    def capture_depth(self):
        depth_image_topic = "/camera/depth/image_raw"
        rospy.Subscriber(depth_image_topic, sensor_msgs.msg.Image, self.depth_callback)
        rospy.sleep(1)

    def rgb_callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        image = np.array(image, dtype=np.uint8)
        self.image = image

    def depth_callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

        image = np.array(image, dtype=np.float32)
        # Normalize the depth array to 0-1
        # cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
        self.depth_image = image

    def take_picture(self, img_title):
        self.capture_rgb()
        cv2.imwrite(img_title, self.image)

    def take_depth_picture(self, img_title):
        self.capture_depth()
        cv2.imwrite(img_title, self.depth_image)


class BaseDataReader():
    def capture_velocity(self):
        velocity_topic = "/mobile_base/commands/velocity"
        rospy.Subscriber(velocity_topic, geometry_msgs.msg.Twist, self.velocity_callback)
        rospy.sleep(1)

    def capture_bumper(self):
        bumper_topic = "/mobile_base/events/bumper"
        rospy.Subscriber(bumper_topic, kobuki_msgs.msg.BumperEvent, self.bumper_callback)
        rospy.sleep(1)

    def velocity_callback(self, data):
        """
        Will be implemented, currently getting /clock error
        """
        self.velocity = data

    def bumper_callback(self, data):
        """
        Bumper state -1 by default
        """
        if data.state == BumperEvent.PRESSED:
            self.bumper = True
        if data.bumper == BumperEvent.CENTER:
            self.bumper_state = 0
        elif data.bumper == BumperEvent.LEFT:
            self.bumper_state = 1
        elif data.bumper == BumperEvent.RIGHT:
            self.bumper_state = 2
        else:
            self.bumper_state = -1


if __name__ == '__main__':
    """
    rospy.init_node('data_reader', anonymous=False)

    camera = PhotoTaker()

    depth_img_title = rospy.get_param('~image_title', 'depth.jpg')
    camera.take_depth_picture(depth_img_title)
    rospy.loginfo("Saved image " + depth_img_title)
    rospy.sleep(1)

    rgb_img_title = rospy.get_param('~image_title', 'photo.jpg')
    camera.take_picture(rgb_img_title)
    rospy.loginfo("Saved image " + rgb_img_title)
    rospy.sleep(1)

    datareader = BaseDataReader()
    datareader.capture_velocity()
    rospy.sleep(1)
    """