import rospy
from geometry_msgs.msg import Twist


class Environment:
    
    def __init__(self, frequency=10):
        rospy.init_node("GoForward", anonymous=False)
        rospy.loginfo("CTRL + C to terminate..")   
        rospy.on_shutdown(self.shutdown)
        
        # Create a publisher which can "talk" to TurtleBot and tell it to move)
        self.vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
        # TurtleBot will stop if we don't keep telling it to move. How often should we tell it to move?
        self.rate = rospy.Rate(frequency)

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

    def shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.vel_pub.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

