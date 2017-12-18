import rospy
from geometry_msgs.msg import Twist


class Navigator():
    
    def __init__(self, frequency=10, velocity=.3, num_iterations=10):
        rospy.init_node("GoForward", anonymous=False)
        rospy.loginfo("CTRL + C to terminate..")   
        rospy.on_shutdown(self.shutdown)
        
        # Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher("cmd_vel_mux/input/navi", Twist, queue_size=10)
        # TurtleBot will stop if we don't keep telling it to move. How often should we tell it to move?
        self.rate = rospy.Rate(frequency)

        self.velocity = velocity
        self.num_iterations = num_iterations

    @staticmethod
    def get_radians_from(degrees):
        return 0.0174532925 * degrees

    def move(self, forward, degrees):
        velocity = self.velocity if forward else -self.velocity

        move_cmd = Twist()
        move_cmd.linear.x = velocity
        move_cmd.angular.z = self.get_radians_from(degrees)

        for _ in range(self.num_iterations):
            if rospy.is_shutdown():
                return

            self.cmd_vel.publish(move_cmd)
            self.rate.sleep()

    def shutdown(self):
        rospy.loginfo("TurtleBot is stopping..")
        self.cmd_vel.publish(Twist())

        rospy.sleep(1)
        rospy.loginfo("TurtleBot stopped!")

