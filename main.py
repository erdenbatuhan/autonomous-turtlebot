import rospy
from scripts.movement import Navigator


if __name__ == '__main__':
	try:
		navigator = Navigator()

		navigator.move(True, 45)    # Move forward with 45 degrees
		navigator.move(False, 45)   # Move backwards with 45 degrees
		navigator.move(False, -45)  # Move backwards with -45 (315) degrees
	except:
		rospy.loginfo("Terminated!")

