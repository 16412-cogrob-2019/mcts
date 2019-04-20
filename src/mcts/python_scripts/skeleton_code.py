#!/usr/bin/env python

# imports
import sys
import numpy as np
import json

# ros imports
import rospy
from std_msgs.msg import String


class Skeleton_Class:
    def __init__(self):
        # subscribers
        self.eep_sub = rospy.Subscriber("/eep/data", String, self.data_callback)

        # publishers
        self.mcts_pub = rospy.Publisher('/mcts/data', String, queue_size = 10) # jsonified data

        # some other stuff
        self.data = {"data": []}


############################# Subscriber Callback functions ####################
    def data_callback(self, data):
        # json loads
        temp_data = json.loads(data.data)

        # do some processing
        print(temp_data)

        # publish it
        self.publish_action()


############################# Publisher functions ##############################
    def publish_action(self):
        # publish action
        self.mcts_pub.publish(json.dumps(self.data))


############################# Main #############################################
def main():
    # init ros node
    rospy.init_node('skeleton', anonymous = True)

    # class instance
    skeleton_instance = Skeleton_Class()

    # create ros loop
    pub_rate = 1 # hertz
    rate = rospy.Rate(pub_rate)

    while (not rospy.is_shutdown()):
        # do some stuff if necessary

        # ros sleep (sleep to maintain loop rate)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
