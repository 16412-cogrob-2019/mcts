#!/usr/bin/env python

# imports
import sys
import numpy as np
import json

# ros imports
import rospy
from std_msgs.msg import String


class Test_Class:
    def __init__(self):
        # publishers
        self.test_pub = rospy.Publisher('/eep/data', String, queue_size = 10) # jsonified data

        # some other stuff
        self.json_dict = {"nodes": [], "edges": []}

        # hardcode
        self.json_dict['nodes'].append({"node_id": 1, "node_reward": 3})
        self.json_dict['nodes'].append({"node_id": 2, "node_reward": 0})
        self.json_dict['edges'].append({"edge_id": 1, "edge_distance": 10})


############################# Publisher functions ##############################
    def test_publish(self):
        # publish action
        self.test_pub.publish(json.dumps(self.json_dict))


############################# Main #############################################
def main():
    # init ros node
    rospy.init_node('test', anonymous = True)

    # class instance
    skeleton_instance = Test_Class()

    # create ros loop
    pub_rate = 10 # hertz
    rate = rospy.Rate(pub_rate)

    while (not rospy.is_shutdown()):
        # pack something in a json object
        skeleton_instance.test_publish()

        # ros sleep (sleep to maintain loop rate)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
