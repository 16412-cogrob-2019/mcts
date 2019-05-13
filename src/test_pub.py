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
        """
        Define hardcoded message
        """
        # publishers
        self.test_pub = rospy.Publisher('/eepp/path_data', String, queue_size = 10) # jsonified data

        self.x_hc               = [0,0,1,1,2]
        self.y_hc               = [0,1,0,1,2]
        self.reward_hc          = [30,20,0,0,50]
        self.has_agent_hc       = [-1,-1,1,0,-1]
        self.connectivity_hc    = [[1,3],[0,2,3],[1,3,4],[0,2,4],[2,3]]
        self.cost_hc            = [[1,1],[2,4,1],[1,2],[2.5,3,2],[1,1]]
        self.paths_hc           = [[[[0.0,0.0],[1.0,1.0]],[[0.0,0.0],[2.0,1.0]]],
                                   [[[0.1,0.1],[1.0,1.0]],[[0.0,0.0],[2.0,1.0]],[[0.0,0.0],[2.0,1.0]]],
                                   [[[0.2,0.2],[1.0,1.0]],[[0.0,0.0],[2.0,1.0]],[[0.0,0.0],[2.0,1.0]]],
                                   [[[0.3,0.3],[1.0,1.0]],[[0.0,0.0],[2.0,1.0]],[[0.0,0.0],[2.0,1.0]]],
                                   [[[0.4,0.4],[1.0,1.0]],[[0.0,0.0],[2.0,1.0]]]]



############################# Publisher functions ##############################
    def test_publish(self):
        """
        Publish hardcoded message
        """
        numNodes = len(self.x_hc)
        message = []

        for i in range(numNodes):
            node = {}
            node["poi_id"] = i
            node["poi_reward"] = self.reward_hc[i]
            node["agent_id"] = self.has_agent_hc[i]
            # random position in a 10x10m space
            node["x"] = self.x_hc[i]
            node["y"] = self.y_hc[i]
            node["connectivity"] =  self.connectivity_hc[i]
            node["costs"] = self.cost_hc[i]
            node["paths"] = self.paths_hc[i]

            message.append(node)

        # publish action
        self.test_pub.publish(json.dumps(message))


############################# Main #############################################
def main():
    """
    Initiates ROS node, Test_Class class, and then loops in ROS at pub_rate hertz.
    """
    # init ros node
    rospy.init_node('test', anonymous = True)

    # class instance
    test_instance = Test_Class()

    # create ros loop
    pub_rate = 1.0/30 # hertz
    rate = rospy.Rate(pub_rate)

    while (not rospy.is_shutdown()):
        # pack something in a json object
        test_instance.test_publish()

        # ros sleep (sleep to maintain loop rate)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
