#!/usr/bin/env python

# imports
import sys
import numpy as np
import json
import random

# ros imports
import rospy
from std_msgs.msg import String


class Test_Class:
    def __init__(self):
        # publishers
        self.test_pub = rospy.Publisher('/mcts/path_data', String, queue_size = 10) # jsonified data

        # some other stuff
        self.json_dict = {"nodes": [], "edges": []}

        # hardcode
        self.json_dict['nodes'].append({"node_id": 1, "node_reward": 3})
        self.json_dict['nodes'].append({"node_id": 2, "node_reward": 0})
        self.json_dict['edges'].append({"edge_id": 1, "edge_distance": 10})


############################# Publisher functions ##############################
    def test_publish(self):
        numNodes = 9
        message = []
        for i in range(numNodes):
            node = {}
            node["node_id"] = i+1
            node["node_reward"] = np.random.random()*5.0
            node["has_agent"] = random.sample([0,1], 1)
            # random position in a 10x10m space
            node["x"] = 10*np.random.random_sample()
            node["y"] = 10*np.random.random_sample()
            node["connectivity"] = random.sample([1,2,3,4,5,6,7,8,9], 3)
            node["costs"] = random.sample([1,2,3,4], 3)
            node["paths"] =  [[[0.0,0.0],[1.0,1.0],[2.0,2.0]],
                              [[3.0,3.0],[4.0,4.0],[5.0,5.0]],
                              [[6.0,6.0],[7.0,7.0],[8.0,8.0]]]
            message.append(node)


        # publish action
        self.test_pub.publish(json.dumps(message))


############################# Main #############################################
def main():
    # init ros node
    rospy.init_node('test', anonymous = True)

    # class instance
    test_instance = Test_Class()

    # create ros loop
    pub_rate = 1.0 # hertz
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
