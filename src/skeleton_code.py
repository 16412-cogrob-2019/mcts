#!/usr/bin/env python

# imports
import sys
import numpy as np
import json

# ros imports
import rospy
from std_msgs.msg import String

from mcts import *
import state 



class MctsNode:
    def __init__(self):

        self.sub_map     = rospy.Subscriber('/mcts/path_data', String, self.cb_map)
        self.pub_command = rospy.Publisher('/mcts/command', Float32MultiArray, queue_size=1)

        self.kolumbo_state = KolumboState(time_remains = 20.0)

        self.kolumbo_mcts  = MonteCarloTreeSearch(kolumbo_state, samples = 1000,
                                            max_tree_depth = 5,
                                            tree_select_policy = select,
                                            tree_expand_policy = expand,
                                            rollout_policy = default_rollout_policy,
                                            backpropagate_method = backpropagate) 


    def cb_map(self, msg):

        json_map = json.loads(msg.data)

        self.kolumbo_state.reset_environment()

        self.kolumbo_state.json_parse_to_map(json_map)

        
        # publish it
        self.publish_action()


    def publish_action(self):
        # publish action
        self.mcts_pub.publish(json.dumps(self.data))


############################# Main #############################################
def main():
    # init ros node
    rospy.init_node('mcts_node', anonymous = True)

    # class instance
    mcts_kolumbo = MctsNode()

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
