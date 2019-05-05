#!/usr/bin/env python

# imports
import sys
import numpy as np
import json

# ros imports
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String

import mcts
import state



class MctsNode:
    def __init__(self):
        # subs and pubs
        self.sub_map     = rospy.Subscriber('/mcts/path_data', String, self.cb_map)
        self.pub_command = rospy.Publisher('/mcts/command', Float32MultiArray, queue_size=1)

        # initialize state
        self.kolumbo_state = state.KolumboState(time_remains = 20.0)
        self.kolumbo_state.add_agent((7,7))

        # initalize mcts
        self.kolumbo_mcts  = mcts.MonteCarloSearchTree(self.kolumbo_state, samples = 1000,
                                            max_tree_depth = 5)


    def cb_map(self, msg):
        # get data
        json_map = json.loads(msg.data)

        # reset environment
        self.kolumbo_state.reset_environment()

        # update state with new information
        self.kolumbo_state.json_parse_to_map(json_map)

        # get best actions for each agent
        actions = self.kolumbo_mcts.search_for_actions(search_depth = 1, random_seed = None)

        # get those actions in the right format
        print(actions)

        # publish it
        # self.publish_action()


    def publish_action(self):
        # publish action
        self.mcts_pub.publish(json.dumps(self.data))


    #def simulate ()


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
