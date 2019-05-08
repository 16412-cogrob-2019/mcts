#!/usr/bin/env python

# imports
import sys
import numpy as np
import json
from pprint import pprint
import copy

# ros imports
import rospy
from std_msgs.msg import String
import mcts_action_selection.msg as cmsg


from mcts import *
from state import *


class MctsNode:
    def __init__(self):
        
        self.sub_map = rospy.Subscriber('/eep/data', String, self.cb_map)
        self.pub_command = rospy.Publisher('/mcts/command', cmsg.ActivityRequest, queue_size=1)

    def cb_map(self, msg):

        action_msg = cmsg.ActivityRequest()

        json_map = json.loads(msg.data)

        self.kolumbo_state = KolumboState()

        self.kolumbo_state.json_parse_to_map(json_map)

        self.kolumbo_mcts = MonteCarloSearchTree(self.kolumbo_state)

        action = self.kolumbo_mcts.search_for_actions(search_depth=1)[0]

        for node in json_map:

            node_id = node['node_id']
            agent_id = node['agent_id']
            connected_to = node['connectivity']
            paths = node['paths']

            if (action._start_location == node_id) and (agent_id != -1):                
                for con_node, con_path in zip(connected_to, paths):                     
                    if (action._end_location == con_node):   
                        path_to_publish = copy.copy(con_path)
                        agent_to_publish = copy.copy(agent_id)

        action_msg.activity_id = agent_to_publish

        plan_msg = cmsg.Plan()
        
        plan = []
        
        for wp in path_to_publish:
            wp_msg = cmsg.Waypoint()
            wp_msg.x = wp[0]
            wp_msg.y = wp[1]
            wp_msg.vel = 0.5
            plan.append(wp_msg)

        plan_msg.wypts = plan
        action_msg.plns = [plan_msg]

        pprint(action_msg)

        self.kolumbo_state = self.kolumbo_state.execute_action(action)
        
        self.kolumbo_mcts.update_root(action)

        self.pub_command.publish(action_msg)
        

############################# Main #############################################
def main():
    # init ros node
    rospy.init_node('mcts_node', anonymous=True)

    # class instance
    mcts_kolumbo = MctsNode()

    # create ros loop
    pub_rate = 1  # hertz
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
