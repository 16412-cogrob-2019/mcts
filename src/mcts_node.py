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

# mcts libraries
from mcts import *
from state import *


class MctsNode:
    def __init__(self):
        """
        Setup ROS subscribers and publishers
        """
        self.sub_map = rospy.Subscriber('/eepp/path_data', String, self.cb_map)
        self.pub_command = rospy.Publisher('/activity/post', cmsg.ActivityRequest, queue_size=1)

    def cb_map(self, msg):
        """
        Callback in response to self.sub_map that:
        1) Resets the environment
        2) Parses the incoming json message into a networkx graph
        3) Performs MCTS to find actions for each agent
        4) Publishes those actions according to the ActivityRequest message definition
        """
        # reset environment
        self.kolumbo_state = KolumboState()

        # load message and parse
        action_msg = cmsg.ActivityRequest()
        json_map = json.loads(msg.data)
        self.kolumbo_state.json_parse_to_map(json_map)

        # perform MCTS and find actions
        self.kolumbo_mcts = MonteCarloSearchTree(self.kolumbo_state)
        n_agents = rospy.get_param("n_agents",2)
        action = self.kolumbo_mcts.search_for_actions(search_depth=n_agents) # set to number of agents
        print(action[0])

        # publish the message according to ActivityRequest
        path_to_publish = []
        agent_to_publish = []
        for node in json_map:

            node_id = node['poi_id']
            agent_id = node['agent_id']
            connected_to = node['connectivity']
            paths = node['paths']

            # for each node/action, find the corresponding node and pass along its path/agent
            for act in action:
                if (act._start_location == node_id) and (agent_id != -1):
                    for con_node, con_path in zip(connected_to, paths):
                        if (act._end_location == con_node):
                            path_to_publish.append(copy.copy(con_path))
                            agent_to_publish.append(copy.copy(agent_id))

        # for each agent/path, form the message and publish it
        for agent,path in zip(agent_to_publish, path_to_publish):
            action_msg.activity_id = agent

            plan_msg = cmsg.Plan()

            plan = []

            for wp in path:
                wp_msg = cmsg.Waypoint()
                wp_msg.x = wp[0]
                wp_msg.y = wp[1]
                wp_msg.vel = 0.5
                plan.append(wp_msg)

            plan_msg.wypts = plan
            action_msg.plns = [plan_msg]
            print(action_msg)

            self.pub_command.publish(action_msg)


############################# Main #############################################
def main():
    """
    Initiates ROS node, MctsNode class, and then loops in ROS at pub_rate hertz.
    """
    # init ros node
    rospy.init_node('mcts_node', anonymous=True)

    # class instance
    mcts_kolumbo = MctsNode()

    # create ros loop
    pub_rate = 1  # hertz
    rate = rospy.Rate(pub_rate)

    while (not rospy.is_shutdown()):
        # do some stuff if necessary, ours is blank as everything is in the callback

        # ros sleep (sleep to maintain loop rate)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
