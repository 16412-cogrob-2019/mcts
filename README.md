# ROS Integration
Code for ROS integration is in src in the "mcts_ros_integration_v2" branch. src/mcts_node.py provides the main function and all of the ROS functionality. src/test_pub.py serves as sanity check to make sure our implementation of MCTS is working with ROS. It takes some hardcoded data (dictionary of lists), uses json dumps to make it a string, and publishes that string at a set frequency. mcts_node.py subscribes to that topic or the topic published by the EEPP group, uses json loads to convert it back to a dictonary, and uses that to build a networkx graph. Finally, MCTS is run over this networkx graph and the best actions are returned for each agent. These actions are then formatted and published to the Santorini main turtlebot controller. 

To run with test_pub.py:
Clone the repo inside an existing catkin workspace src folder.
In terminal: "cd .. && catkin_make"
In terminal: "roscore"
In new terminal: "source devel/setup.bash && rosrun mcts_action_selection mcts_node.py"
In new terminal: "source devel/setup.bash && rosrun mcts_action_selection test_pub.py"

To run with the rest of the Santorini packages
Clone the MAAS, EEPP, MCTS, and Santorini packages into an existing catkin workspace src folder.
In terminal: "cd .. && catkin_make"
In terminal: "roslaunch Santorini santorini.launch"
