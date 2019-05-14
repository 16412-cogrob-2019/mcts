# MCTS for Kolumbo Volcano Exploration

## The Base MCTS
We are coordinating multiple agents to explore valuable areas around the Kolumbo Volcano. The estimated values of locations
and the estimated cost of moving from one location to another have been provided by other teams.

The base MCTS code is in `mcts.py`. Users can define their own method for tree policy and rollout policy and enter them as input.

We have provided a simple discrete-time model, in `discrete/maze.py` in the `master` branch.
Execute `maze.example.py` to run examples of solving the problem using MCTS.

We have a more realistic continuous-time model, where locations are modeled as nodes on a graph and the cost of going from a location to another becomes the weight of the associated edges. Run `example.py` in the `master` branch to solve an example problem using MCTS.


## 


## ROS Integration
This repo has two .py files in it. test_pub.py that takes some hardcoded data (dictionary of lists), uses json dumps to make it a string, and publishes that string at 10 hertz. skeleton _code.py subscribes to that topic, uses json loads to convert it back to a dictonary, and prints it out. 

To run...

1) clone the repo
2) in terminal: "cd mcts && catkin_make"
3) in terminal: "roscore"
4) in new terminal: "source devel/setup.bash && rosrun mcts skeleton_code.py"
5) in new terminal: "source devel/setup.bash && rosrun mcts test_pub.py"

In the third terminal (the one where you are running skeleton_code.py), you should be seeing a printed out message...that confirms it's working. 
