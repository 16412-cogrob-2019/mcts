# MCTS for Kolumbo Volcano Exploration
The repo, except having the base MCTS algorithm and Kolumbo exploration problem model, has several extensions - 
a neural network extension to rollout policy, a hierarchical model which includes the mothership in the problem, decentralized MCTS that considers intermittent communication, and a ROS integration interface. 


## The Base MCTS
The base algorithm and model are in the `master` branch.

We are coordinating multiple agents to explore valuable areas around the Kolumbo Volcano. The estimated values of locations
and the estimated cost of moving from one location to another have been provided by other teams.

The base MCTS code is in `src/mcts.py`. Users can define their own method for tree policy and rollout policy and enter them as input.

We have provided a simple discrete-time model, in `src/discrete/maze.py` in the `master` branch.
Execute `src/discrete/maze.example.py` to run examples of solving the problem using MCTS.

We have a more realistic continuous-time model, where locations are modeled as nodes on a graph and the cost of going from a location to another becomes the weight of the associated edges. Run `src/example.py` in the `master` branch to solve an example problem using MCTS.


## Neural Network Rollout
This extension is in the `master` branch.

We implemented a neural network that estimates the expected reward, and the estimation is used in a modified rollout policy.
In `src/heuristics/heuristics.py`, MCTS generates training data for the neural network, and the neural network updates its weights. Note: running the script may take hours.

We have a trained neural network with its data in `src/heuristics/neural_net.db`. To run an example using it, execute `src/heuristics/example.py`. Note: running the neural network rollout policy takes longer than the default random rollout policy; if the program takes too long, you can reduce the number of samples.

In `src/heuristics/compare.py`, we compare the results between the rollout policies. Note: running the script may take very long. Reduce the number of samples if needed.


## Hierarchical Model
This extension is in the `master` branch.

Using a modified version of the base model (`src/hierarchical/hierarchical_state.py`) and the base MCTS algorithm (`src/hierarchical/hierarchical_mcts.py`), a hierarchical model is built which considers a mothership and a tethered AUV. The set of primitive actions in this environment includes the AUV motions from the base model, as well as the motions of the mothership between search regions. The macro-actions in our model represent the deployment of the AUV from the mothership using different rollout policies. 

Execute `src/hierarchical/hierarchical_example.py` for a demonstration of a mothership and tethered AUV planning their actions while exploiting the problem's hierarchical structure. You will see two examples of tethered AUV deployments, which depict the primitive actions that compose a single macro-action. Then, you will see an example of the mothership's motion between regions, which depicts a sequence of macro-actions.


## Decentralized MCTS 
This extension is in the `master` branch.

Using a modified version of the base model (`src/decentralized/maze.py`) and the base MCTS algorithm (`src/decentralized/mcts.py`), a decentralized model is built that considers two agents that are intermittently communicate with each other while performing MCTS. You can execute the `src/decentralized/dec_demo.py` file to observe a working example of the decentralized MCTS scenario in a simple environment that requires proper collaboration in order to retrieve the optimal amount of reward. 


## ROS Integration
To view the ROS integration code, switch to the `mcts_ros_integration_v2` branch.

Code for ROS integration is in `src`. `src/mcts_node.py` provides the main function. `test_pub.py` takes some hardcoded data (dictionary of lists), uses json dumps to make it a string, and publishes that string at 10 hertz. skeleton_code.py subscribes to that topic, uses json loads to convert it back to a dictionary, and prints it out. 

To run...

1. clone the repo
1. in terminal: "cd mcts && catkin_make"
1. in terminal: "roscore"
1. in new terminal: "source devel/setup.bash && rosrun mcts mcts_node.py"
1. in new terminal: "source devel/setup.bash && rosrun mcts test_pub.py"

In the third terminal (the one where you are running skeleton_code.py), you should be seeing a printed out message...that confirms it's working. 
