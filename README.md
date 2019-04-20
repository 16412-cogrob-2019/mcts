This repo has two .py files in it. test_pub.py that takes some hardcoded data (dictionary of lists), uses json dumps to make it a string, and publishes that string at 10 hertz. skeleton _code.py subscribes to that topic, uses json loads to convert it back to a dictonary, and prints it out. 

To run...

1) clone the repo
2) in terminal: "cd mcts && catkin_make"
3) in terminal: "roscore"
4) in new terminal: "source devel/setup.bash && rosrun mcts skeleton_code.py"
5) in new terminal: "source devel/setup.bash && rosrun mcts test_pub.py"

In the third terminal (the one where you are running skeleton_code.py), you should be seeing a printed out message...that confirms it's working. 
