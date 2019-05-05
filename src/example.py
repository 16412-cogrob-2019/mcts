from state import *
from mcts import *

initial_state = KolumboState()
initial_state.add_location(0, 1.0, (0, 0)).add_location(1, 2.0, (0, 1)) \
    .add_location(2, 1.5, (1, 0)).add_location(3, 3.0, (1, 1))
initial_state.add_path(0, 1, 1.0).add_path(1, 2, 2.0).add_path(1, 3, 4.0) \
    .add_path(2, 3, 1.0).add_path(3, 0, 2.5)
initial_state.add_agent(0).add_agent(3)
# mcts = MonteCarloSearchTree(initial_state)
# action = mcts.search_for_actions()[0]
action1 = KolumboAction(0, 0, 1, 1.0)
state = initial_state.execute_action(action1)
action2 = KolumboAction(1, 3, 0, 2.5)
state = state.execute_action(action2)
action3 = KolumboAction(0, 1, 2, 2.0)
state = state.execute_action(action3)
initial_state.visualize()
state.visualize()
