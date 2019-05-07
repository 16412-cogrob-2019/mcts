from state import *
from mcts import *

initial_state = KolumboState(time_remains=5)
initial_state.add_location(0, 1.0, (0, 0)).add_location(1, 2.0, (0, 1)) \
    .add_location(2, 1.5, (1, 0)).add_location(3, 3.0, (1, 1)) \
    .add_location(4, 1.5, (3, 2)).add_location(5, 1.0, (2, 2)) \
    .add_location(6, 2.0, (4, 3)).add_location(7, 2.0, (3, 3)) \
    .add_location(8, 3.0, (4, 0)).add_location(9, 1.0, (0, 3))
initial_state.add_path(0, 1, 1.0).add_path(1, 2, 2.0).add_path(1, 3, 4.0) \
    .add_path(2, 3, 1.0).add_path(3, 0, 2.5).add_path(3, 9, 2.0) \
    .add_path(9, 1, 2.0).add_path(9, 0, 1.0).add_path(9, 2, 1.0) \
    .add_path(5, 9, 9.0).add_path(5, 3, 1.0).add_path(5, 2, 9.0) \
    .add_path(5, 1, 9.0).add_path(5, 0, 9.0).add_path(8, 5, 1.0) \
    .add_path(8, 6, 1.0).add_path(5, 8, 4.0).add_path(1, 8, 9.0) \
    .add_path(6, 8, 2.0).add_path(6, 4, 1.0).add_path(4, 7, 1.0) \
    .add_path(7, 6, 3.0).add_path(5, 4, 3.0).add_path(2, 9, 9.0) \
    .add_path(7, 9, 3.0).add_path(9, 3, 1.0).add_path(9, 5, 1.0) \
    .add_path(8, 1, 1.0).add_path(8, 5, 1.0).add_path(8, 6, 3.0) \
    .add_path(5, 3, 2.0).add_path(3, 5, 2.0).add_path(5, 9, 1.0)
initial_state.add_agent(0).add_agent(3).add_agent(6).add_agent(9).add_agent(8)
mcts = MonteCarloSearchTree(initial_state)
state = initial_state.__copy__()
while not state.is_terminal:
    action = mcts.search_for_actions(search_depth=1)[0]
    state = state.execute_action(action)
    mcts.update_root(action)
initial_state.visualize()
state.visualize()
