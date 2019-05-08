from state import *
from mcts import *
import numpy as np



initial_m_state = MothershipState()
initial_m_state.add_location(0, 1.0, (0, 0)).add_location(1, 2.0, (0, 1)) \
    .add_location(2, 1.5, (1, 0)).add_location(3, 3.0, (1, 1))
initial_m_state.add_path(0, 1, 1.0).add_path(1, 2, 2.0).add_path(1, 3, 4.0) \
    .add_path(2, 3, 1.0).add_path(3, 0, 2.5)

exploration_regions = {}
values = {}

for location in initial_m_state._environment:

    exploration_regions[location] = KolumboState()
    exploration_regions[location].add_location(0, np.random.rand(), (0, 0)).add_location(1, np.random.rand(), (0, 1)) \
        .add_location(2, np.random.rand(), (1, 0)).add_location(3, np.random.rand(), (1, 1))
    exploration_regions[location].add_path(0, 1, 1.0).add_path(1, 2, 2.0).add_path(1, 3, 4.0) \
        .add_path(2, 3, 1.0).add_path(3, 0, 2.5)
    exploration_regions[location].add_agent(0)
    mcts = MonteCarloSearchTree(exploration_regions[location])
    state = exploration_regions[location].__copy__()
    while not state.is_terminal:
        action = mcts.search_for_actions(search_depth=1)[0]
        state = state.execute_action(action)
        mcts.update_root(action)
    exploration_regions[location].visualize()
    state.visualize()
    values[location] = mcts._root.tot_reward / mcts._root.num_samples
    initial_m_state.set_location_reward(location, values[location])


initial_m_state.add_agent(0)
mcts = MonteCarloSearchTree(initial_m_state)
m_state = initial_m_state.__copy__()
while not m_state.is_terminal:
    m_action = mcts.search_for_actions(search_depth=1)[0]
    m_state = m_state.execute_action(m_action)
    mcts.update_root(m_action)
initial_m_state.visualize()
m_state.visualize()
