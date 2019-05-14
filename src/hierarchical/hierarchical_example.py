from hierarchical_state import *
from hierarchical_mcts import *


def line(start: (int, int), increment: (int, int), length: int) -> set:
    return set([(start[0] + k * increment[0], start[1] + k * increment[1]) for
                k in range(length + 1)])


def circle(center: (int, int), radius: int = 1):
    return set([(i, j)
                for i in range(center[0] - radius, center[0] + radius + 1)
                for j in range(center[1] - radius, center[1] + radius + 1)
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2])


def hierarchical_example():

    caldera_targets = {(6, 3): 3, (4, 8): 3, (2, 3): 3, (14, 4): 3, (15, 8): 3,
               (17, 3): 3, (9, 12): 3, (14, 14): 3, (8, 16): 3, (11, 11): 1,}
    #caldera_env = KolumboEnvironment(xlim=(0, 20), ylim=(0, 20), obstacles={},
    #                      targets=caldera_targets, is_border_obstacle_filled=True)
    caldera_state = KolumboState(time_remains=15)
    target_locs = list(caldera_targets.keys())
    for i in range(len(target_locs)):
        caldera_state.add_location(i, caldera_targets[target_locs[i]], target_locs[i])
    for i in range(len(target_locs)):
         for j in range(len(target_locs)):
            caldera_state.add_path(i, j, ((target_locs[j][0] - target_locs[i][0])**2 + (target_locs[j][1] - target_locs[i][1])**2)**0.5)
    caldera_state.add_agent(1)

    ridge_targets = {(17, 2): 10, (4, 19): 10, (2, 3): 10,
               (9, 11): 1, (13, 12): 1, (14, 13): 1, (16, 15): 1,
               (18, 14): 1, (19, 17): 1, (17, 19): 4, (8, 17): 1}
    ridge_state = KolumboState(time_remains=15)
    target_locs = list(ridge_targets.keys())
    for i in range(len(target_locs)):
        ridge_state.add_location(i, ridge_targets[target_locs[i]], target_locs[i])
    for i in range(len(target_locs)):
         for j in range(len(target_locs)):
            ridge_state.add_path(i, j, ((target_locs[j][0] - target_locs[i][0])**2 + (target_locs[j][1] - target_locs[i][1])**2)**0.5)
    ridge_state.add_agent(1)


    falkor_targets = {(1, 1): 'C', (1, 2): 'R', (1, 3): 'R',
                    (2, 1): 'C', (2, 2): 'R', (2, 3): 'R',
                    (3, 1): 'R', (3, 2): 'C', (3, 3): 'C',}

    #falkor_env = FalkorEnvironment(xlim=(1, 3), ylim=(1, 3), obstacles={},
    #                      feature_map=falkor_targets, is_border_obstacle_filled=True, primitive_states={'C': caldera_state, 'R': ridge_state}, 
    #                      simulation_function = simulate)

    primitive_states = {'C': caldera_state, 'R': ridge_state}
    meta_action_rewards = {'C': 0, 'R': 0}

    for meta_action in primitive_states.keys():
        sim_result = simulate(primitive_states[meta_action])
        sim_result.visualize()
        meta_action_rewards[meta_action] = sim_result.reward

    for target in falkor_targets:
        falkor_targets[target] = meta_action_rewards[falkor_targets[target]]

    print(falkor_targets)


    falkor_state = FalkorState(time_remains=70)

    target_locs = list(falkor_targets.keys())
    for i in range(len(target_locs)):
        falkor_state.add_location(i, falkor_targets[target_locs[i]], target_locs[i])
    for i in range(len(target_locs)):
         for j in range(len(target_locs)):
            falkor_state.add_path(i, j, ((target_locs[j][0] - target_locs[i][0])**2 + (target_locs[j][1] - target_locs[i][1])**2)**0.5)
    falkor_state.add_agent(1)

    return falkor_state


def simulate(initial_state: KolumboState) -> KolumboState:

    mcts = MonteCarloSearchTree(initial_state)
    state = initial_state.__copy__()
    while not state.is_terminal:
        actions = mcts.search_for_actions(search_depth=1)
        time = state.time_remains
        print("Time remaining: {0}".format(time))
        action = actions[0]
        print(action)
        state = state.execute_action(action)
        mcts.update_root(action)
        if state.is_terminal:
            break
    return state



if __name__ == "__main__":
    print("===== Start of Example 1 =====")

    simulate(initial_state=hierarchical_example()).visualize()
