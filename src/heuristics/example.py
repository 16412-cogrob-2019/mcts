from maze import *
from mcts import *
from heuristics import *


def maze_example_1():
    targets = {(6, 3): 3, (4, 8): 3, (2, 3): 3, (14, 4): 3, (15, 8): 3,
               (17, 3): 3, (9, 12): 3, (14, 14): 3, (8, 16): 3, (11, 11): 1,
               (11, 13): 1, (12, 6): 1, (10, 7): 1, (8, 6): 1, (8, 8): 1,
               (16, 12): 1, (11, 16): 1, (18, 4): 1}
    env = MazeEnvironment(xlim=(0, 20), ylim=(0, 20), obstacles=set(),
                          targets=targets, is_border_obstacle_filled=True)
    state = MazeState(environment=env, time_remains=15)
    state.add_agent((7, 7)).add_agent((13, 8)).add_agent((12, 12))
    return state


def maze_example_2():
    targets = {(17, 2): 10, (4, 19): 10, (2, 3): 10,
               (9, 11): 1, (13, 12): 1, (14, 13): 1, (16, 15): 1,
               (18, 14): 1, (19, 17): 1, (17, 19): 4, (8, 17): 1,
               (3, 12): 1,
               (10, 15): 1, (14, 10): 1, (14, 11): 1, (17, 11): 1,
               (14, 4): 1,
               (19, 5): 1, (8, 6): 1, (8, 5): 1, (4, 8): 4, (1, 6): 1,
               (15, 8): 1, (6, 1): 1, (3, 4): 1, (6, 12): 1, (11, 2): 4,
               (19, 11): 4}
    env = MazeEnvironment(xlim=(0, 20), ylim=(0, 20), obstacles=set(),
                          targets=targets, is_border_obstacle_filled=True)
    state = MazeState(environment=env, time_remains=15)
    state.add_agent((7, 7)).add_agent((13, 8)).add_agent((12, 12))
    return state


def simulate(mcts_select_policy, mcts_expand_policy, mcts_rollout_policy,
             mcts_backpropagate_policy, initial_state: MazeState,
             rand_seed: int = 0, num_samples: int = 100) -> MazeState:
    mcts = MonteCarloSearchTree(initial_state, max_tree_depth=15,
                                samples=num_samples,
                                tree_select_policy=mcts_select_policy,
                                tree_expand_policy=mcts_expand_policy,
                                rollout_policy=mcts_rollout_policy,
                                backpropagate_method=mcts_backpropagate_policy)
    random.seed(rand_seed)
    state = initial_state.__copy__()
    time = 0
    while not state.is_terminal:
        actions = mcts.search_for_actions(search_depth=3)
        time += 1
        print("Time step {0}".format(time))
        for i in range(len(actions)):
            action = actions[i]
            print(action)
            state = state.execute_action(action)
            mcts.update_root(action)
    return state


if __name__ == '__main__':

    neural_net_model_file = 'neural_net.db'
    num_samples = 400

    trainer = KolumboHeuristicsGenerator()
    model = trainer.init_neural_network()
    model.load_weights(neural_net_model_file)
    neural_net_rollout = trainer.get_rollout_policy(model, epsilon=0.05)

    print("===== Start of Example 1 =====")
    maze_example_1().visualize()
    simulate(initial_state=maze_example_1(),
             mcts_select_policy=select,
             mcts_expand_policy=expand,
             mcts_rollout_policy=neural_net_rollout,
             mcts_backpropagate_policy=backpropagate,
             num_samples=num_samples).visualize()
    print("\n===== Start of Example 2 =====")
    simulate(initial_state=maze_example_2(),
             mcts_select_policy=select,
             mcts_expand_policy=expand,
             mcts_rollout_policy=neural_net_rollout,
             mcts_backpropagate_policy=backpropagate,
             num_samples=num_samples).visualize()



