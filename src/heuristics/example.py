from maze import *
from mcts import *
from heuristics import *


def maze_example_1():
    targets = {(6, 3): 3, (4, 8): 3, (2, 3): 3, (8, 6): 1, (8, 8): 1, (7, 2): 3}
    env = MazeEnvironment(xlim=(0, 10), ylim=(0, 10), obstacles=set(),
                          targets=targets, is_border_obstacle_filled=True)
    state = MazeState(environment=env, time_remains=10)
    state.add_agent((7, 7)).add_agent((3, 3)).add_agent((5, 5))
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
    num_samples = 100

    trainer = KolumboHeuristicsGenerator()
    model = trainer.init_neural_network()
    model.load_weights(neural_net_model_file)
    neural_net_rollout = trainer.get_rollout_policy(model, epsilon=0.05)

    print("===== Start of Example =====")
    maze_example_1().visualize()
    simulate(initial_state=maze_example_1(),
             mcts_select_policy=select,
             mcts_expand_policy=expand,
             mcts_rollout_policy=neural_net_rollout,
             mcts_backpropagate_policy=backpropagate,
             num_samples=num_samples).visualize()


