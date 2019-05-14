from heuristics import *
from mcts import *
import numpy as np


if __name__ == '__main__':
    neural_net_model_file = 'neural_net.db'
    res_file = 'compare.db'
    num_trials = 100
    num_samples = 200

    trainer = KolumboHeuristicsGenerator()
    model = trainer.init_neural_network()
    model.load_weights(neural_net_model_file)
    neural_net_rollout = trainer.get_rollout_policy(model, epsilon=0.05)
    try:
        res = np.loadtxt(res_file, skiprows=0, delimiter=' ')
    except IOError:
        res = np.zeros((2, 0))

    for i in range(num_trials):
        print("Trial {0} of {1}".format(i + 1, num_trials))
        initial_state = trainer.generate_maze_example()
        mcts_default = MonteCarloSearchTree(initial_state, samples=num_samples)
        mcts_nn = MonteCarloSearchTree(initial_state, samples=num_samples,
                                       rollout_policy=neural_net_rollout)
        state_default = initial_state.__copy__()
        state_neural = initial_state.__copy__()
        while not state_default.is_terminal:
            actions = mcts_default.search_for_actions(search_depth=
                                                      len(initial_state.paths))
            for action in actions:
                state_default = state_default.execute_action(action)
                mcts_default.update_root(action)
        while not state_neural.is_terminal:
            actions = mcts_nn.search_for_actions(search_depth=
                                                 len(initial_state.paths))
            for action in actions:
                state_neural = state_neural.execute_action(action)
                mcts_nn.update_root(action)
        reward_neural = state_neural.reward
        reward_default = state_default.reward
        res = np.hstack((res, np.array([[reward_default], [reward_neural]])))
        np.savetxt(fname=res_file, X=res, delimiter=' ')
        print("\tThe default rollout policy has final reward {0}, and the "
              "neural network rollout policy has final reward {1}".
              format(reward_default, reward_neural))

    data = np.loadtxt(res_file)
    mean1 = np.mean(data[0, :])
    mean2 = np.mean(data[1, :])
    print("The total number of trials is {0}.".format(data.shape[1]))
    print("The average improvement by new rollout is {0}%".
          format(round((mean2 - mean1) / mean1 * 100), 2))
    counter0 = 0
    counter1 = 0
    counter2 = 0
    for i in range(data.shape[1]):
        if (data[0, i] - data[1, i]) / data[1, i] > 0.05:
            counter1 += 1
        elif data[1, i] > data[0, i]:
            counter0 += 1
            if (data[1, i] - data[0, i]) / data[0, i] > 0.05:
                counter2 += 1
    print("In {0} trials, the new rollout outperforms the old.".
          format(counter0))
    print("In {0} trials, the new rollout outperforms the old one by 5%".
          format(counter2))
    print("In {0} trials, the old rollout outperforms the new one by 5%".
          format(counter1))

