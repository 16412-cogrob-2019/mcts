import random
import numpy as np
from copy import deepcopy
from heuristics.maze import MazeState as State, MazeAction as Action, \
    MazeEnvironment as Environment
from mcts import *


class KolumboHeuristicsGenerator:
    def __init__(self, size: int = 10, num_agents: int = 3) -> None:
        """ Create a trainer that can be used to generate a neural network
            with the number of input units determined by the environment size,
            which has a square shape, and the number of agents
        :param size: The number of positions along a row or column
        :param num_agents
        """
        self._env_size = size
        self._num_agents = num_agents

    def generate_maze_example(self, time: int = 10,
                              reward_prob: float = 0.4,
                              reward_mean: float = 1.0,
                              reward_var: float = 1.0) -> State:
        """ Generate a square-shaped maze, with the specified probability that a
            position has non-zero reward; the reward value is a random number
            drawn from a log-normal distribution
        :param time: The number of time steps
        :param reward_prob: The probability that a position has non-zero reward
        :param reward_mean: The mean in log-normal distribution
        :param reward_var: The variance in log-normal distribution
        :return: The random initial State object
        """
        env = Environment(xlim=(0, self._env_size + 1),
                          ylim=(0, self._env_size + 1),
                          is_border_obstacle_filled=True,
                          targets={(i, j): np.random.lognormal(reward_mean,
                                                               reward_var)
                                   for i in range(1, self._env_size + 1)
                                   for j in range(1, self._env_size + 1) if
                                   random.random() < reward_prob})
        unselected = set(
            (i, j) for i in range(1, self._env_size + 1)
            for j in range(1, self._env_size + 1))
        state = State(env, time)
        for _ in range(self._num_agents):
            while True:
                i, j = random.sample(unselected, 1)[0]
                unselected.remove((i, j))
                if any(nb not in env.obstacles for nb in
                       [(i - 1, j), (i + 1, j),
                        (i, j - 1), (i, j + 1)]):
                    state.add_agent((i, j))
                    break
        return state

    def state_action_to_array(self, state: State, action: Action) -> np.array:
        """ Transform the given state and action into a column vector so that
            it can be used in neural network
        :rtype: A 2d numpy array that is a column vector
        """
        rewards = state.environment.rewards
        array = np.zeros((1, self.num_input_units), dtype=float)
        index = 0
        for i in range(1, self._env_size + 1):
            for j in range(1, self._env_size + 1):
                if ((i, j) in state.environment.rewards and
                        (i, j) not in state.visited):
                    array[0, index] = rewards[(i, j)]
                index += 1
        paths = state.paths
        for k in range(self._num_agents):
            x, y = paths[k][-1]
            array[0, index] = x
            index += 1
            array[0, index] = y
            index += 1
        array[0, index] = state.turn
        index += 1
        array[0, index] = state.time_remains
        index += 1
        io, jo = action.position
        i, j = paths[state.turn][-1]
        array[0, index] = 1 if io - i == 1 else -1
        index += 1
        array[0, index] = 1 if jo - j == 1 else -1
        index += 1
        array[0, index] = 1 if i - io == 1 else -1
        index += 1
        array[0, index] = 1 if j - jo == 1 else -1
        index += 1
        return array

    def flip_x(self, state: State, action: Action) -> (State, Action):
        """ Flip the locations of rewards, agents, action and historical paths
            along the x-axis
            The method can be used to generate more training examples
        """
        new_state = state.__copy__()
        rewards = deepcopy(new_state.environment.rewards)
        new_state.environment.clear_rewards()
        x_max = state.environment.x_max
        x_min = state.environment.x_min
        for pos, r in rewards.items():
            i, j = pos
            new_state.environment.add_reward((x_max + x_min - i, j), r)
        for k in range(self._num_agents):
            for index in range(len(new_state.paths[k])):
                i, j = new_state.paths[k][index]
                new_state.paths[k][index] = (x_max + x_min - i, j)
        i, j = action.position
        new_action = Action(action.agent_index, (x_max + x_min - i, j))
        return new_state, new_action

    def flip_y(self, state: State, action: Action) -> (State, Action):
        """ Flip the locations of rewards, agents, action and historical paths
            along the y-axis
            The method can be used to generate more training examples
        """
        new_state = state.__copy__()
        rewards = deepcopy(new_state.environment.rewards)
        new_state.environment.clear_rewards()
        y_max = state.environment.y_max
        y_min = state.environment.y_min
        for pos, r in rewards.items():
            i, j = pos
            new_state.environment.add_reward((i, y_max + y_min - j), r)
        for k in range(self._num_agents):
            for index in range(len(new_state.paths[k])):
                i, j = new_state.paths[k][index]
                new_state.paths[k][index] = (i, y_max + y_min - j)
        i, j = action.position
        new_action = Action(action.agent_index, (i, y_max + y_min - j))
        return new_state, new_action

    def flip_agents(self, state: State, action: Action) -> (State, Action):
        """ Flip agents of a state
            The method can be used to generate more training data
        """
        new_state = state.__copy__()
        new_paths = state.paths[::-1]
        for k in range(self._num_agents):
            new_state.paths[k] = new_paths[k]
        new_state.switch_agent(self._num_agents - 1 - state.turn)
        new_action = Action(self._num_agents - 1 - action.agent_index,
                            action.position)
        return new_state, new_action

    def generate_training_data(self, num_train_samples: int = 1000,
                               num_mcts_samples: int = 500,
                               verbose: int = 100,
                               data_file_name: str = 'heuristics/'
                                                     'training_data.db',
                               time: int = 10,
                               reward_prob: float = 0.4,
                               reward_mean: float = 1.0,
                               reward_var: float = 1.0) -> np.array:
        """ Add new data to the specified training set file (or create it if
            it does not exist yet)
            For any random initial state generated, MCTS will be used to
            generate an entire sequence of history, and all intermediate states
            will be added to the training set
            To obtain more data using limited resources, symmetry is used
            to flip the geometry and to flip agent ids
        :param num_train_samples: The number of random initial states to be
            generated
        :param num_mcts_samples: The number of samples per MCTS step uses
        :param verbose: After the number of samples, the console prints out the
            progress
            When verbose is 0 or None, no information will be printed by console
        :param data_file_name: The file name to store data
        :param time: The number of time steps for an initial state
        :param reward_prob: The probability that a location has a reward
            at an initial state
        :param reward_mean: The expectation of the log-normal distribution
            in the initial state reward
        :param reward_var: The variance of the log-normal distribution
            in the initial state reward
        :return:
        """
        try:
            data = np.loadtxt(data_file_name, skiprows=0, delimiter=' ')
        except IOError:
            data = np.zeros((0, self.num_input_units + 1))

        if verbose:
            print("Training set generation: {0}({1}) samples".
                  format(num_train_samples, num_train_samples * 8))
        for i in range(num_train_samples):
            state = self.generate_maze_example(time=time,
                                               reward_prob=reward_prob,
                                               reward_mean=reward_mean,
                                               reward_var=reward_var)
            actions = []
            states = []
            rewards = []
            mcts = MonteCarloSearchTree(initial_state=state,
                                        samples=num_mcts_samples)
            while not state.is_terminal:
                new_actions = mcts.search_for_actions(
                    search_depth=self._num_agents)
                for j in range(len(new_actions)):
                    act = new_actions[j]
                    states.append(state)
                    actions.append(act)
                    rewards.append(state.reward)
                    state = state.execute_action(act)
                    mcts.update_root(act)
            rewards = [state.reward - r for r in rewards]
            new_data = np.zeros((len(states) * 8, self.num_input_units + 1))
            for j in range(len(states)):
                state, action = states[j], actions[j]
                row = np.hstack((
                    self.state_action_to_array(state, action),
                    np.array([[rewards[j]]])))
                new_data[8 * j, :] = row.copy()
                row[[0], :self.num_input_units] = \
                    self.state_action_to_array(*self.flip_agents(state, action))
                new_data[8 * j + 1, :] = row

                state_x, action_x = self.flip_x(state, action)
                row[[0], :self.num_input_units] = \
                    self.state_action_to_array(state_x, action_x)
                new_data[8 * j + 2, :] = row
                row[[0], :self.num_input_units] = \
                    self.state_action_to_array(*self.flip_agents(state_x,
                                                                 action_x))
                new_data[8 * j + 3, :] = row

                state_y, action_y = self.flip_y(state, action)
                row[[0], :self.num_input_units] = \
                    self.state_action_to_array(state_y, action_y)
                new_data[8 * j + 4, :] = row
                row[[0], :self.num_input_units] = \
                    self.state_action_to_array(*self.flip_agents(state_y,
                                                                 action_y))
                new_data[8 * j + 5, :] = row

                state_xy, action_xy = self.flip_x(state_y, action_y)
                row[[0], :self.num_input_units] = \
                    self.state_action_to_array(state_xy, action_xy)
                new_data[8 * j + 6, :] = row
                row[[0], :self.num_input_units] = \
                    self.state_action_to_array(*self.flip_agents(state_xy,
                                                                 action_xy))
                new_data[8 * j + 7, :] = row
            data = np.vstack((data, new_data))
            if verbose and (i+1) % verbose == 0:
                print("\t{}% of the generation has been finished".format(
                    str(round((i+1) / num_train_samples * 100, 2))))
        np.savetxt(fname=data_file_name, X=data, delimiter=' ')
        return data

    @property
    def num_input_units(self) -> int:
        """ The number of input units in the neural network
            Each location has a unit for its reward
            Each agent has two units for its location
            The current agent id has a unit
            Time has a unit
            Each action of the four has a unit (hot one encoding)
        """
        return self._env_size ** 2 + self._num_agents * 2 + 6


if __name__ == '__main__':
    trainer = KolumboHeuristicsGenerator(9, 3)
    trainer.generate_training_data(num_train_samples=3, verbose=1)
    for i in range(10):
        print("Batch {0}".format(i))














