from state import *
import random
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Rectangle, Circle
import numpy as np
from copy import deepcopy


class MazeAction(AbstractAction):
    def __init__(self, agent_index: int, position: (int, int)):
        self._agent_index = agent_index
        self._position = position

    @property
    def agent_index(self) -> int:
        return self._agent_index

    @property
    def position(self) -> (int, int):
        return self._position

    def __eq__(self, other) -> bool:
        return (self.__class__ == other.__class__ and
                self.agent_index == other.agent_index and
                self.position == other.position)

    def __hash__(self) -> int:
        return hash(tuple([self._agent_index] + list(self._position)))

    def __str__(self) -> str:
        return "Action: Agent {0} moves to {1}".format(self.agent_index,
                                                       self.position)


class MazeEnvironment:
    def __init__(self, xlim: (int, int) = (0, 1), ylim: (int, int) = (0, 1),
                 obstacles: set = None, targets: dict = None,
                 is_border_obstacle_filled: bool = True):
        """ Create an Environment object in which the AUV problem is defined
        """
        self._x_min, self._x_max = xlim
        self._y_min, self._y_max = ylim
        self._obstacles = obstacles if obstacles else set()
        self._targets = targets if targets else {}

        # Make border obstacles if specified
        if is_border_obstacle_filled:
            for j in range(self.y_min, self.y_max + 1):
                self.add_obstacle((self.x_min, j)).add_obstacle((self.x_max, j))
            for i in range(self.x_min, self.x_max + 1):
                self.add_obstacle((i, self.y_min)).add_obstacle((i, self.y_max))

        # Remove target regions that were defined within obstacles
        for position in self._obstacles:
            if position in self._targets.keys():
                del self._targets[position]

    @property
    def x_min(self) -> int:
        return self._x_min

    @property
    def y_min(self) -> int:
        return self._y_min

    @property
    def x_max(self) -> int:
        return self._x_max

    @property
    def y_max(self) -> int:
        return self._y_max

    @property
    def max_reward(self) -> float:
        if self._targets:
            return max(self._targets.values())
        else:
            return 0.0

    @property
    def x_range(self) -> int:
        return self._x_max + 1 - self._x_min

    @property
    def y_range(self) -> int:
        return self._y_max + 1 - self._y_min

    @property
    def obstacles(self) -> set:
        return deepcopy(self._obstacles)

    @property
    def rewards(self) -> dict:
        return deepcopy(self._targets)

    def add_obstacle(self, obstacle_position: (int, int)) -> "MazeEnvironment":
        if obstacle_position not in self._targets:
            self._obstacles.add(obstacle_position)
        return self

    def remove_obstacle(self, obstacle_position: (int, int)) \
            -> "MazeEnvironment":
        if obstacle_position in self._obstacles:
            self._obstacles.remove(obstacle_position)
            return self

    def add_reward(self, position: (int, int), reward: float) \
            -> "MazeEnvironment":
        if position not in self._obstacles:
            self._targets[position] = reward
        return self

    def remove_reward(self, position: (int, int)) -> "MazeEnvironment":
        if position in self._targets:
            del self._targets[position]
        return self

    def __eq__(self, other) -> bool:
        return (self.__class__ == other.__class and self.x_max == other.x_max
                and self.x_min == other.x_min and self.y_min == other.y_min and
                self.y_max == other.y_max and
                self._obstacles == other.obstacles and
                self._targets == other.rewards)


def gen_random_environment(xlim: (int, int) = (0, 10),
                           ylim: (int, int) = (0, 10),
                           obstacle_coverage: float = 0.2,
                           target_coverage: float = 0.2,
                           reward_range: (float, float) = (1.0, 3.0),
                           is_border_obstacle_filled: bool = True):
    if not (obstacle_coverage >= 0 and target_coverage >= 0 and
            obstacle_coverage + target_coverage <= 1):
        raise ValueError("The probability is not valid")

    env = MazeEnvironment(xlim, ylim,
                          is_border_obstacle_filled=is_border_obstacle_filled)

    # Generate obstacles and targets by the given probability
    if obstacle_coverage > 0 or target_coverage > 0:
        for i in range(ylim[0], ylim[1] + 1):
            for j in range(xlim[0], xlim[1] + 1):
                if (i, j) not in env.obstacles:
                    r = random.random()
                    if r <= obstacle_coverage:
                        env.add_obstacle((i, j))
                    elif r <= obstacle_coverage + target_coverage:
                        env.add_reward((i, j),
                                       random.randrange(reward_range[0],
                                                        reward_range[1]))
    return env


class MazeState(AbstractState):
    def __init__(self, environment: MazeEnvironment, time_remains: int = 10):
        """ Create a state of the AUV reward-collection game
        """
        if time_remains < 0:
            raise ValueError("The remaining time cannot be negative")
        self._paths = []
        self._environment = environment
        self._time_remains = time_remains
        self._turn = 0  # The index of which agent should move next
        self._num_executed_actions = 0
        self._collaborator_actions = {}
        self._collaborator_sigma = {}
        self._collaborator_expectations = {}

    def __copy__(self) -> "MazeState":
        """ Deep copy does not apply to the Environment object because
            it is supposed to be static
        """
        copy = MazeState(self._environment)
        copy._time_remains = self._time_remains
        copy._paths = deepcopy(self._paths)
        copy._turn = self._turn
        copy._num_executed_actions      = self._num_executed_actions
        copy._collaborator_actions      = deepcopy(self._collaborator_actions)
        copy._collaborator_sigma        = deepcopy(self._collaborator_sigma)
        copy._collaborator_expectations = deepcopy(self._collaborator_expectations)
        return copy

    def is_in_range(self, position: (int, int)) -> bool:
        return (self._environment.x_min <= position[0] <=
                self._environment.x_max and self._environment.y_min
                <= position[1] <= self._environment.y_max)

    def add_agent(self, position: (int, int)) -> "MazeState":
        if (self.is_in_range(position) and
                position not in self._environment.obstacles):
            self._paths.append([position])
        else:
            raise ValueError("The given position is invalid")
        return self

    def add_collaborator(self, actions, sigma):
        self._collaborator_actions = actions
        self._collaborator_sigma = sigma
        self._collaborator_expectations = {}

        # compute expected probabilities of collaborator reaching a particular 
        #   state for each time-step being considered
        time_steps = len(self._collaborator_actions)
        for t in range(time_steps):
            self._collaborator_expectations[t] = {}

            # consider 'past' actions that are known with certainty 
            for past_action in self._collaborator_actions[:t]:
                self._collaborator_expectations[t][past_action] = 1

            # consider 'future' actions that are predicted based on uncertainty
            for future_dist in self._collaborator_sigma[t:]:
                for future_action in future_dist:
                    p = future_dist[future_action] / sum(future_dist.values())
                    if (future_action in self._collaborator_expectations[t]):
                        self._collaborator_expectations[t][future_action] += p
                    else:
                        self._collaborator_expectations[t][future_action] = p

    @property
    def paths(self) -> list:
        return self._paths

    @property
    def environment(self) -> MazeEnvironment:
        return self._environment

    @property
    def visited(self) -> set:
        visited = []
        for path in self._paths:
            visited += path
        return set(visited)

    @property
    def reward(self) -> float:
        reward = 0.0
        # reward collected by the ownship
        for target_position in self._environment.rewards:
            if target_position in self.visited:
                reward += self._environment.rewards[target_position]
        
        # reward collected by the collaborators
        if self._collaborator_actions != {}:
            time_steps = min(len(self._paths[0]), 
                             len(self._collaborator_expectations))

            for collaborator_position in self._collaborator_expectations[time_steps-1]:
                if (collaborator_position not in self.visited) and \
                   (collaborator_position in self.environment.rewards):
                   prob = min(self._collaborator_expectations[time_steps-1][collaborator_position]**2,1)
                   reward += self._environment.rewards[collaborator_position] * prob

        # reward collected by the collaborator
        return reward

    def switch_agent(self) -> "MazeState":
        """ After the movement of one agent, it would be the turn of the next
            agent
        """
        self._turn = (self._turn + 1) % len(self._paths)
        if self._turn == 0:  # When all agents have taken a turn of actions
            self._time_remains -= 1
        return self

    @property
    def time_remains(self) -> int:
        return self._time_remains

    @property
    def is_terminal(self) -> bool:
        """ A state is terminal if and only if the time runs out
        """
        return self._time_remains <= 0

    @property
    def turn(self) -> int:
        return self._turn

    def execute_action(self, action: MazeAction) -> "MazeState":
        """ Make a copy of the current state, execute the action and return the
            new state
        :param action: The action
        :return: A copy of the new state
        """
        new_state = self.__copy__()
        new_state._num_executed_actions += 1;
        new_state.paths[new_state.turn].append(action.position)
        new_state.switch_agent()
        return new_state

    @property
    def possible_actions(self) -> list:
        i, j = self._paths[self._turn][-1]
        actions = [MazeAction(self._turn, (i + 1, j)),
                   MazeAction(self._turn, (i - 1, j)),
                   MazeAction(self._turn, (i, j + 1)),
                   MazeAction(self._turn, (i, j - 1))]
        return [action for action in actions if
                self.is_in_range(action.position) and
                action.position not in self._environment.obstacles]

    def visualize(self, file_name=None, fig_size: (float, float) = (6.5, 6.5),
                  size_auv_path: float = 0.8, size_max_radius: float = 0.3,
                  size_min_radius: float = 0.1,
                  tick_size: float = 14, grid_width: float = 0.25,
                  size_arrow_h_width: float = 0.4,
                  size_arrow_h_length: float = 0.3,
                  size_arrow_width: float = 0.4,
                  color_obstacle: str = 'firebrick',
                  color_target: str = 'deepskyblue',
                  color_auv: str = 'darkorange',
                  color_auv_path: str = 'peachpuff',
                  visited_reward_opacity: float = 0.15) -> Figure:

        ownship_ship_color = 'darkorange'
        ownship_traj_color = 'orange'
        collab_ship_color  = 'darkorchid'
        collab_traj_color  = 'orchid'
        collab_unty_color  = 'lightpink' 

        if (fig_size[0] <= 0 or fig_size[1] <= 0 or size_auv_path <= 0 or
                size_max_radius <= 0 or size_arrow_h_width <= 0 or
                size_arrow_h_length <= 0 or size_arrow_width <= 0 or
                tick_size <= 0 or grid_width <= 0):
            raise ValueError("Size must be positive")
        max_reward = self._environment.max_reward
        title_font = {'fontname': 'Sans Serif', 'size': '16', 'color': 'black',
                      'weight': 'bold'}
        z = {'auv_path': 1, 'target': 2, 'obstacle': 4, 'auv': 5, 
             'collab_uncertainty': 0}
        opacity_threshold = 0.2
        half_len = 0.5
        edge_width = 1.5

        # Initialize the figure
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        plt.hlines(y=range(self._environment.y_min, self._environment.y_max + 1)
                   , xmin=self._environment.x_min, xmax=self._environment.x_max,
                   color='k', linewidth=grid_width, zorder=0)
        plt.vlines(x=range(self._environment.x_min, self._environment.x_max + 1)
                   , ymin=self._environment.y_min, ymax=self._environment.y_max,
                   color='k', linewidth=grid_width, zorder=0)

        # Plot obstacles
        for i, j in self._environment.obstacles:
            ax.add_patch(Rectangle(xy=(i, j), width=1, height=1,
                                   color=color_obstacle, zorder=z['obstacle']))

        # Plot rewards
        for position, reward in self._environment.rewards.items():
            target_radius = ((reward / max_reward)
                             * (size_max_radius - size_min_radius)
                             + size_min_radius)
            centroid  = (position[0] + half_len, position[1] + half_len)

            # change the behavior depending on ownship vs. collaborator visit
            edgecolor = None 
            alpha     = 1
            if (position in self.visited):
                alpha = visited_reward_opacity

            elif self._collaborator_actions != {}:
                first_t = min(self._collaborator_expectations.keys())
                if (position in self._collaborator_actions):
                    alpha = visited_reward_opacity  
                elif (position in self._collaborator_expectations[first_t] and 
                    self._collaborator_expectations[first_t][position] > opacity_threshold):
                    alpha     = visited_reward_opacity
                    edgecolor = 'navy'

            # Plot the reward region
            ax.add_patch(Circle(xy=centroid, radius=target_radius,
                                color=color_target, zorder=z['target'],
                                alpha=alpha, ec=edgecolor,
                                linewidth=edge_width))

        # Plot collaborator ship 
        if self._collaborator_actions != {}:
            x, y = self._collaborator_actions[-1]
            dx, dy = 0, 1
            if len(self._collaborator_actions) >= 2:
                x_p, y_p = self._collaborator_actions[-2]
                if x == x_p + 1 and y == y_p:
                    dx, dy = 1, 0
                elif x == x_p - 1 and y == y_p:
                    dx, dy = -1, 0
                elif x == x_p and y == y_p - 1:
                    dx, dy = 0, -1
            x += half_len * float(1 - dx)
            y += half_len * float(1 - dy)
            ax.add_patch(FancyArrow(x=x, y=y, dx=dx, dy=dy, fc=collab_ship_color,
                                    width=size_arrow_width,
                                    head_width=size_arrow_h_width,
                                    head_length=size_arrow_h_length,
                                    zorder=z['auv'], length_includes_head=True))

            # Plot collaborator trajectories
            for j in range(1, len(self._collaborator_actions)):
                x, y = self._collaborator_actions[j]
                x_p, y_p = self._collaborator_actions[j - 1]
                ax.add_line(Line2D(xdata=(x + half_len, x_p + half_len),
                                   ydata=(y + half_len, y_p + half_len),
                                   linewidth=size_auv_path * 10,
                                   color=collab_traj_color, zorder=z['auv_path']))

            # Plot collaborator uncertainties 
            for i, j in self._collaborator_sigma[-1]:
                ax.add_patch(Rectangle(xy=(i, j), width=1, height=1,
                                       color=collab_unty_color, 
                                       zorder=z['collab_uncertainty'],
                                       alpha=(self._collaborator_sigma[-1][(i,j)] / 
                                              sum(self._collaborator_sigma[-1].values()))**(1/4)))

        # Plot agent ships
        for i in range(len(self._paths)):
            x, y = self._paths[i][-1]
            dx, dy = 0, 1
            if len(self._paths[i]) >= 2:
                x_p, y_p = self._paths[i][-2]
                if x == x_p + 1 and y == y_p:
                    dx, dy = 1, 0
                elif x == x_p - 1 and y == y_p:
                    dx, dy = -1, 0
                elif x == x_p and y == y_p - 1:
                    dx, dy = 0, -1
            x += half_len * float(1 - dx)
            y += half_len * float(1 - dy)
            ax.add_patch(FancyArrow(x=x, y=y, dx=dx, dy=dy, fc=ownship_ship_color,
                                    width=size_arrow_width,
                                    head_width=size_arrow_h_width,
                                    head_length=size_arrow_h_length,
                                    zorder=z['auv'], length_includes_head=True))
            # Plot agent trajectories
            for j in range(1, len(self._paths[i])):
                x, y = self._paths[i][j]
                x_p, y_p = self._paths[i][j - 1]
                ax.add_line(Line2D(xdata=(x + half_len, x_p + half_len),
                                   ydata=(y + half_len, y_p + half_len),
                                   linewidth=size_auv_path * 10,
                                   color=ownship_traj_color, zorder=z['auv_path']))



        # Plotting
        plt.title('AUV Trajectory \n Accumulated Reward: ' + 
            str(math.floor(self.reward)), title_font)
        plt.xlabel('x', title_font)
        plt.ylabel('y', title_font)
        x_ticks = np.arange(self._environment.x_min, self._environment.x_max + 1
                            , 1)
        y_ticks = np.arange(self._environment.y_min, self._environment.y_max + 1
                            , 1)
        plt.xticks(x_ticks + half_len, x_ticks.astype(int))
        plt.yticks(y_ticks + half_len, y_ticks.astype(int))
        ax.tick_params(labelsize=tick_size)
        ax.grid(False)
        ax.axis('equal')
        ax.set_xlim(self._environment.x_min - half_len,
                    self._environment.x_max + half_len + 1)
        ax.set_ylim(self._environment.y_min - half_len,
                    self._environment.y_max + half_len + 1)

        # Save and display
        plt.show()
        if file_name is not None:
            plt.savefig(file_name)

        return fig
