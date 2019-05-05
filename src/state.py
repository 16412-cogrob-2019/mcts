import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from abc import ABCMeta
from copy import deepcopy


class AbstractAction:
    __metaclass__ = ABCMeta


class AbstractState:
    __metaclass__ = ABCMeta

    @property
    def reward(self):
        # type: () -> float
        raise NotImplementedError("The method not implemented")

    @property
    def is_terminal(self):
        # type: () -> bool
        raise NotImplementedError("The method not implemented")

    @property
    def possible_actions(self):
        # type: () -> list
        """ The possible actions to take at this state
            Make sure that the returned list is not empty unless the state is
            a terminal state
        :return: A list of possible states
        """
        raise NotImplementedError("The method not implemented")

    def execute_action(self, action):
        # type: (AbstractAction) -> AbstractState
        """ Execute the specified action on a copy of the current state
        :param action
        :return: The copy of the updated state
        """
        raise NotImplementedError("The method not implemented")


class KolumboAction(AbstractAction):
    def __init__(self, agent_id, start_loc, end_loc, time_duration):
        # type: (int, int, int, float) -> None
        self._agent_id = agent_id
        self._start_location = start_loc
        self._end_location = end_loc
        self._time = time_duration

    @property
    def agent_index(self):
        # type: () -> int
        return self._agent_id

    @property
    def start_location(self):
        # type: () -> int
        return self._start_location

    @property
    def goal_location(self):
        # type: () -> int
        return self._end_location

    @property
    def time_duration(self):
        # type: () -> float
        return self._time

    def __str__(self):
        # type: () -> str
        return "Action: agent {0} move from location {1} to location {2} " \
               "in time {3}".format(self._agent_id, self._start_location,
                                    self._end_location, self._time)


class KolumboState(AbstractState):
    def __init__(self, environment=nx.DiGraph(), time_remains=10.0):
        # type: (nx.DiGraph, float) -> None
        """ Create a state of the Kolumbo volcano exploration mission
            statuses[agent_id] describes the moving condition of an agent in the
            format (start_node, end_node, time_remaining_to_get_to_end_node);
            when start_node and end_node are the same, the agent is exactly at
            that node
        """
        # TODO: Add interfaces for ROS
        if time_remains < 0:
            raise ValueError("The remaining time cannot be negative")
        self._histories = []
        self._statuses = []
        self._terminal_locations = set()
        self._environment = environment
        self._time_remains = time_remains
        self._agent_id = 0  # The index for the agent that should take action

    def __copy__(self):
        # type: () -> KolumboState
        """ Make a copy of the state; histories, statuses and terminal_locations
            are copied by value
        """
        new_state = KolumboState(environment=self._environment,
                                 time_remains=self._time_remains)
        new_state._histories = deepcopy(self._histories)
        new_state._statuses = deepcopy(self._statuses)
        new_state._terminal_locations = deepcopy(self._terminal_locations)
        return new_state

    def __str__(self):
        # type: () -> str
        res = ""
        for i in range(len(self._statuses)):
            status = self._statuses[i]
            if i != 0:
                res += "\n"
            res += "Agent {0} ".format(i)
            if status[2] == 0:
                res += "is at location {0}".format(status[0])
            else:
                res += "is moving from location {0} to location {1} in " \
                       "time {2}".format(status[0], status[1], status[2])
        return res

    def json_parse_to_map(self, json_map):
        # type: (dict) -> KolumboState
        """ Parses incoming data to create new graph
        """
        for node in json_map:
            node_id = node['node_id']
            has_agent = node['has_agent']
            reward = node['node_reward']
            x = node['x']
            y = node['y']
            connected_to = node['connectivity']
            costs = node['costs']
            paths = node['paths']

            # add locations to graph
            self.add_location(node_id, reward, [x,y])

            # add connectivity to graph
            for con_node, con_cost, con_path in zip(connected_to, costs, paths):
                self.add_path(node_id, con_node, con_cost, con_path)

            # add if agent is there
            if has_agent != 0:
                self.add_agent(node_id)

        return self

    def set_location_terminal(self, location_id, is_terminal=True):
        # type: (int, bool) -> KolumboState
        """ Set a location to be a terminal or nonterminal location
        """
        if is_terminal:
            self._terminal_locations.add(location_id)
        else:
            self._terminal_locations.remove(location_id)
        return self

    def add_location(self, location_id, reward, coord):
        # type: (int, float, (float, float)) -> KolumboState
        """ Add a location with the specified reward and coordinates
        """
        self._environment.add_node(location_id, reward=reward, coord=coord)
        return self

    def remove_location(self, location_id):
        # type: (int) -> KolumboState
        """ Remove a node and all adjacent edges
        """
        self._environment.remove_node(location_id)
        return self

    def set_location_reward(self, location_id, reward):
        # type: (int, float) -> KolumboState
        """ Update the reward at a specified node
        """
        nx.set_node_attributes(self._environment, name='reward',
                               values={location_id: reward})
        return self

    def set_location_coord(self, location_id, coord):
        # type: (int, (float, float)) -> KolumboState
        """ Update the coordinates at a specified node
        """
        nx.set_node_attributes(self._environment, name='coord',
                               values={location_id: coord})
        return self

    def reset_environment(self):
        # type: () -> KolumboState
        """ Clear all rewards_at_all_locations and costs_at_all_paths in the environment
        """
        self._environment = nx.DiGraph()
        return self

    @property
    def rewards_at_all_locations(self):
        # type: () -> dict
        """ All possible locations and rewards in the format
            {location_id: reward}
        """
        return nx.get_node_attributes(self._environment, 'reward')

    def reward_at_location(self, location_id):
        # type: (int) -> float
        """ The reward at the specified location
        """
        return nx.get_node_attributes(self._environment, 'reward')[location_id]

    @property
    def coord_at_all_locations(self):
        # type: () -> dict
        """ All possible locations and coordinates in the format
            {location_id: coord}
        """
        return nx.get_node_attributes(self._environment, 'coord')

    def add_path(self, start_location, end_location, cost, trajectory=None):
        # type: (int, int, float, list) -> KolumboState
        """ Add a path from start_location to end_location with the specified
            cost and path (if existing)
        """
        if trajectory:
            self._environment.add_edge(start_location, end_location, cost=cost,
                                       trajectory=trajectory)
        else:
            self._environment.add_edge(start_location, end_location, cost=cost)
        return self

    def remove_path(self, start_location, end_location):
        # type: (int, int) -> KolumboState
        """ Remove a path from start_location to end_location
        """
        self._environment.remove_edge(start_location, end_location)
        return self

    def set_cost(self, start_location, end_location, cost):
        # type: (int, int, float) -> KolumboState
        """ Update the cost at a specified path
        """
        nx.set_edge_attributes(self._environment, name='cost',
                               values={(start_location, end_location): cost})
        return self

    def set_trajectory(self, start_location, end_location, trajectory):
        # type: (int, int, list) -> KolumboState
        """ Update the trajectory of a specified path
        """
        nx.set_edge_attributes(self._environment, name='trajectory', values={
            (start_location, end_location): trajectory})
        return self

    @property
    def costs_at_all_paths(self):
        # type: () -> dict
        """ All possible paths and costs in the format
            {(start_id, end_id): cost}
        """
        return nx.get_edge_attributes(self._environment, 'cost')

    def cost_at_path(self, start_location, end_location):
        # type: (int, int) -> float
        """ The cost of a specified path
        """
        return nx.get_edge_attributes(
            self._environment, 'cost')[(start_location, end_location)]

    @property
    def trajectories_at_all_paths(self):
        # type: () -> dict
        """ All possible paths and trajectories in the format
            {(start_id, end_id): trajectory}
        """
        return nx.get_edge_attributes(self._environment, 'trajectory')

    def outgoing_paths(self, location_id):
        # type: (int) -> dict
        """ Locations that can be reached from the specified location with a
            single-step action in the format {(location_id, end_location): cost}
        """
        return {(tup[0], tup[1]): tup[2]['cost'] for tup in
                self._environment.out_edges([location_id], data=True)}

    def add_agent(self, location_id):
        # type: (int) -> KolumboState
        """ Add an agent at the specified location
        """
        self._histories.append([location_id])
        self._statuses.append((location_id, location_id, 0.0))
        return self

    @property
    def nonterminal_agents(self):
        # type: () -> list
        """ The list of agents that can still move; if time runs out, then no
            agent can move; if an agent reaches a terminal location, then it
            can no longer move
        """
        if self._time_remains <= 0:
            return []
        return [index for index in range(len(self._histories)) if
                self._statuses[index][0] not in
                self._terminal_locations]

    def evolve(self):
        # type: () -> KolumboState
        """ Evolve the state so that one agent finishes the ongoing action
            Update the time remaining, histories, statuses, and the index of the
            agent that should take the next action
        """
        if self.is_terminal:
            return self
        self._agent_id = min(self.nonterminal_agents,
                             key=lambda robot: self._statuses[robot][2])
        time_elapsed = min(self._time_remains,
                           self._statuses[self._agent_id][2])
        for agent in range(len(self._histories)):
            start_loc, end_loc, time_remains = self._statuses[agent]
            if agent != self._agent_id:
                self._statuses[agent] = (start_loc, end_loc,
                                         time_remains - time_elapsed)
            else:
                self._statuses[agent] = (end_loc, end_loc, 0.0)
                if time_elapsed != 0:
                    self._histories[agent].append(end_loc)
        self._time_remains -= time_elapsed
        return self

    @property
    def visited(self):
        # type: () -> set
        """ The set of all visited rewards_at_all_locations
        """
        return set(location for history in self._histories
                   for location in history)

    @property
    def is_recovered(self):
        # type: () -> bool
        """ Whether all agents have reached a terminal location (if required)
            Return True when recovery is not required (no terminal locations)
        """
        if self._terminal_locations:
            return all(self._statuses[robot][0] in self._terminal_locations
                       for robot in range(len(self._histories)))
        else:
            return True

    @property
    def reward(self):
        # type: () -> float
        """ The reward at a terminal state
            If a state is not terminal, or if not all robots are recovered
            while required, return 0 reward
        """
        if not self.is_terminal or not self.is_recovered:
            return 0.0
        return sum(
            reward for loc, reward in self.rewards_at_all_locations.items() if
            loc in self.visited)

    @property
    def is_terminal(self):
        # type: () -> bool
        """ Whether a state is terminal
            If time runs out, then a state is terminal; if all agents reach a
            terminal location, then a state is terminal
        """
        return len(self.nonterminal_agents) == 0

    @property
    def possible_actions(self):
        # type: () -> list
        """ The possible actions to take at this state
            Make sure that the returned list is not empty unless the state is
            a terminal state
        :return: A list of possible actions
        """
        start_loc = self._statuses[self._agent_id][0]
        return [KolumboAction(self._agent_id, start_loc, path[1],
                              self.cost_at_path(*path))
                for path in self.outgoing_paths(start_loc)]

    def execute_action(self, action):
        # type: (KolumboAction) -> KolumboState
        """ Execute the action on a copy of the current state
        :param action: The action to take
        :return: A copy of the state after the action is executed
        """
        new_state = self.__copy__()
        new_state._statuses[action.agent_index] = (action.start_location,
                                                   action.goal_location,
                                                   action.time_duration)
        new_state.evolve()
        return new_state

    def visualize(self, file_name=None, fig_size=(15, 15), buffer_size=0.25,
                  max_reward_radius=0.35, min_reward_radius=0.15,
                  visited_reward_transparency=0.25, trajectory_width=10.0,
                  agent_radius=0.1, boundary_width=0.05):
        # type: (str, (float, float), float, float, float, float, float, float, float) -> Figure
        colors = {'reward': 'deepskyblue', 'trajectory': 'peachpuff',
                  'boundary': 'firebrick', 'agent': 'darkorange'}
        z = {'reward': 1, 'trajectory': 2, 'boundary': 3, 'agent': 4}
        coords = self.coord_at_all_locations
        rewards = self.rewards_at_all_locations
        costs = self.costs_at_all_paths
        title_font = {'fontname': 'Sans Serif', 'size': '16', 'color': 'black',
                      'weight': 'bold'}
        x_min = min(coord[0] for coord in coords.values()) - buffer_size \
            - max_reward_radius
        x_max = max(coord[0] for coord in coords.values()) + buffer_size \
            + max_reward_radius
        y_min = min(coord[1] for coord in coords.values()) - buffer_size \
            - max_reward_radius
        y_max = max(coord[1] for coord in coords.values()) + buffer_size \
            + max_reward_radius

        # Initialize the figure
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

        # Plot boundaries
        plt.hlines(y=[y_min, y_max], xmin=x_min-boundary_width,
                   xmax=x_max+boundary_width, color=colors['boundary'],
                   linewidth=boundary_width, zorder=z['boundary'])
        plt.vlines(x=[x_min, x_max], ymin=y_min-boundary_width,
                   ymax=y_max+boundary_width, color=colors['boundary'],
                   linewidth=boundary_width, zorder=z['boundary'])

        # Plot rewards
        max_reward = max(rewards.values())
        non_zero_rewards = [reward for reward in rewards.values()
                            if reward != 0]
        min_reward = 0 if non_zero_rewards == [] else min(non_zero_rewards)
        for node, location in coords.items():
            reward = self.reward_at_location(node)
            reward_radius = ((reward - min_reward) *
                             (max_reward_radius - min_reward_radius) /
                             (max_reward - min_reward) + min_reward_radius)
            x, y = location
            ax.add_patch(Circle(xy=(x, y), radius=reward_radius,
                         facecolor=colors['reward'],
                         alpha=(visited_reward_transparency
                                if node in self.visited else 1.0),
                         zorder=z['reward']))

        # Plot agents and trajectories
        for k in range(len(self._histories)):
            agent_history = self._histories[k]
            # Plot trajectories for completed actions
            for i in range(1, len(agent_history)):
                prev_loc_id = agent_history[i-1]
                cur_loc_id = agent_history[i]
                prev_loc = coords[prev_loc_id]
                cur_loc = coords[cur_loc_id]
                ax.add_line(Line2D(xdata=(prev_loc[0], cur_loc[0]),
                                   ydata=(prev_loc[1], cur_loc[1]),
                                   linewidth=trajectory_width,
                                   color=colors['trajectory'],
                                   zorder=z['trajectory']))
            # Plot trajectories for ongoing actions
            status = self._statuses[k]
            if status[2] == 0:
                x_c, y_c = coords[status[0]]
            else:
                cost = costs[(status[0], status[1])]
                x_s, y_s = coords[status[0]]
                x_e, y_e = coords[status[1]]
                x_c = x_e - status[2] / cost * (x_e - x_s)
                y_c = y_e - status[2] / cost * (y_e - y_s)
                ax.add_line(Line2D(xdata=(x_s, x_c), ydata=(y_s, y_c),
                                   linewidth=trajectory_width,
                                   color=colors['trajectory'],
                                   zorder=z['trajectory']))
            # Plot agents
            ax.add_patch(Circle(xy=(x_c, y_c), radius=agent_radius,
                         edgecolor=colors['agent'], facecolor=colors['agent'],
                         alpha=1.0, zorder=z['agent']))

        # Plotting
        plt.title(
            'Agents Trajectories \n Accumulated Reward: ' + str(
                sum(reward for loc, reward in
                    self.rewards_at_all_locations.items() if
                    loc in self.visited)), title_font)
        plt.xlabel('x', title_font)
        plt.ylabel('y', title_font)
        ax.grid(False)
        ax.axis('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Save and display
        plt.show()
        if file_name is not None:
            plt.savefig(file_name)

        return fig
