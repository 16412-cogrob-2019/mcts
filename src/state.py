import networkx as nx
from matplotlib.figure import Figure


class AbstractAction:
    pass


class AbstractState:
    @property
    def reward(self) -> float:
        raise NotImplementedError("The method not implemented")

    @property
    def is_terminal(self) -> bool:
        raise NotImplementedError("The method not implemented")

    @property
    def possible_actions(self) -> list:
        """ The possible actions to take at this state
            Make sure that the returned list is not empty unless the state is
            a terminal state
        :return: A list of possible states
        """
        raise NotImplementedError("The method not implemented")

    def execute_action(self, action: AbstractAction) -> "AbstractState":
        raise NotImplementedError("The method not implemented")


class KolumboAction(AbstractAction):
    def __init__(self, agent_id: int, start_loc: int, end_loc: int,
                 time_duration: float):
        self._agent_id = agent_id
        self._start_location = start_loc
        self._end_location = end_loc
        self._time = time_duration

    @property
    def agent_index(self) -> int:
        return self._agent_id

    @property
    def start_location(self) -> int:
        return self._start_location

    @property
    def goal_location(self) -> int:
        return self._end_location

    @property
    def time_duration(self) -> float:
        return self._time


class KolumboState(AbstractState):
    def __init__(self, environment: nx.DiGraph = nx.DiGraph(),
                 time_remains: float = 10.0, recovery_required: bool = False):
        """ Create a state of the Koloumb volcano exploration mission
        """
        # TODO: Add the planned path in
        # TODO: Add interfaces for ROS
        if time_remains < 0:
            raise ValueError("The remaining time cannot be negative")
        self._histories = []
        self._statuses = []
        self._terminal_locations = set()
        self._environment = environment
        self._time_remains = time_remains
        self._agent_id = 0  # The index for the agent that should take action
        self._recovery_required = recovery_required

    def __copy__(self) -> "KolumboState":
        """ Make a copy of the state; histories, statuses and terminal_locations
            are copied by value
        """
        new_state = KolumboState(environment=self._environment,
                                 time_remains=self._time_remains)
        new_state._histories = self._histories.copy()
        new_state._statuses = self._statuses.copy()
        new_state._terminal_locations = self._terminal_locations.copy()
        return new_state

    def add_location(self, location_id: int, reward: float) -> "KolumboState":
        """ Add a location with the specified reward
        """
        self._environment.add_node(location_id, reward=reward)
        return self

    def remove_location(self, location_id: int) -> "KolumboState":
        """ Remove a node and all adjacent edges
        """
        self._environment.remove_node(location_id)
        return self

    def set_reward(self, location_id: int, reward: float) -> "KolumboState":
        """ Update the reward at a specified node
        """
        nx.set_node_attributes(self._environment, {location_id: {
            'reward': reward}})
        return self

    def reset_environment(self) -> "KolumboState":
        """ Clear all locations and paths in the environment
        """
        self._environment = nx.DiGraph()
        return self

    @property
    def locations(self) -> dict:
        """ All possible locations and reward in the format
            {location_id: reward}
        """
        return nx.get_node_attributes(self._environment, 'reward')

    def reward_at_location(self, location_id: int) -> float:
        """ The reward at the specified location
        """
        return nx.get_node_attributes(self._environment, 'reward')[location_id]

    def add_path(self, start_location: int, end_location: int, cost: float) \
            -> "KolumboState":
        """ Add a path from start_location to end_location with the specified
            cost
        """
        self._environment.add_edge(start_location, end_location, cost=cost)
        return self

    def remove_path(self, start_location: int, end_location: int) \
            -> "KolumboState":
        """ Remove a path from start_location to end_location
        """
        self._environment.remove_edge(start_location, end_location)
        return self

    def set_cost(self, start_location: int, end_location: int, cost: float) \
            -> "KolumboState":
        """ Update the cost at a specified path
        """
        nx.set_edge_attributes(self._environment,
                               {(start_location, end_location): {'cost': cost}})
        return self

    @property
    def paths(self) -> dict:
        """ All possible paths and cost in the format
            {(start_id, end_id): cost}
        """
        return nx.get_edge_attributes(self._environment, 'cost')

    def cost_at_path(self, start_location: int, end_location: int) -> float:
        """ The cost of a specified path
        """
        return nx.get_edge_attributes(
            self._environment, 'cost')[(start_location, end_location)]

    def outgoing_paths(self, location_id: int) -> dict:
        """ Locations that can be reached from the specified location with a
            single-step action in the format {(location_id, end_location): cost}
        """
        return {(tup[0], tup[1]): tup[2]['cost'] for tup in
                self._environment.out_edges([location_id], data=True)}

    def add_agent(self, location_id: int) -> "KolumboState":
        """ Add an agent at the specified location
        """
        self._histories.append([location_id])
        self._statuses.append((location_id, location_id, 0.0))
        return self

    @property
    def nonterminal_agents(self) -> list:
        """ The list of agents that can still move; if time runs out, then no
            agent can move; if an agent reaches a terminal location, then it
            can no longer move
        """
        if self._time_remains <= 0:
            return []
        return [index for index in range(len(self._histories)) if
                self._statuses[index][0] not in
                self._terminal_locations]

    def evolve(self) -> "KolumboState":
        """ Evolve the state so that one agent finished the ongoing action
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
                self._histories[agent].append(end_loc)
        self._time_remains -= time_elapsed
        return self

    @property
    def visited(self) -> set:
        """ The set of all visited locations
        """
        return set(location for history in self._histories
                   for location in history)

    @property
    def is_recovered(self) -> bool:
        """ Whether all agents have reached a terminal location (if required)
            Return True when recovery is not required
        """
        if self._terminal_locations:
            return all(self._statuses[robot][0] in self._terminal_locations
                       for robot in range(len(self._histories)))
        else:
            return True

    @property
    def reward(self) -> float:
        """ The reward at a terminal state
            If a state is not terminal, or if not all robots are recovered
            while required, return 0 reward
        """
        if not self.is_terminal or not self.is_recovered:
            return 0.0
        return sum(reward for loc, reward in self.locations.items() if
                   loc in self.visited)

    @property
    def is_terminal(self) -> bool:
        """ Whether a state is terminal
            If time runs out, then a state is terminal; if all agents reach a
            terminal location, then a state is terminal
        """
        return len(self.nonterminal_agents) == 0

    @property
    def possible_actions(self) -> list:
        """ The possible actions to take at this state
            Make sure that the returned list is not empty unless the state is
            a terminal state
        :return: A list of possible actions
        """
        start_loc = self._statuses[self._agent_id][0]
        return [KolumboAction(self._agent_id, start_loc, path[1],
                              self.cost_at_path(*path))
                for path in self.outgoing_paths(start_loc)]

    def execute_action(self, action: KolumboAction) -> "KolumboState":
        """ Execute the action on a copy of the current state
        :param action: The action io take
        :return: A copy of the state after the action is executed
        """
        new_state = self.__copy__()
        new_state._statuses[action.agent_index] = (action.start_location,
                                                   action.goal_location,
                                                   action.time_duration)
        new_state.evolve()
        return new_state

    def visualize(self) -> Figure:
        raise NotImplementedError("Visualization method not implemented")

    def input_from_ros(self, whatever_args) -> "KolumboState":
        raise NotImplementedError("ROS interface not implemented")


if __name__ == '__main__':
    env = nx.DiGraph()
    env.add_node(1, reward=1.0)
    env.add_node(2, reward=2.0)
    env.add_edge(1, 2, cost=2.4)
    print(env.out_edges([1], data=True))
