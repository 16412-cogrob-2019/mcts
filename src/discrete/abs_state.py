import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.patches import Circle, FancyArrow, Rectangle, Polygon
from matplotlib import transforms
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