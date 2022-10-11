from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class Agent(ABC):

    def __init__(self, preference, endowment):
        self.endowment = endowment
        self.preference = preference

    @abstractmethod
    def act(self, seed):
        pass


class Subject(ABC):

    def __init__(self, preference, endowment, softmax_temp):
        self.softmax_temp = softmax_temp
        self.preference = preference
        self.endowment = endowment
        self.posterior_beliefs = None

    @abstractmethod
    def act(self, seed, offer):
        pass
