import copy
import random
import numpy as np

class OUNoise(object):
    """
    Ornstein-Unlenbeck process
    """
    def __init__(self,size,seed,mu=0.,theta=0.15,sigma=0.01):
        """
        Initialize parameters
        :param size:
        :param seed:
        :param mu:
        :param theta:
        :param sigma:
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = copy.copy(self.mu)

    def reset(self):
        self.state =copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu -x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state
