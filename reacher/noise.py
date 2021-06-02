import copy
import random
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class GuassianNoise(object):

    def __init__(self,noise_clip,size,low,high):
        """
        Returns Guassian Noise
        :param noise:
        :param size:
        :param low:
        :param high:
        """

        self.noise_clip = noise_clip
        self.size = size
        self.low = low
        self.high = high

    def sample(self,batch_actions,policy_noise):
        noise = torch.Tensor(batch_actions).data.normal_(0,policy_noise).to(device)
        noise = noise.clamp(-self.noise_clip,self.noise_clip)
        return noise.detach().cpu().numpy()


