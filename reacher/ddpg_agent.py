import numpy as np
import torch
import random
import torch.nn.functional as F
import torch.optim as optimum

from models import Actor,Critic
from storage import ReplayBuffer


BATCH_SIZE = 256
BUFFER_SIZE = int(1e5)
GAMMA = 0.9
TAU = 1e-3
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):

    def __init__(self,state_dim,action_dim,seed):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(seed)

        # Actor Network (Local / Target networks)
        self.actor_local = Actor(state_dim,action_dim,self.seed).to(device)
        self.actor_target = Actor(state_dim,action_dim,self.seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optimum.Adam(self.actor_local.parameters(),lr=LR_ACTOR)

        # Twin Critic Network(s) (Local / Target networks)
        self.critic_local = Critic(state_dim,action_dim,self.seed).to(device)
