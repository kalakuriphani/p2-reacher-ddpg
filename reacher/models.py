import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor (Policy) Model
    """
    def __init__(self,state_dim,action_dim,units=(128,64),seed=0):
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer_1 = nn.Linear(state_dim,units(0))
        self.layer_2 = nn.linear(units(0),units(1))
        self.layer_3 = nn.Linear(units(1),action_dim)

    def forward(self,state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        return F.tanh(self.layer_3(x))



class Critic(nn.Module):
    """
    Crtic (Value) Model
    """
    def __init__(self,state_dim,action_dim,units=(128,64),seed=0):
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        # Critic model 1
        self.layer_1 = nn.Linear(state_dim+action_dim,units(0))
        self.layer_2 = nn.Linear(units(0),units(1))
        self.layer_3 = nn.Linear(units(1),action_dim)
        # Critic model 2
        self.layer_4 = nn.Linear(state_dim + action_dim, units(0))
        self.layer_5 = nn.Linear(units(0), units(1))
        self.layer_6 = nn.Linear(units(1), action_dim)

    def forward(self,state,action):
        """
        Forward propogation on both two critic models
        :param state:
        :param action:
        :return:
        """
        xu = torch.cat((state,action),1)
        # Forward propogation on critic model 1
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward propogation on critic model 2
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self,state,action):
        """
        Forward propogation only on Critic Model 1
        :param state:
        :param action:
        :return:
        """
        xu = torch.cat((state,action),1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1



