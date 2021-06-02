import numpy as np
import torch
import random
import torch.nn.functional as F
import torch.optim as optimum

from reacher.models import Actor,Critic
from reacher.noise import OUNoise, GuassianNoise
from reacher.storage import ReplayBuffer


BATCH_SIZE = 256
BUFFER_SIZE = int(1e5)
GAMMA = 0.9
TAU = 1e-3
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
POLICY_FREQ = 2
POLICY_NOISE = 0.2
NOISE_CLIP =0.5
MAX_TIMESTEPS=10000


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):

    def __init__(self,state_dim,action_dim,seed):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(seed)

        # Actor Network (Local / Target networks)
        self.actor_local = Actor(state_dim,action_dim,seed).to(device)
        self.actor_target = Actor(state_dim,action_dim,seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optimum.Adam(self.actor_local.parameters(),lr=LR_ACTOR)

        # Twin Critic Network(s) (Local / Target networks)
        self.critic_local = Critic(state_dim,action_dim,seed).to(device)
        self.critic_target = Critic(state_dim,action_dim,seed).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optimum.Adam(self.critic_local.parameters(),lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_dim,seed)


        # Replay Memory
        self.memory = ReplayBuffer(action_dim,BUFFER_SIZE,BATCH_SIZE,seed)

    def step(self,state,action,reward,next_state,done,iteration=0):
        """
        Save experience in replay buffer and use random sample from buffer to learn.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        self.memory.add(state,action,reward,next_state,done)
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences,GAMMA,iteration)

    def act(self,state,total_timesteps=0):
        """
        Returns actions for given state as per current policy
        :param state:
        :return:
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample()
        return np.clip(action,-1,1)

    # def reset(self):
    #     """
    #     Reset the noise
    #     :return:
    #     """
    #     self.noise.reset()

    def learn(self,experiences,gamma,iteration):
        """
        To update policy and value parameters using given batch of experience tuples
        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        :param experiences:
        :param gamma:
        :return:
        """
        states, actions, rewards, next_states, dones = experiences
        #------------- Update Critic ---------------#
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target.Q1(next_states,actions_next)
        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local.Q1(states,actions)
        critic_loss = F.mse_loss(q_expected,q_targets)
        #Miniize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #---------------- update actor ------------#
        #Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local.Q1(states,actions_pred).mean()
        # Minimze the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #-------------- Update target networks --------#
        self.soft_update(self.critic_local,self.critic_target,TAU)
        self.soft_update(self.actor_local,self.actor_target,TAU)
        #self.soft_udpate(self.actor_local,self.actor_target,TAU)



    def soft_update(self,local_model,target_model,tau):
        """
        Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model:
        :param target_model:
        :param tau:
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

        # To load pre-trained model

    def load(self, filename, directory):
        self.actor_local.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic_local.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


