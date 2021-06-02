import torch
from reacher.noise import GuassianNoise
from reacher.models import Actor,Critic
from reacher.ddpg_agent import TD3Agent
from unityagents import UnityEnvironment
import time

# TODO remove the below declarations

env = UnityEnvironment("../Reacher.app")

def get_brain(env):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    return brain_name,brain

def reset(env):
    #env = UnityEnvironment("../Reacher.app")
    brain_name,_  = get_brain(env)
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations[0]
    rewards = env_info.rewards[0]
    dones = env_info.local_done[0]
    return states,rewards,dones

def step(env,action):
    brain_name, _ = get_brain(env)
    brain_info = env.step(action)[brain_name]
    states = brain_info.vector_observations[0]
    rewards = brain_info.rewards[0]
    dones = brain_info.local_done[0]
    return states, rewards, dones




def evaluate_policy(env,policy,eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        states, _, _  =  reset(env)
        done = False
        while not done:
            action = policy.act(states)
            next_state, reward, done = step(env,action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("--------------------------------------------")
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print("--------------------------------------------")



if __name__ == '__main__':
    env = UnityEnvironment("../Reacher.app")
    brain = get_brain(env)
    env_info, states, rewards, done = reset(env)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]
    #policy = Actor(state_size,action_size)
    agent = TD3Agent(state_size,action_size,0)
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    max_timesteps = 1e4
    t0 = time.time()
    while total_timesteps < max_timesteps:
        if done:
            if total_timesteps!=0:
                print("Total Timesteps: {} Episode Num: {} Reward {} ".format(total_timesteps, episode_num,
                                                                              episode_reward))
                #agent.



