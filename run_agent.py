import os
from unityagents import UnityEnvironment
from reacher.ddpg_agent import Agent


env = UnityEnvironment('Reacher.app')

if __name__ == '__main__':

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]
    state = env_info.vector_observations[0]
    score = 0
    eps = 0.05
    agent = Agent(state_size,action_size,seed=0)
    file_name = "%s_%s_%s" % ("DDPG", brain_name, str(0))
    agent.load(file_name,'./pytorch_models')
    while True:
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break

    print("Score: {}".format(score))