#Implementation
## Problem Statement:
In Reacher app unity environment the single agent must get an average score of +30 over 100 consecutive episodes.
##Algorithm:
Deep Deterministic Policy Gradient (DDPG) is a reinforcement learning technique that combines both Q-learning and Policy gradients. DDPG being an actor-critic technique consists of two models: Actor and Critic. The actor is a policy network that takes the state as input and outputs the exact action (continuous), instead of a probability distribution over actions. The critic is a Q-value network that takes in state and action as input and outputs the Q-value. DDPG is an “off”-policy method. DDPG is used in the continuous action setting and the “deterministic” in DDPG refers to the fact that the actor computes the action directly instead of a probability distribution over actions.
DDPG is used in a continuous action setting and is an improvement over the vanilla actor-critic.<br>

![Alt text](images/ddpg.png?raw=true "DDPG Algorithm")
#Implementation
Following are the hyper-parameters used to solve the problem.

- BATCH_SIZE = 256
- BUFFER_SIZE = int(1e5)
- GAMMA = 0.9
- TAU = 1e-3
- LR_ACTOR = 1e-3
- LR_CRITIC = 1e-3
- POLICY_FREQ = 2
- POLICY_NOISE = 0.2
- NOISE_CLIP =0.5
- MAX_TIMESTEPS=10000

## Neural Network Architecture
Two neural network models with one Actor , Critic for current and Actor, Critic for target models has been implemented with the following architecture

###Actor Model:

  - Total Layers: 3 with final Activation function with Tanh<br>
  - First Layer with Input shape 33, output dimension 400 with Relu Activation
  - Second Layer with Input shape 400, output dimension 300 with Relu Activation
  - Third Layer with Input shape 300 and output dimension 4 with Tanh Activation

###Critic Model:

  - Total Layers: 3 
  - First Layer with Input shape 33, output dimension 400 with Relu Activation
  - Second Layer with Input shape 400, output dimension 300 with Relu Activation
  - Third layer without any activation function.


### Training
The DDPG Agent took around 148 episodes to achieve the average score of 30.18 <br>

![Alt text](images/ddpg_scores.png?raw=true "Agent Score")
![Alt text](images/learning.png?raw=true "Agent Score")




