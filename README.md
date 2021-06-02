# The Environment

![Alt text](images/env.png?raw=true "Unity ML-Agents Reacher Environment")
<p>
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, agent must get an average score of +30 over 100 consecutive episodes.
## Environment Info 
Number of Brains: 1 <br>
Number of Visual Observations (per agent): 0 <br>
Vector Observation space type: continuous <br>
Vector Observation space size (per agent): 33 <br>
Number of stacked Vector Observation: 1 <br>
Vector Action space type: continuous <br>
Vector Action space size (per agent): 4 <br>



# Installation Steps:
The project is built on conda 3 and can be created by exporting the environment.yml file. By executing the following command

conda env create -f environment.yml

## Setup the environment:
Following installations are available and can be downloaded from here <br>
   -  <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip">Mac OSX</a>
   -  <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip">Linux</a>
   -  <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip">Windows 64</a>

## To Train the agent
Run the following command to train and run the agent. <br>
python train_agent.py

## Run the agent
To Run the use the following command <br>
python run_agent.py

