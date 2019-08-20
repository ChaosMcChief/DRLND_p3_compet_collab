# DRLND_p3_compet_collab

This repository is a possible solution to the third Project of the Deep reinforcement learning Nanodegree - the collaboration-competition task.

## Intro
The environment in this project is derived from the Unity tennis environment and the solution is written for Pytorch. A quick description to the environment as well as to the state- and action-space can be found on the [Udacity Deep reinforcement learning github repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet). The solution for this project uses the code from the second project as boilerplate code and reuses elements such as the replay memory and other helperfunctions.

## Getting Started
Besides some Python standard packages, this project utilises [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [NumPy](http://www.numpy.org/) and [PyTorch](https://pytorch.org/)

To get things running, the tennis-app is needed also. An detailed guide for setting up the environment can be found in the [Project Github repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)

## What's in here?
This solution mainly uses the DDPG-approach as described in this [paper](https://arxiv.org/pdf/1509.02971.pdf). On top of that, it uses the distributional approach and n-step-bootstrapping from the [D4PG-Algorithm](https://arxiv.org/pdf/1804.08617.pdf). The jupyter notebook sets up the environment, trains two agents and lets you watch the trained agent.

## Instructions
This repo is tested in a Windows 64bit OS. If you use any different operating system, you have to set up the environment accordingly as describe in the already mentioned [udacity repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) under "Getting Started". And don't forget to change the path to the environment folder in the Notebook.

The module `models.py` contains the definitions of the Pytorch-models and `d4pg_agent.py` contains the definition of the agent as well as the experience memory.
In the jupyter-notebook `Tennis.ipynb` lies everything you need to start the environment and train the agent.

## ToDos
This solution doesn't contain useful improvements e.g. prioritized experience replay (PER). I experimented a bit with PER, but it seems that in this particular case, there isn't a great gain. One thing I'd like to improve is the distributed experience gathering, so that the two agents contribute to a shared memory and learn from it. Also, it could be helpful to evaluate the individual performance of the two agents from time to time and copy the model-parameters of the winner-agent to the loser-agent. This could improve the speed of training.
