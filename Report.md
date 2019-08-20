# Report
---

## Used algorithm
This implementation uses the Deep deterministic policy gradient as explained in this [paper](https://arxiv.org/pdf/1509.02971.pdf). It is an off-policy actor-critic method which uses two seperate deep neural networks to perform both tasks. The actor-network gives the actions as real numbers, therefore this method suits perfectly for continous action spaces. Moreover, the solution contains two key features of the [D4PG-Algorithm](https://arxiv.org/pdf/1804.08617.pdf), i.e. the distributional approach for the critic and n-step bootstrapping. So instead of estimating the q-value of the given state-action-pair, the critic estimates a distribution of a random variable for q.

## Code structure
To start the environment and train/watch an agent, you'll just need to run the jupyter notebook `Tennis.ipynb`. Because previously trained model parameters are also included in this repository, you can skip the training section and directly watch an trained agents interact with the environment. 

Firstly all the hyperparameters for the d4pg-agent are stored in an own simple class `config`, which lies in the [`Config.py`](https://github.com/ChaosMcChief/DRLND_p3_compet_collab/blob/master/Config.py). An instance of the `config`-class has to be passed to the agent while instantiating the agent.

The [`d4pg-agent.py`](https://github.com/ChaosMcChief/DRLND_p3_compet_collab/blob/master/d4pg_agent.py) contains the d4pg-like-agent alongside with the replay memory and an implementation of the Ornstein-Uhlenbeck random process to generate action-noise in order to explore the action-space. An explaination of this random process can be found on the [wikipedia-site](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)

The last python-module needed includes the neural networks for the actor and the critic in the [`models.py`](https://github.com/ChaosMcChief/DRLND_p3_compet_collab/blob/master/models.py). The neural networks are standard feed forward networks with two layers for the actor and three layers for the critic. In the critic the state is first passed through the first layer before being concatenated with the action values. The last layer of the critic includes a softmax-function, since the critic is estimating a random distribution of the q-value. The actor has an additional batchnorm-layer at the end, to stabilize learning.

## Hyperparameters
As mentioned above, the [`Config.py`](https://github.com/ChaosMcChief/DRLND_p3_compet_collab/blob/master/Config.py)-file stores all the relevant hyperparameters:

```
self.BUFFER_SIZE = int(1e5)  # replay buffer size
self.BATCH_SIZE = 512        # minibatch size
self.GAMMA = 0.99            # discount factor
self.TAU = 1e-3              # for soft update of target parameters
self.LR_actor = 1e-4         # learning rate for the actor 
self.LR_critic = 1e-4        # learning rate for the critic
self.EPSILON_START = 1       # Start-value of epsilon for action-noise
self.EPSILON_MIN = 0.01      # Min-value of epsilon
self.EPSILON_DECAY = 0       # Stepsize of the decay of epsilon

self.N_BOOTSTRAP = 1         # Value for bootstrapping

self.weight_decay=.0001      # Weight-decay for the critic network

# Use the GPU if one is available
self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Results
The results are shown in the plot below. Interestingly the shown and best results are achieved with a length of 1 for the n-step-bootstrapping. Also, the reason that the score isn't higher, is because agent_1 outperformed agent_2 and agent_2 loses fairly often and quickly. If you only let agents play with the learned parameters from agent_1, the games lasts way longer resulting in much higher scores.

![Scoreplot](https://github.com/ChaosMcChief/DRLND_p3_compet_collab/blob/master/Results/Scores.png)

## Ideas for future experiments
Because of the described behaviour above, it seems reasonable to implement a function, which evaluates the performance of the agents and keep only the parameters of the better agents for future training. This could improve the training speed. Also it would be nice to have the two agents contribute to a single replay memory.
