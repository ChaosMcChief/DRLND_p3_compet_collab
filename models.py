import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).normal_(0.0, v)

# ----------------------------------------------------
# actor model, MLP
# ----------------------------------------------------
# 2 hidden layers, 400 units per layer, tanh output to bound outputs between -1 and 1
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.tanh=nn.Tanh()
        self.init_weights()
    
    def init_weights(self, init_w=10e-3):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

    def forward(self, state):
        x=F.leaky_relu(self.fc1(state))
        x=F.leaky_relu(self.fc2(x))
        x=self.bn2(x)
        x=self.tanh(x)

        return x


# ----------------------------------------------------
# critic model, MLP
# ----------------------------------------------------
# 2 hidden layers, 300 units per layer, outputs rewards therefore unbounded
# Action not to be included until 2nd layer of critic (from paper). Make sure 
# to formulate your critic.forward() accordingly

class Critic(nn.Module):
    def __init__(self, state_size, action_size, n_atoms):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128 + action_size, 128)
        self.fc3 = nn.Linear(128, n_atoms)
        self.init_weights()

    def init_weights(self, init_w=10e-3):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.normal_(0, 3e-4)


    def forward(self, state, action):
        xs=F.leaky_relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x=F.leaky_relu(self.fc2(x))
        x=self.fc3(x)
        out = F.softmax(x, dim=1)   # Probability distribution over n_atom q_values

        return out

