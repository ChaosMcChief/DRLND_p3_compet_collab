import numpy as np
import random
import copy
from collections import namedtuple, deque

import models

import torch
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, config, state_size, action_size, num_agents, seed, per=True):
        """Initialize an Agent object.
        
        Params
        ======
            config (config): instance of a config-class, which stores all the hyperparameters
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.config = config
        self.epsilon = self.config.EPSILON_START

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = seed

        # Initialize bins
        self.v_min = 0
        self.v_max = 5
        self.n_atoms = 51
        self.delta = (self.v_max-self.v_min)/float(self.n_atoms-1)
        self.bin_centers = torch.from_numpy(np.array([self.v_min+i*self.delta for i in range(self.n_atoms)]).reshape(-1,1)).to(self.config.device)

        # Initialize the Actor and Critic Networks
        self.actor_local = models.Actor(state_size, action_size).to(self.config.device)
        self.actor_target = models.Actor(state_size, action_size).to(self.config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), self.config.LR_actor)

        self.critic_local = models.Critic(state_size, action_size, self.n_atoms).to(self.config.device)
        self.critic_target = models.Critic(state_size, action_size, self.n_atoms).to(self.config.device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), self.config.LR_critic, weight_decay=self.config.weight_decay)     
        
        # Initialize the random-noise-process for action-noise
        self.is_training = True
        self.noise = OUNoise((self.num_agents, self.action_size), self.seed)

        # Hard update the target networks to have the same parameters as the local networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(param.data)

        # Initialize the replay-buffer according to `per`
        self.memory = ReplayBuffer(self.config.BUFFER_SIZE, self.config.BATCH_SIZE, seed, self.config.device, self.config.N_BOOTSTRAP)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done):
        """ Processes one experience-tuple (i.e store it in the replay-buffer
        and take a learning step, if it is time to do that.
        """
        
        # Save experience in replay memory
        if self.num_agents>1:
            for i in range(self.num_agents):
                self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])
        else:
            self.memory.add(state, action, reward, next_state, done)
        
        self.t_step += 1

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.config.BATCH_SIZE:
            if self.t_step % 4 == 0:
                # for i in range(0,10):
                self.learn(self.memory.sample(), self.config.GAMMA)
            
    
    def act(self, states):
        """Returns actions for given state as per current policy.
        Also adds random action-noise to the action-values while training.

        Params
        ======
            states (array_like): current state  
        """       
        # Convert the state to a torch-tensor
        states = torch.from_numpy(states).float().to(self.config.device)

        # Compute the action-values
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states)
        self.actor_local.train()
        action = action.cpu().numpy()
        
        # Add noise while training
        if self.is_training:
            action += self.epsilon * self.noise.noise().squeeze()
        action = np.clip(action, -1.0, 1.0)
        
        return action

    def learn(self, mini_batch, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            mini_batch (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = mini_batch
        # states = states.view(self.config.N_BOOTSTRAP, self.config.BATCH_SIZE, -1)    
        # actions = actions.view(self.config.N_BOOTSTRAP, self.config.BATCH_SIZE, -1)    
        # rewards = rewards.view(self.config.N_BOOTSTRAP, self.config.BATCH_SIZE, -1)    
        # next_states = next_states.view(self.config.N_BOOTSTRAP, self.config.BATCH_SIZE, -1)    
        # dones = dones.view(self.config.N_BOOTSTRAP, self.config.BATCH_SIZE, -1)    

        
        # Get the noised actions from the local network for the next states
        actions_target_next = self.actor_target(next_states[:,-1])
        
        # Evaluate the computed actions with the critic-target-network
        Q_targets_next = self.critic_target(next_states[:,-1], actions_target_next).detach()

        # Compute the estimated q-value-distribution using the local network
        q_dist = self.critic_local(states[:,0], actions[:,0]).type(torch.float64)
        
        rewards = rewards.reshape(self.config.BATCH_SIZE,self.config.N_BOOTSTRAP).cpu().numpy()

        # Calculate the sum of the discounted rewards because of N-step bootstrapping
        gammas = np.array([self.config.GAMMA**i for i in range(self.config.N_BOOTSTRAP+1)]).reshape((1,-1))
        gamma = gammas[0,-1]
        rewards = np.sum(gammas[:,:-1]*rewards, axis=1)
        
        # Scale and move the estimated distribution with gamma and the rewards 
        z_p = self.bin_centers + torch.tensor(gamma + rewards, device=self.config.device)
        z_p.transpose_(1,0)
        z_p[dones[:,-1],:] = 0.0
        
        # Project the scaled distribution onto the original supports
        reprojected_dist = self.l2_project(z_p, Q_targets_next, self.bin_centers.squeeze())

        # Compute the loss
        loss_critic = -(reprojected_dist*torch.log(q_dist+1e-10)).sum(dim=1).mean()      

        # Compute critic gradient and perform the optimizer step
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        

        # Compute the actor gradient and perform the optimizer step
        actions_local = self.actor_local(states[:,0])
        loss_actor = (self.critic_local(states[:,0], actions_local)).type(torch.float64)
        loss_actor = -(loss_actor.matmul(self.bin_centers).mean())
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # update the target networks
        self.soft_update(self.actor_local, self.actor_target, self.config.TAU)                     
        self.soft_update(self.critic_local, self.critic_target, self.config.TAU) 


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    
    def l2_project(self, z_p, p, z_q):
        """ I found this function on https://github.com/msinto93/D4PG/blob/master/utils/l2_projection.py
        for tensorflow and then translated into pytorch.

        Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
        The supports z_p and z_q are specified as tensors of distinct atoms (given
        in ascending order).
        Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
        support z_q, in particular Kq need not be equal to Kp.
        Args:
          z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
          p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
          z_q: Tensor holding support to project onto, shape `[Kq]`.
        Returns:
          Projection of (z_p, p) onto support z_q under Cramer distance.
        """
        # Broadcasting of tensors is used extensively in the code below. To avoid
        # accidental broadcasting along unintended dimensions, tensors are defensively
        # reshaped to have equal number of dimensions (3) throughout and intended
        # shapes are indicated alongside tensor definitions. To reduce verbosity,
        # extra dimensions of size 1 are inserted by indexing with `None` instead of
        # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
        # `[k, l]' to one of shape `[k, 1, l]`).
        
        # Extract vmin and vmax and construct helper tensors from z_q
        vmin, vmax = z_q[0], z_q[-1]
        d_pos = torch.cat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
        d_neg = torch.cat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
        # Clip z_p to be in new support range (vmin, vmax).
        z_p = torch.clamp(z_p, vmin, vmax)[:, None, :]  # B x 1 x Kp
        
        # Get the distance between atom values in support.
        d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
        d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
        z_q = z_q[None, :, None]  # 1 x Kq x 1
        
        # Ensure that we do not divide by zero, in case of atoms of identical value.
        d_neg = torch.where(d_neg > 0, 1./d_neg, torch.zeros_like(d_neg))  # 1 x Kq x 1
        d_pos = torch.where(d_pos > 0, 1./d_pos, torch.zeros_like(d_pos))  # 1 x Kq x 1
        
        delta_qp = z_p - z_q   # clip(z_p)[j] - z_q[i]. B x Kq x Kp
        
        d_sign = torch.zeros(delta_qp.shape, device=self.config.device, dtype=torch.float64)
        d_sign[delta_qp>=0.] = 1
        
        # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
        # Shape  B x Kq x Kp.
        delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
        p = p[:, None, :].type(torch.float64)  # B x 1 x Kp.
        return torch.sum(torch.clamp(1. - delta_hat, 0., 1.) * p, 2)        


class ReplayBuffer:
    """Replay buffer which supports N-step-bootstrapping to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device, N_bootstrap=1):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): device on which the tensors are stored
            N_bootstrap (int): length of the N-step-bootstrapping
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
        self.N_bootstrap = N_bootstrap

        self.init_bootstrap_deq()


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        self.deq_state.append(state)
        self.deq_action.append(action)
        self.deq_reward.append(reward)
        self.deq_next_state.append(next_state)
        self.deq_done.append(done)
        
        if len(self.deq_state)==self.N_bootstrap:       
            e = self.experience(np.expand_dims(np.array(self.deq_state), 0),
                                np.expand_dims(np.array(self.deq_action), 0), 
                                np.expand_dims(np.array(self.deq_reward), 0),
                                np.expand_dims(np.array(self.deq_next_state), 0),
                                np.expand_dims(np.array(self.deq_done),0))
            self.memory.append(e)

        if np.any(done):
            self.init_bootstrap_deq()

    def init_bootstrap_deq(self):
        # Initializes deques for N-Step bootstrapping
        self.deq_state = deque(maxlen=self.N_bootstrap)
        self.deq_action = deque(maxlen=self.N_bootstrap)
        self.deq_reward = deque(maxlen=self.N_bootstrap)
        self.deq_next_state = deque(maxlen=self.N_bootstrap)
        self.deq_done = deque(maxlen=self.N_bootstrap)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def noise(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state