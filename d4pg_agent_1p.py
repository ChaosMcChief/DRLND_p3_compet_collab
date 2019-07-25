import numpy as np
import random
import copy
from collections import namedtuple, deque

import models

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, config, state_size, action_size, num_agents, seed, per=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            per (bool): If True, uses prioritized experience replay
            duelling (bool): If True, uses duelling network architecture
        """

        self.config = config
        self.epsilon = self.config.EPSILON_START

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = seed
        
        self.max_gradients_actor = []
        self.max_gradients_critic = []
        self.losses_actor = []
        self.losses_critic = []


        # Initialize bins
        self.v_min = -1
        self.v_max = 1
        self.n_atoms = 51
        self.delta = (self.v_max-self.v_min)/float(self.n_atoms-1)
        self.bin_centers = torch.from_numpy(np.array([self.v_min+i*self.delta for i in range(self.n_atoms)]).reshape(-1,1)).to(self.config.device)

        # Initialize the Actor and Critic Networks
        self.actor = models.Actor(state_size, action_size).to(self.config.device)
        self.actor_target = models.Actor(state_size, action_size).to(self.config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.config.LR_actor)

        self.critic = models.Critic(state_size, action_size, self.n_atoms).to(self.config.device)
        self.critic_target = models.Critic(state_size, action_size, self.n_atoms).to(self.config.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.config.LR_critic, weight_decay=self.config.weight_decay)     
        
        # Initialize the random-noise-process for action-noise
        self.is_training = True
        self.randomer = OUNoise((self.num_agents, self.action_size), self.seed)

        # Hard update the target networks to have the same parameters as the local networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Initialize the replay-buffer according to `per`
        self.memory = ReplayBuffer(self.config.BUFFER_SIZE, self.config.BATCH_SIZE, seed, self.config.device, self.config.N_BOOTSTRAP)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def decay_epsilon(self):
        """ Perform one decay-step for epsilon"""
        self.epsilon -= self.config.EPSILON_DECAY

    def reset(self):
        self.randomer.reset()

    def step(self, state, action, reward, next_state, done):
        """ Processes one experience-tuple (i.e store it in the replay-buffer
        and take a learning step, if it is time to do that.
        """
        
        # Save experience in replay memory
        for i in range(self.num_agents):
            self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])
        self.t_step += 1

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.config.BATCH_SIZE:
#            if self.t_step % 20 == 0:
#                for i in range(0,10):
            self.learn(self.memory.sample(), self.config.GAMMA)
            
    
    def act(self, states):
        """Returns actions for given state as per current policy.
        Also adds random action-noise to the action-values while training.

        Params
        ======
            state (array_like): current state  
        """       
        # Convert the state to a torch-tensor
        states = torch.from_numpy(states).float().to(self.config.device)

        # Compute the action-values
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(states)
        self.actor.train()
        action = action.cpu().numpy()
        action += self.is_training * max(self.epsilon, self.config.EPSILON_MIN) * self.randomer.noise().squeeze()
#        action = np.clip(action, -1.0, 1.0)

        return action

    def learn(self, mini_batch, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
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
        Q_targets_next = self.critic_target(next_states[:,-1], actions_target_next)

        q_dist = self.critic(states[:,0], actions[:,0])
        reprojected_dist = self.reproject(Q_targets_next.cpu().data.numpy(), rewards, dones)

        reprojected_dist = torch.tensor(reprojected_dist, requires_grad=False).to(self.config.device)

        loss_critic = -(reprojected_dist*torch.log(q_dist+1e-010)).sum(dim=1).mean()
        
        
        
        self.losses_critic.append(loss_critic.detach().cpu().numpy())

        # critic gradient
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        
#        temp = []
#        for param in self.critic.parameters():
#            temp.append(torch.max(torch.abs(param.grad.reshape(-1,1))).numpy())        
#        self.max_gradients_critic.append(np.max(temp))
        
        
        
        self.critic_optimizer.step()

        # actor gradient
        actions_local = self.actor(states[:,0])
        loss_actor = (self.critic(states[:,0], actions_local)).type(torch.float64)
#        print(loss_actor.dtype)
#        print(torch.from_numpy(self.bin_centers).dtype)
        loss_actor = -loss_actor.matmul(self.bin_centers).mean()
        self.losses_actor.append(loss_actor.detach().cpu().numpy())
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        
#        temp = []
#        for param in self.actor.parameters():
#            temp.append(torch.max(torch.abs(param.grad.reshape(-1,1))).numpy())        
#        self.max_gradients_actor.append(np.max(temp))
#        
        
        
        self.actor_optimizer.step()

        # update the target networks
        self.soft_update(self.actor, self.actor_target, self.config.TAU)                     
        self.soft_update(self.critic, self.critic_target, self.config.TAU) 


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

#    def hat_function(self, z)
#        
#        if z<=
        
        
            
    


    def reproject(self, target_z_dist, rewards, terminates):
        
    #next_distr = next_distr_v.data.cpu().numpy()

        rewards = rewards.reshape(self.config.BATCH_SIZE,self.config.N_BOOTSTRAP).cpu().numpy()
        terminates = terminates.reshape(-1).cpu().numpy().astype(bool)
        #dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
        #batch_size = len(rewards)
        proj_distr = np.zeros((self.config.BATCH_SIZE, self.n_atoms), dtype=np.float32)

        # Calculated the sum of the discounted rewards because of N-step bootstrapping
        gammas = np.array([self.config.GAMMA**i for i in range(self.config.N_BOOTSTRAP+1)]).reshape((1,-1))
        gamma = gammas[0,-1]
        rewards = np.sum(gammas[:,:-1]*rewards, axis=1)

        #pdb.set_trace()

        for atom in range(self.n_atoms):
            tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards + (self.v_min + atom * self.delta) * gamma))
            b_j = (tz_j - self.v_min) / self.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = (u == l).astype(bool)
            proj_distr[eq_mask, l[eq_mask]] += target_z_dist[eq_mask, atom]
            ne_mask = (u != l).astype(bool)
            proj_distr[ne_mask, l[ne_mask]] += target_z_dist[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += target_z_dist[ne_mask, atom] * (b_j - l)[ne_mask]

#        if terminates.any():
#            proj_distr[terminates] = 0.0
#            tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards[terminates]))
#            b_j = (tz_j - self.v_min) / self.delta
#            l = np.floor(b_j).astype(np.int64)
#            u = np.ceil(b_j).astype(np.int64)
#            eq_mask = (u == l).astype(bool)
#            eq_dones = terminates.copy()
#            eq_dones[terminates] = eq_mask
#            if eq_dones.any():
#                proj_distr[eq_dones, l] = 1.0
#            ne_mask = (u != l).astype(bool)
#            ne_dones = terminates.copy()
#            ne_dones[terminates] = ne_mask.astype(bool)
#            if ne_dones.any():
#                proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
#                proj_distr[ne_dones, u] = (b_j - l)[ne_mask]

        return proj_distr

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device, N_bootstrap=1):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
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
        # Deques for N-Step bootstrapping
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
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
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