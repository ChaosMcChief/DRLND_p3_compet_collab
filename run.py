import os
from collections import deque
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from tqdm import tqdm

import Config
from d4pg_agent_1p import Agent

def setup_env():
    env = UnityEnvironment(file_name='tennis_windows_x86_64/Tennis.exe')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    return env, brain, brain_name


def prepop_memory(agent, agent_2, env, action_size):
    
    print("Prepopulating Memory...")
    pretrain_length = agent.memory.tree.capacity 
    
    actions = 2*np.random.rand(pretrain_length, 2, action_size) - 1
    
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations

    for i in tqdm(range(actions.shape[0])):
               
        # Random action
        action = actions[i]
        
        # Take the action
        env_info = env.step(action)[brain_name] 

        # Get next_state, reward and done
        next_state = env_info.vector_observations   # get the next state
        reward = env_info.rewards                   # get the reward
        done = env_info.local_done                  # see if episode has finished

        # Store the experience in the memory
        # Save experience in replay memory

        agent.memory.add(state, action, reward, next_state, done)

        #agent.memory.add(state, action, reward, next_state, done)
               
        # Reset env if done
        if done:
            env_info = env.reset(train_mode=True)[brain_name] # reset the environment
            state = env_info.vector_observations
        else:
            state = next_state
    
    return agent

def train_agent(n_episodes=2000, model_suff=""):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    
    actions = []
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations
        score = np.zeros(agent.num_agents)
#        score_2 = np.zeros(agent_2.num_agents)
        # Decay the action-noise and reset randomer
        #agent.decay_epsilon()
        agent.reset()
#        agent_2.reset()

        while True:
            
            action = agent.act(state)
#            action_2 = agent_2.act(state[1])
            
            actions.append(action)
#            actions_2.append(action_2)
            
            action = np.clip(action, -1.0, 1.0)
#            action_2 = np.clip(action_2, -1.0, 1.0)
            
#            action = [action_1, action_2]
            env_info = env.step(action)[brain_name]         # send the action to the environment
            next_state = env_info.vector_observations       # get the next state
            reward = env_info.rewards                       # get the reward
            done = env_info.local_done                      # see if episode has finished
            
            agent.step(state, action, reward, next_state, done)
#            agent_2.step(state[1], action_2, reward[1], next_state[1], done[1])
            state = next_state
            score += np.array(reward)
            if np.any(done): 
                break 

        scores_window.append(np.max(score))       # save most recent score
        scores.append(np.max(score))              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}\tLast Score: {:.2f}'.format(i_episode, np.mean(scores_window), scores[-1]), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.critic.state_dict(), f"Results/checkpoint_critic_{model_suff}.pth")
            torch.save(agent.actor.state_dict(), f"Results/checkpoint_actor_{model_suff}.pth")
            pickle.dump(scores, open(f"Results/Scores_{model_suff}.pkl", 'wb'))
            pickle.dump(agent.losses_actor, open(f"Results/losses_actor_{model_suff}.pkl", 'wb'))
            pickle.dump(agent.losses_critic, open(f"Results/losses_critic_{model_suff}.pkl", 'wb'))
#            pickle.dump(agent.max_gradients_actor, open(f"max_gradients_actor2.pkl", 'wb'))
#            pickle.dump(agent.max_gradients_critic, open(f"max_gradients_critic2.pkl", 'wb'))
        if (np.mean(scores_window)>=.5) or (i_episode>=n_episodes+1):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.critic.state_dict(), f"Results/checkpoint_critic_final_{model_suff}.pth")
            torch.save(agent.actor.state_dict(), f"Results/checkpoint_actor_final_{model_suff}.pth")
            pickle.dump(scores, open(f"Results/Scores_{model_suff}.pkl", 'wb'))
            break
    return scores

def play(agent, env, n_episodes_to_watch=1):
    scores = []
    for i_episode in range(1, n_episodes_to_watch+1):    # Loop through five episodes
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations
        score = np.zeros(agent.num_agents)
        agent.is_training = False
        
        while True:
            action = agent.act(state)
            action = np.clip(action, -1.0, 1.0)
            
            env_info = env.step(action)[brain_name]        # send the action to the environment
            state = env_info.vector_observations             # get the next state
            reward = env_info.rewards                        # get the reward
            done = env_info.local_done                       # see if episode has finished
    
            score += np.array(reward)                                     # update the score
    
            if np.any(done):
                break 
        scores.append(np.max(score))              # save most recent score
    print(f"Avg. score: {np.mean(scores):.3}")

train=True

# Setup the environment
env, brain, brain_name = setup_env()
env_info = env.reset(train_mode=True)[brain_name]

# Define the config and kind of Replaybuffer
config = Config.config()
per = False

# number of actions
action_size = brain.vector_action_space_size

# dimension of the state-space 
num_agents = env_info.vector_observations.shape[0]
state_size = env_info.vector_observations.shape[1]

# Create the Agent and let it train
agent = Agent(config=config, 
                state_size=state_size, 
                action_size=action_size, 
                num_agents=num_agents, 
                seed=1234, per=per)
#agent_2 = Agent(config=config, 
#                state_size=state_size, 
#                action_size=action_size, 
#                num_agents=1, 
#                seed=0, per=per)

#agent = ddpg_agent.Agent(config=config, 
#                state_size=state_size, 
#                action_size=action_size, 
#                num_agents=2, 
#                seed=1234, per=False)
#agent_2 = ddpg_agent.Agent(config=config, 
#                state_size=state_size, 
#                action_size=action_size, 
#                num_agents=1, 
#                seed=0, per=False)

#prepop_memory(agent, agent_2, env, action_size)

# agent = Agents(state_size=state_size, 
#                 action_size=action_size, 
#                 num_agents=num_agents, 
#                 random_seed=0)


scores = train_agent(n_episodes=10000, model_suff='d4pg_1p_nnloss')
##pickle.dump(scores, open('Scores_benchmark.pkl', 'wb'))
#env.close()


#os.system("shutdown /s /t 1")
# fig = plt.figure()

# ax1 = plt.plot(range(len(scores_w_per)), scores_w_per)
# ax2 = plt.plot(range(len(scores_wo_per)), scores_wo_per)

# fig.legend(ax1, ax2, 'Mit PER', 'Ohne PER')
