# Commented out IPython magic to ensure Python compatibility.
import time
import retro
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math
import pandas as pd

# %matplotlib inline

import sys
sys.path.append('../../')
from algos.agents import PPOAgent
from algos.models import ActorCnn, CriticCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

from mortal_kombat import MortalKombatIIDiscretizer

"""## Create our environment

Initialize the environment in the code cell below.

"""
# https://www.ign.com/wikis/mortal-kombat-2/Characters_and_Move_List
env = retro.make(game='MortalKombatII-Genesis')
env = MortalKombatIIDiscretizer(env)

env.seed(0)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

"""## Viewing our Enviroment"""

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)

"""### Execute the code cell below to play Pong with a random policy."""

def random_play():
    score = 0
    env.reset()
    for i in range(20000):
        env.render()
        action =  env.action_space.sample() #possible_actions[np.random.randint(len(possible_actions))]
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            print("Your Score at end of game is: ", score)
            break
    env.reset()
    env.render(close=True)
random_play()



"""## Stacking Frame"""

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (1, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

"""## Step 6: Creating our Agent"""

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
SEED = 0
GAMMA = 0.99           # discount factor
ALPHA= 0.0001          # Actor learning rate
BETA = 0.0001          # Critic learning rate
TAU = 0.95
BATCH_SIZE = 32
PPO_EPOCH = 5
CLIP_PARAM = 0.2
UPDATE_EVERY = 100     # how often to update the network 
mode = 1

agent = PPOAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn, CriticCnn, mode)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

"""## Step 9: Train the Agent with Actor Critic"""

def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    for i_episode in range(start_epoch + 1, n_episodes+1):
        state = stack_frames(None, env.reset(), True)
        score = 0

        # Punish the agent for not moving forward
        prev_state = {}
        steps_stuck = 0
        timestamp = 0
        while timestamp < 10000:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, info = env.step(action)
            score += reward

            timestamp += 1
            # Punish the agent for standing still for too long.
            if (prev_state == info):
                steps_stuck += 1
            else:
                steps_stuck = 0
            prev_state = info
    
            if (steps_stuck > 20):
                reward -= 1

            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, value, log_prob, reward, done, next_state)
            if done:
                break
            else:
                state = next_state

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        pd.DataFrame({"score":scores}).to_csv("./runs/mortal_k_ppo_scores_5k_action_wrapper.csv")
    return scores

scores = train(5000)


"""## Watch a Smart Agent!"""

env.viewer = None
# watch an untrained agent
state = stack_frames(None, env.reset(), True) 
for j in range(10000):
    env.render(close=False)
    action, _, _ = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = stack_frames(state, next_state, False)
    if done:
        env.reset()
env.render(close=True)