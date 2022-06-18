# -*- coding: utf-8 -*-
"""sonic2_ppo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14W5I5mGS3cJjR5UuWp3k5_wSXl04lM7Q

# Sonic The Hedgehog 2 with Poximal Policy Optimization

## Step 1: Import the libraries
"""

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

# %matplotlib inline

import sys
sys.path.append('../../')
from algos.agents import PPOAgent
from algos.models import ActorCnn, CriticCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

"""## Step 2: Create our environment

Initialize the environment in the code cell below.

"""
# https://www.ign.com/wikis/mortal-kombat-2/Characters_and_Move_List
env = retro.make(game='MortalKombatII-Genesis', record='./lvideo')
# env = retro.make(game='SonicTheHedgehog2-Genesis', state='EmeraldHillZone.Act1', scenario='contest')
# print("Actions: ", env.action_space)
acts = np.array(np.identity(env.action_space.n), dtype=int)
possible_actions = {}
for index, a in enumerate(acts):
    possible_actions[index] = a
# print(possible_actions)
env.seed(0)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (1, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

"""## Step 6: Creating our Agent"""

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = len(possible_actions)
SEED = 0
GAMMA = 0.99           # discount factor
ALPHA= 0.0001          # Actor learning rate
BETA = 0.0001          # Critic learning rate
TAU = 0.95
BATCH_SIZE = 32
PPO_EPOCH = 5
CLIP_PARAM = 0.2
UPDATE_EVERY = 1000     # how often to update the network 

mode = 0
agent = PPOAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn, CriticCnn, mode)

scores = []

"""## Step 10: Watch a Smart Agent!"""

print("Watch a Smart Agent!")
env.viewer = None
# watch an untrained agent
state = stack_frames(None, env.reset(), True) 
for j in range(30000):
    env.render(close=False)
    action, _, _ = agent.act(state)
    next_state, reward, done, _ = env.step(possible_actions[action])
    state = stack_frames(state, next_state, False)
    scores.append(reward)              # save most recent score
    if done:
        env.reset()
        # break 
env.render(close=True)
