# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/14W5I5mGS3cJjR5UuWp3k5_wSXl04lM7Q

# Game with Poximal Policy Optimization

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
import math
import argparse

import sys
sys.path.append('../../')
from algos.agents import PPOAgent
from algos.models import ActorCnn, CriticCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame
"""
Initialize the environment in the code cell below.

"""
parser = argparse.ArgumentParser(description='Game Test Script File.')
parser.add_argument('--game_title', type=str, default='MortalKombatII-Genesis',
                    help='Game title as our environment (default: MortalKombatII-Genesis)')
parser.add_argument('--version', default=1, required=True,
                    help='version the integer (default: 1) - checkpoint saved at timestep t')
parser.add_argument('--mode', default=0,
                    help='mode the integer (default: 1) - training mode 1 else 0 (test mode)')
parser.add_argument('--video_path', default='videos',
                    help='Video records path (default: videos)')
parser.add_argument('--ckpt_path', default='checkpoints/mortal_kombat/ppo_actor_',
                    help='Checkpoints path for trained models (default: checkpoints/mortal_kombat/ppo_actor_)')

args = parser.parse_args()
print("Version: ", args.version)

# https://www.ign.com/wikis/mortal-kombat-2/Characters_and_Move_List
env = retro.make(game=args.game_title, record=args.video_path)
acts = np.array(np.identity(env.action_space.n), dtype=int)
possible_actions = {}
for index, a in enumerate(acts):
    possible_actions[index] = a
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

Actor_PATH = '../../cgames/06_sonic2/checkpoints/Street_ppo_actor_'+str(args.version)+'.pth'
Critic_PATH = '../../cgames/06_sonic2/checkpoints/Street_ppo_critics_'+str(args.version)+'.pth'

## check if the checkpoint exists
file_exists = os.path.exists(Actor_PATH)
if file_exists:
    print(f'The file {Actor_PATH} exists, will be loaded into the checkpoint.')
    agent = PPOAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn, CriticCnn, Actor_PATH, Critic_PATH, mode=args.mode)
    # watch an untrained agent
    scores = []
    state = stack_frames(None, env.reset(), True) 
    for j in range(30000):
        env.render(close=False)
        action, _, _ = agent.act(state)
        next_state, reward, done, _ = env.step(possible_actions[action])
        state = stack_frames(state, next_state, False)
        scores.append(reward)              # save most recent score
        if done:
            env.reset()
    env.render(close=True)
else:
    print(f'The file {Actor_PATH} does not exist')
