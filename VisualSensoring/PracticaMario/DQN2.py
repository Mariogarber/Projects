'''
avoid.py

Sample client for the Pioneer P3DX mobile robot that implements a
kind of heuristic, rule-based controller for collision avoidance.

Copyright (C) 2023 Javier de Lope

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import matplotlib.pyplot as plt
import math
import random
import time
import gym
import cv2
from gym import spaces
import keras
from clases import Coppelia, P3DX, environment, State
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

GAMMA = 0.99
MEMORY_SIZE = 100000
LEARNING_RATE = 0.1
BATCH_SIZE = 64
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99
MAX_NUMBER_OF_EPISODES_FOR_TRAINING = 300
UPDATE_EVERY_T_STEPS = 4
NUMBER_OF_EPISODES_FOR_TESTING = 30
GOAL_SCORE = 200
NUMBER_OF_EPISODES_FOR_TESTING_GOAL_SCORE = 20
LEARN_BATCH = 4




episode = 0
start_time = time.perf_counter()
total_steps = 1
exploration_rate = EXPLORATION_MAX
goal_reached = False
coppelia = Coppelia()
robot = P3DX(coppelia.sim, 'PioneerP3DX')
env = environment('PioneerP3DX', sim = coppelia.sim)
robot.state = State()
sim = coppelia.sim
coppelia.start_simulation()
learn_batch = 0
reward = 0
while episode < MAX_NUMBER_OF_EPISODES_FOR_TRAINING and not (goal_reached):
    episode += 1
    score = 0
    state = robot.update_state()
    last_action = None
    end_episode = False
    steps = 0

    while coppelia.is_running() and not(end_episode):
        learn_batch +=1
        readings = robot.get_sonar()
        # Select an action for the current state
        action = robot.get_action(exploration_rate, test=False)
        reward += env.reward(robot, readings, action)
        # Execute the action on the environment
        robot.set_speed(action)
        state_next = robot.update_state()
        print(f"state: {state}, step: {steps}, action: {action}, reward: {reward}, next_state: {state_next}")


        # Store in memory the transition (s,a,r,s')
        robot.learning_network.remember(state, action, reward, state_next, False)

        score += reward

        # Learn using a batch of experience stored in memory
        if learn_batch == LEARN_BATCH:
            robot.learning_network.learn(True)
            learn_batch = 0
        else:
            robot.learning_network.learn(False)

        state = state_next
        last_action = action
        total_steps += 1
        steps += 1

    # Decrease exploration rate
    exploration_rate *= EXPLORATION_DECAY
    exploration_rate = max(EXPLORATION_MIN, exploration_rate)

coppelia.stop_simulation()






