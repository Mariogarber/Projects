from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from time import time
import keras
import numpy as np
import random
import matplotlib.pyplot as plt
from class_DQN import ReplayMemory, DQN
# import zmqRemoteApi

GAMMA = 0.99
MEMORY_SIZE = 50000
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.95
MAX_NUMBER_OF_EPISODES_FOR_TRAINING = 600
NUMBER_OF_EPISODES_FOR_TESTING = 30
GOAL_SCORE = 200
NUMBER_OF_EPISODES_FOR_TESTING_GOAL_SCORE = 30
LEARN_BATCH = 30

class Coppelia():

    def __init__(self):
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')
        
    def start_simulation(self):
        self.default_idle_fps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.startSimulation()

    def stop_simulation(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.default_idle_fps)

    def is_running(self):
        return self.sim.getSimulationState() != self.sim.simulation_stopped
    
class P3DX():
    num_sonar = 16
    sonar_max = 1.0

    def __init__(self, sim, robot_id) -> None:
        self.sim = sim
        self.left_motor = self.sim.getObject(f'/{robot_id}/leftMotor')
        self.right_motor = self.sim.getObject(f'/{robot_id}/rightMotor')
        self.sonar = [self.sim.getObject(f'/{robot_id}/visible/ultrasonicSensor[{idx}]') for idx in range(self.num_sonar)]
        self.learning_network = DQN(9, 5)
        self.state = state()
        self.readings = self.get_sonar()
        
    def get_sonar(self):
        readings = []
        for i in range(self.num_sonar):
            res, dist, _, _, _ = self.sim.readProximitySensor(self.sonar[i])
            readings.append(dist if res == 1 else self.sonar_max)
        return readings

    def set_speed(self, exploration_rate):
        action = self.learning_network.select(self.status.list_state(), exploration_rate)
        if action == 0:
            ispeed, rspeed = self.turn_left(0)
        if action == 1:
            ispeed, rspeed = self.turn_left(0.5)
        if action == 2:
            ispeed, rspeed = self.move_forward()
        if action == 3:
            ispeed, rspeed = self.turn_right(0.5)
        if action == 4:
            ispeed, rspeed = self.turn_right(0)
        self.sim.setJointTargetVelocity(self.left_motor, ispeed)
        self.sim.setJointTargetVelocity(self.right_motor, rspeed)
        return action

    def move_forward():
        ispeed, rspeed = 1, 1
        return ispeed, rspeed

    def turn_right(grade):
        ispeed, rspeed = 1, grade
        return ispeed, rspeed

    def turn_left(grade):
        ispeed, rspeed = grade, 1
        return ispeed, rspeed

    def zigzag(grade):
        ispeed, rspeed = grade, 1
        return (ispeed / max(ispeed, rspeed)), (rspeed / max(ispeed, rspeed))

    def get_metrics(self, readings):
        metrics = {}
        metrics["forward_metrics"] = readings[3] * readings[4]
        metrics["left_metrics"] = readings[1] * readings[2]
        metrics["right_metrics"] = readings[5] * readings[6]
        metrics["right_wall_metrics"] = readings[7]
        metrics["left_wall_metrics"] = readings[0]
        metrics["readings"] = readings
        return metrics

    def set_state(self):
        readings = self.get_sonar()
        metrics = self.get_metrics(readings)
        self.state.right_metric = metrics["right_metrics"]
        self.state.left_metric = metrics["left_metrics"]
        self.state.forward_metric = metrics["forward_metrics"]
        self.state.right_wall_metric = metrics["right_wall_metrics"]
        self.state.left_wall_metric = metrics["left_wall_metrics"]
        if metrics["fordward_metrics"] < 0.5:
            self.state.wandering = 0
            self.state.front_wall = 1
        if metrics["left_metrics"] < 0.5:
            self.state.wandering = 0
            self.state.left_wall = 1
        if metrics["right_metrics"] < 0.5:
            self.state.wandering = 0
            self.state.right_wall = 1
        else:
            self.state.wandering = 1
            self.state.right_wall = 0
            self.state.left_wall = 0
            self.state.front_wall = 0
        return self.state

class state():
    def __init__(self):
        self.right_wall_metric = 0
        self.left_wall_metric = 0
        self.right_metric = 0
        self.left_metric = 0
        self.forward_metric = 0
        self.wandering = 0
        self.right_wall = 0
        self.left_wall = 0
        self.front_wall = 0
    
    def list_state(self):
        return [self.right_wall_metric, self.left_wall_metric, self.right_metric, self.left_metric, self.forward_metric, self.wandering, self.right_wall, self.left_wall, self.front_wall]


class environment():
    def __init__(self, robot_id, sim):
        self.actions = [0, 1, 2, 3, 4] # 0: left, 1: fordward-left, 2:fordward , 3: fordward-right, 4: right
        self.number_of_observations = 9 # fordward-metrics, left-metrics, right-metrics, right-wall-metrics, left-wall-metrics, wandering, front-wall, right-wall, left-wall
        self.number_of_actions = 5
        self.robot = P3DX(sim, robot_id)

    def reward(self):
        reward = 0
        state = self.robot.state
        if state.wandering:
            reward += -100
        else:
            if state.front_wall:
                reward += -10 / state.forward_metric
            if self.robot.state.right_wall:
                if state.right_metric < 0.1:
                    reward += -5 / state.right_metric
                reward += 5 / (state.right_wall_metric - 0.1)
            if self.robot.state.left_wall:
                if state.left_metric < 0.1:
                    reward += -5 / state.left_metric
                reward += 5 / (state.left_wall_metric - 0.1)
        return reward
    