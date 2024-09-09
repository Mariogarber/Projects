# Description: Clases para el entorno de CoppeliaSim y el robot P3DX
import matplotlib.pyplot as plt
import random
import time
import cv2
from gym import spaces
import keras
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

GAMMA = 0.99
MEMORY_SIZE = 100000
LEARNING_RATE = 0.1
BATCH_SIZE = 0
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99
MAX_NUMBER_OF_EPISODES_FOR_TRAINING = 300
UPDATE_EVERY_T_STEPS = 4
NUMBER_OF_EPISODES_FOR_TESTING = 30
GOAL_SCORE = 200
NUMBER_OF_EPISODES_FOR_TESTING_GOAL_SCORE = 20
LEARN_BATCH = 4

class ReplayMemory:

    def __init__(self,number_of_observations):
        # Create replay memory
        self.states = np.zeros((MEMORY_SIZE, number_of_observations))
        self.states_next = np.zeros((MEMORY_SIZE, number_of_observations))
        self.actions = np.zeros(MEMORY_SIZE, dtype=np.int32)
        self.rewards = np.zeros(MEMORY_SIZE)
        self.terminal_states = np.zeros(MEMORY_SIZE, dtype=bool)
        self.current_size=0
        self.current_indx = 0

    def store_transition(self, state, action, reward, state_next, terminal_state):
        # Store a transition (s,a,r,s') in the replay memory
        i = self.current_indx
        self.states[i] = state
        self.states_next[i] = state_next
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminal_states[i] = terminal_state
        if self.current_size < MEMORY_SIZE - 1:
            self.current_size += 1
            self.current_indx += 1
        else:
            self.current_indx = (self.current_indx + 1) % MEMORY_SIZE

    def sample_memory(self, batch_size):
        # Generate a sample of transitions from the replay memory
        batch = np.random.choice(self.current_size, batch_size)
        states = self.states[batch]
        states_next = self.states_next[batch]
        rewards = self.rewards[batch]
        actions = self.actions[batch]
        terminal_states = self.terminal_states[batch]
        return states, actions, rewards, states_next, terminal_states
    

class Coppelia():

    def __init__(self):
        print('*** connecting to coppeliasim')
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')

    def start_simulation(self):
        # print('*** saving environment')
        self.default_idle_fps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.startSimulation()

    def stop_simulation(self):
        # print('*** stopping simulation')
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        # print('*** restoring environment')
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.default_idle_fps)
        print('*** done')

    def is_running(self):
        return self.sim.getSimulationState() != self.sim.simulation_stopped
    

class DQN:

    def __init__(self, number_of_observations, number_of_actions):
        # Initialize variables and create neural model
        self.number_of_actions = number_of_actions
        self.number_of_observations = number_of_observations
        self.scores = []
        self.memory = ReplayMemory(number_of_observations)

        # Neural model
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(50, input_shape=(number_of_observations,), activation="relu", kernel_initializer="he_normal"))
        self.model.add(keras.layers.Dense(750, activation="relu", kernel_initializer="he_normal"))
        self.model.add(keras.layers.Dense(1000, activation="relu", kernel_initializer="he_normal"))
        self.model.add(keras.layers.Dense(750, activation="relu", kernel_initializer="he_normal"))
        self.model.add(keras.layers.Dense(number_of_actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

        # Neural model target
        self.model_target = keras.models.Sequential()
        self.model_target.add(keras.layers.Dense(50, input_shape=(number_of_observations,), activation="relu", kernel_initializer="he_normal"))
        self.model_target.add(keras.layers.Dense(750, activation="relu", kernel_initializer="he_normal"))
        self.model_target.add(keras.layers.Dense(1000, activation="relu", kernel_initializer="he_normal"))
        self.model_target.add(keras.layers.Dense(750, activation="relu", kernel_initializer="he_normal"))
        self.model_target.add(keras.layers.Dense(number_of_actions, activation="linear"))
        self.model_target.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, terminal_state):
        # Store a tuple (s, a, r, s') for experience replay
        
        self.memory.store_transition(state, action, reward, next_state, terminal_state)

    def select(self, state, exploration_rate):
        # Generate an action for a given state using epsilon-greedy policy
        if np.random.rand() < exploration_rate:
            return random.randrange(self.number_of_actions)
        else:
            state = np.reshape(state, [1, self.number_of_observations])
            q_values = self.model(state).numpy()
            return np.argmax(q_values[0])

    def select_greedy_policy(self, state):
        # Generate an action for a given state using greedy policy
        
        q_values = self.model(state).numpy()
        return np.argmax(q_values[0])

    def learn(self, learn: bool):
        # Learn the value Q using a sample of examples from the replay memory
        
        if self.memory.current_size < BATCH_SIZE: return
        

        states, actions, rewards, next_states, terminal_states = self.memory.sample_memory(BATCH_SIZE)

        q_targets = self.model(states).numpy()
        q_next_states = self.model(next_states).numpy()
        

        for i in range(BATCH_SIZE):
             if (terminal_states[i]):
                  q_targets[i][actions[i]] = rewards[i]
             else:
                  q_targets[i][actions[i]] = rewards[i] + GAMMA * np.max(q_next_states[i])

        self.model.train_on_batch(states, q_targets)

        if learn:
            print("Weights has been updated")
            weights = self.model.get_weights()
            self.model_target.set_weights(weights)

    def add_score(self, score):
        # Add the obtained score to a list to be presented later
        self.scores.append(score)

    def delete_scores(self):
        # Delete the scores
        self.scores = []


    def display_scores_graphically(self):
        # Display the obtained scores graphically
        plt.plot(self.scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")

class State():
    def __init__(self):
        self.ball_detected = False
        self.ball_centroid_x = 0.0
        self.ball_centroid_y = 0.0
        self.ball_area = 0.0
        

    def list_state(self):
        return [self.ball_detected, self.ball_centroid_x, self.ball_centroid_y, self.ball_area]

class P3DX():

    num_sonar = 16
    sonar_max = 1.0

    def __init__(self, sim, robot_id, use_camera=True, use_lidar=False):
        self.sim = sim
        print('*** getting handles', robot_id)
        self.left_motor = self.sim.getObject(f'/{robot_id}/leftMotor')
        self.right_motor = self.sim.getObject(f'/{robot_id}/rightMotor')
        self.robot = self.sim.getObject(f'/{robot_id}')
        self.sonar = []
        for i in range(self.num_sonar):
            self.sonar.append(self.sim.getObject(f'/{robot_id}/ultrasonicSensor[{i}]'))
        if use_camera:
            self.camera = self.sim.getObject(f'/{robot_id}/camera')
        if use_lidar:
            self.lidar = self.sim.getObject(f'/{robot_id}/lidar')
        self.learning_network = DQN(4,6)
        self.state = State()
        self.readings = self.get_sonar()
        self.ball_in_view_time = 0

    def get_position(self):
        actual_position = self.sim.getObjectPosition(self.robot)
        return actual_position
    
    def get_sonar(self):
        readings = []
        for i in range(self.num_sonar):
            res,dist,_,_,_ = self.sim.readProximitySensor(self.sonar[i])
            readings.append(dist if res == 1 else self.sonar_max)
        return readings

    def get_action(self, exploration_rate, test: bool = False):
        if not test:
            return self.learning_network.select(self.state.list_state(), exploration_rate)
        else:
            return self.learning_network.select_greedy_policy(self.state.list_state())
        
    def get_image(self):
        img, resX, resY = self.sim.getVisionSensorCharImage(self.camera)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        return img

    def get_lidar(self):
        data = self.sim.getStringSignal('PioneerP3dxLidarData')
        if data is None:
            return []
        else:
            return self.sim.unpackFloatTable(data)

    def set_speed(self, action):
        if action == 0:
            ispeed, rspeed = self.turn_right(0)
        if action == 1:
            ispeed, rspeed = self.turn_right(0.75)
        if action == 2:
            ispeed, rspeed = self.move_forward()
        if action == 3:
            ispeed, rspeed = self.turn_left(0.75)
        if action == 4:
            ispeed, rspeed = self.turn_left(0)
        if action == 5:
            ispeed, rspeed = self.move_backward()
        self.sim.setJointTargetVelocity(self.left_motor, ispeed)
        self.sim.setJointTargetVelocity(self.right_motor, rspeed)

    def move_forward(self):
        ispeed, rspeed = 1.5, 1.5
        return ispeed, rspeed
    
    def move_backward(self):
        ispeed, rspeed = -0.5, -0.5
        return ispeed, rspeed

    def turn_right(self, speed):
        ispeed, rspeed = 1.5, speed
        return ispeed, rspeed

    def turn_left(self, speed):
        ispeed, rspeed = speed, 1.5
        return ispeed, rspeed
    
    def process_image(self, image):
        
        lower_red = np.array([0, 0, 20])
        upper_red = np.array([30, 30, 255])
        mask = cv2.inRange(image, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    area = M['m00']
                    return True,cx,cy,area
        
        return False,0,0,0

    def update_state(self):
        
        image = self.get_image()
        ball_detected, ball_centroid_x, ball_centroid_y, ball_area = self.process_image(image)
        self.state.ball_area = ball_area
        self.state.ball_detected = ball_detected
        self.state.ball_centroid_x = ball_centroid_x
        self.state.ball_centroid_y = ball_centroid_y
        

        return  self.state
    
    
                
        
    def set_position(self, x,y):
        actual_position = self.sim.getObjectPosition(self.robot)
        self.sim.setObjectPosition(self.robot, [x, y, actual_position[2]])
        
    def set_orientation(self, angle):
        self.sim.setObjectOrientation(self.robot, [0, 0, angle])


class environment():
    def __init__(self, robot_id, sim):
        self.actions = [0, 1, 2, 3, 4, 5] # 0: left, 1: fordward-left, 2:fordward , 3: fordward-right, 4: right
        self.number_of_observations = 4 # fordward-metrics, left-metrics, right-metrics, right-wall-metrics, left-wall-metrics, wandering, front-wall, right-wall, left-wall
        self.number_of_actions = 6
        self.state = State()
        self.robot = P3DX(sim, robot_id)



    def is_red_ball_visible(self,image):
        #Recompensa por mantener la bola roja en el campo de vision
        
        # Definir el rango de colores rojos en BGR
        lower_red = np.array([0, 0, 20])
        upper_red = np.array([30, 30, 255])

    # Crear una máscara para los píxeles que caen dentro del rango de rojo
        mask = cv2.inRange(image, lower_red, upper_red)
    # Encontrar contornos de los objetos rojos en la imagen
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return len(contours) > 0

    """"Crear funcion is_following_ball que si sigue la bola recompensa (es decir, bola en el campo de vision y distancia cercana ) 
    Divodir con is_wall_near para que no se acerque a la pared, si se acerca a la pared se penaliza. A no ser que sea necesario para seguir la bola"""
    """Si toca pared, volver a posicion inicial y penalizar"""


   
    def reward(self, robot, readings, action):
        image = robot.get_image()
        
        ball_detected, cx, cy, ball_area = self.robot.process_image(image)

        # Actualizar el estado del robot
        
        #elf.state.update(ball_detected, ball_centroid_x, ball_centroid_y, ball_area)


        # Definir la recompensa basada en la acción
        reward = 0

        if self.state.ball_detected or ball_detected:
            center_x = 256 // 2
            offset = abs(cx - center_x)
            
            # Recompensa por detectar la bola
            reward += 20
            max_reward_centering = 20
            reward_centering = max_reward_centering * (1 - (offset / center_x))
            reward += reward_centering

            # Recompensa basada en el área (distancia a la bola)
            optimal_area = 7000000  # valor óptimo aproximado del área
            max_reward_distance = 10
            reward_distance = max_reward_distance * (1 - abs(ball_area - optimal_area) / optimal_area)
            reward += reward_distance

            print(f'reward_centering: {reward_centering}, reward_distance: {reward_distance}')

            if action == 0:  # Girar a la derecha sin moverse
                if self.state.ball_centroid_x > center_x:
                    reward += 10  # Recompensa por mover la bola hacia el centro
                else:
                    reward -= 5  # Penalización por alejar la bola del centro

            elif action == 1:  # Girar a la derecha moviéndose
                if self.state.ball_centroid_x  > center_x:
                    reward += 10  # Recompensa por mover la bola hacia el centro
                else:
                    reward -= 5  # Penalización por alejar la bola del centro

            elif action == 2:  # Avanzar
                if 5500000 < self.state.ball_area < 8500000:
                    reward += 10  # Recompensa por mantener una distancia adecuada
                else:
                    reward -= 5  # Penalización por estar demasiado lejos o demasiado cerca

            elif action == 3:  # Girar a la izquierda moviéndose
                if self.state.ball_centroid_x  < center_x:
                    reward += 10  # Recompensa por mover la bola hacia el centro
                else:
                    reward -= 5  # Penalización por alejar la bola del centro

            elif action == 4:  # Girar a la izquierda sin moverse
                if self.state.ball_centroid_x < center_x:
                    reward += 10  # Recompensa por mover la bola hacia el centro
                else:
                    reward -= 5  # Penalización por alejar la bola del centro

            elif action == 5:  # Retroceder
                if self.state.ball_area > 9500000:
                    reward += 10  # Recompensa por mantener una distancia adecuada
                reward -= 1  # Penalización por retroceder innecesariamente
            
            self.robot.ball_in_view_time += 1
            reward += self.robot.ball_in_view_time * 2  # Recompensa por mantener la bola en el campo de visión
        else:
            reward -= 10  # Penalización por perder la bola
            self.robot.ball_in_view_time = 0

        # Penalización por acercarse demasiado a los obstáculos
        if any(reading <0.1 for reading in readings):
             reward -= 5
             robot.set_position(+0.65553, -0.650)
             robot.set_orientation(+45.00)

        return reward