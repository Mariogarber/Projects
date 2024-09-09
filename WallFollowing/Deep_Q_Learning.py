from class_robots import Coppelia, P3DX, state, environment

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

def writte_log(state, action, reward, score, episode, step):
    with open("log.txt", "a") as file:
        file.write(f"State: {state}, Action: {action}, Reward: {reward}, Score: {score}, Episode: {episode}, Step: {step}\n")

def main():
    coppelia = Coppelia()
    env = environment(robot_id= "PioneerP3DX", sim=coppelia.sim)
    robot = env.robot

    episode = 0
    # start_time = time.perf_counter()
    total_steps = 1
    exploration_rate = EXPLORATION_MAX
    goal_reached = False
    learn_batch = 0
    while (episode < MAX_NUMBER_OF_EPISODES_FOR_TRAINING):
        episode += 1
        steps = 1
        score = 0
        state = robot.set_state()
        end_episode = False
        while steps < 5000:
            learn_batch +=1
            # Select an action for the current state
            action = robot.set_speed(exploration_rate)

        # Execute the action on the environment
            reward = environment.reward()
            state_next = robot.set_state()

        # Store in memory the transition (s,a,r,s')
            robot.learning_network.remember(state, action, reward, state_next)

            score += reward

        # Learn using a batch of experience stored in memory
        if learn_batch == LEARN_BATCH:
            robot.learning_network.learn(True)
            learn_batch = 0
        else:
            robot.learning_network.learn(False)

        # Detect end of episode
        # if terminal_state or truncated:
        #     end_episode = True
        #     agent.add_score(score)
        #     average_score = agent.average_score(NUMBER_OF_EPISODES_FOR_TESTING_GOAL_SCORE)
        #     if average_score >= GOAL_SCORE: goal_reached = True
        #     print("Episode {0:>3}: ".format(episode), end = '')
        #     print("score {0:>3} ".format(math.trunc(score)), end = '')
        #     print("(exploration rate: %.3f, " % exploration_rate, end = '')
        #     print("average score: {0:>3}, ".format(round(average_score)), end = '')
        #     print("transitions: " + str(agent.memory.current_size) + ")")
        # else:
        state = state_next
        total_steps += 1
        steps += 1

        writte_log(state.list_state(), action, reward, score, episode, steps)

    # Decrease exploration rate
        exploration_rate *= EXPLORATION_DECAY
        exploration_rate = max(EXPLORATION_MIN, exploration_rate)


if __name__ == "__main__":
    main()