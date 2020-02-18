import numpy as np
import gym
import random


class q_runner:

    def __init__(self, grid, env, fixed=False):
        # MODEL HYPERPARAMETERS
        self.grid = grid
        self.states = grid.copy()
        state = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i][j] == 0:
                    self.states[i][j] = state
                    state += 1
        self.state_size = len(self.states)
        self.action_size = 4  # 4 possible actions
        self.qtable = np.zeros((state+1, self.action_size))

        self.learning_rate = 0.1  # Alpha (aka learning rate)

        ### TRAINING HYPERPARAMETERS
        self.total_episodes = 200  # Total episodes for training
        self.max_steps = 10000  # Max possible steps in an episode

        # Exploration parameters for epsilon greedy strategy
        self.epsilon = 1.0  # Exploration rate
        self.max_epsilon = 1.0  # Exploration probability at start
        self.min_epsilon = 0.01  # Minimum exploration probability
        self.decay_rate = 0.7

        # Q learning hyperparameters
        self.gamma = 0.99  # Discounting rate

        # TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
        self.episode_render = True
        self.env = env

    def train(self):
        for episode in range(self.total_episodes):
            # Reset the environment
            self.env.fixed_reset(self.grid)
            state = self.states[self.env.env.player]

            step = 0
            done = False

            for step in range(self.max_steps):
                # 3. Choose an action a in the current world state (s)
                ## First we randomize a number
                exp_exp_tradeoff = random.uniform(0, 1)

                ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
                if exp_exp_tradeoff > self.epsilon:
                    action = np.argmax(self.qtable[state, :])

                # Else doing a random choice --> exploration
                else:
                    action = np.random.randint(4, size=1)
                    # action = self.possible_actions[choice]

                # Take the action (a) and observe the outcome state(s') and reward (r)
                _, reward, done, info = self.env.step(action)
                new_state = self.states[self.env.env.player]
                self.env.render()
                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.qtable[state, action] = self.qtable[state, action] + self.learning_rate * (reward + self.gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])


                # Our new state is state
                state = new_state

                # If done : finish episode
                if done == True:
                    print('Goal in step: '+str(step)+' P = '+str(self.epsilon) + ' episode: '+str(episode))
                    break

            # Reduce epsilon (because we need less and less exploration)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)


    def test(self):
        print('test q-learning')
        rewards = []
        data = []
        for episode in range(1):
            self.env.fixed_reset(self.grid)
            state = self.states[self.env.env.player]
            step = 0
            done = False
            total_rewards = 0
            # print("****************************************************")
            # print("EPISODE ", episode)
            steps = 0
            for step in range(self.max_steps):
                steps += 1
                # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
                # env.render()
                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(self.qtable[state, :])

                _, reward, done, info = self.env.step(action)
                new_state = self.states[self.env.env.player]
                self.env.render()
                total_rewards += reward

                if done:
                    rewards.append(total_rewards)
                    # print ("Score", total_rewards)
                    print("Score", total_rewards)
                    return data, steps

                state = new_state
        self.env.close()
        print("Score over time: " + str(sum(rewards) / 1))
