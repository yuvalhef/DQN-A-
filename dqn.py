import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import gym
import gym_maze
import pygame
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import matplotlib.pyplot as plt # Display graphs
from collections import deque# Ordered collection with ends
import random

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')


class dqn_runner:
    
    def __init__(self, grids, env, fixed=False):
        # MODEL HYPERPARAMETERS
        self.state_size = [*list(env.env.maze.grid.shape), 1]  # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
        self.action_size = 4  # 4 possible actions
        self.learning_rate = 0.0001         # Alpha (aka learning rate)

        ### TRAINING HYPERPARAMETERS
        self.total_episodes = 5000          # Total episodes for training
        self.max_steps = 10000               # Max possible steps in an episode
        self.batch_size = 32                # Batch size
        self.fixed = fixed
        
        # Exploration parameters for epsilon greedy strategy
        self.explore_start = 0.99            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability
        self.decay_rate = 0.000001           # exponential decay rate for exploration prob
        
        # Q learning hyperparameters
        self.gamma = 0.99                    # Discounting rate
        
        ### MEMORY HYPERPARAMETERS
        self.pretrain_length = self.batch_size   # Number of experiences stored in the Memory when initialized for the first time
        self.memory_size = 1000000          # Number of experiences the Memory can keep
        
        ### PREPROCESSING HYPERPARAMETERS

        # MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
        self.training = True
        
        # TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
        self.episode_render = True
        self.max_tau = 1000
        # Here we create an hot encoded version of our actions
        # possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
        self.possible_actions = np.array(np.identity(4, dtype=int).tolist())
        
        self.grids = grids
        self.env = env
        

    
    def is_arr_in_list(self,myarr, list_arrays):
        return next((True for elem in list_arrays if elem is myarr), False)
    
    def Initialized(self):
        # Reset the graph
        tf.reset_default_graph()
        
        # Instantiate the DQNetwork
        self.DQNetwork = DQNetwork(self.state_size, self.action_size, self.learning_rate, name="DQNetwork")

        self.TargetNetwork = DQNetwork(self.state_size, self.action_size, self.learning_rate, name="TargetNetwork")

        # Instantiate memory
        self.memory = Memory(max_size=self.memory_size)
        for i in range(self.pretrain_length):
            # If it's the first step
            if i == 0:
                if self.fixed:
                    self.env.fixed_reset(self.grids)
                else:
                    self.env.reset()
                state = self.env.env.maze.grid.copy()
                state[self.env.env.target] = 100
                state[self.env.env.player] = 50
                state = state.reshape([*list(state.shape), 1])

        
            # Get the next_state, the rewards, done by taking a random action
            choice = random.randint(1, len(self.possible_actions)) - 1
            action = self.possible_actions[choice]
            next_state, reward, done, _ = self.env.step(np.where(action == 1)[0][0])
            next_state = self.env.env.maze.grid.copy()
            next_state[self.env.env.target] = 100
            next_state[self.env.env.player] = 50
            next_state = next_state.reshape([*list(next_state.shape), 1])

            # self.env.render()

            # If the episode is finished (we're dead 3x)
            if done:
                # We finished the episode
                next_state = np.zeros([*list(state.shape)])
        
                # Add experience to memory
                self.memory.add((state, action, reward, next_state, done))
        
                # Start a new episode
                if self.fixed:
                    self.env.fixed_reset(self.grids)
                else:
                    self.env.reset()
                state = self.env.env.maze.grid.copy()
                state[self.env.env.target] = 100
                state[self.env.env.player] = 50
                state = state.reshape([*list(state.shape), 1])


            else:
                # Add experience to memory
                self.memory.add((state, action, reward, next_state, done))
        
                # Our new state is now the next_state
                state = next_state


    def predict_action(self,explore_start, explore_stop, decay_rate, decay_step, state, actions,sess):
        
        """
        This function will do the part
        With ϵϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
        """

        # EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        # First we randomize a number
        exp_exp_tradeoff = np.random.rand()
    
        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            choice = random.randint(1, len(self.possible_actions)) - 1
            action = self.possible_actions[choice]
    
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: state.reshape((1, *state.shape))})
    
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = self.possible_actions[choice]
    
        return action, explore_probability

    def update_target_graph(self):

        # Get the parameters of our DQNNetwork
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

        # Get the parameters of our Target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

        op_holder = []

        # Update our target_network parameters with DQNNetwork parameters
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder


    def train(self, name = ''):
            # Setup TensorBoard Writer
        writer = tf.summary.FileWriter("/home/admin/Documents/")
        
        ## Losses
        tf.summary.scalar("Loss", self.DQNetwork.loss)
        
        write_op = tf.summary.merge_all()        
        
        last_model = 0
        # Saver will help us to save our model
        saver = tf.train.Saver(max_to_keep=10)
        if self.training == True:
            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())
                # saver.restore(sess, "models/model1500.ckpt")
                # Initialize the decay rate (that will use to reduce epsilon)
                decay_step = 0

                # Set tau = 0
                tau = 0
                done = False
                for episode in range(self.total_episodes):
                    rewards_list = []
                    # Set step to 0
                    step = 0

                    # Initialize the rewards of the episode
                    episode_rewards = []
        
                    # Make a new episode and observe the first state
                    if self.fixed:
                        self.env.fixed_reset(self.grids)
                    else:
                        self.env.reset()
                    state = self.env.env.maze.grid.copy()
                    state[self.env.env.target] = 100
                    state[self.env.env.player] = 50
                    state = state.reshape([*list(state.shape), 1])

                    #skip grids for test
                    
                    if not self.fixed:
                        if self.is_arr_in_list(self.env.env.maze.grid, self.grids):
                            continue

                    while step < self.max_steps:

                        step += 1
                        tau += 1
                        # Increase decay_step
                        decay_step += 1
        
                        # Predict the action to take and take it
                        action, explore_probability = self.predict_action(self.explore_start, self.explore_stop, self.decay_rate, decay_step, state,
                                                                     self.possible_actions,sess)
        
                        # Perform the action and get the next_state, reward, and done information
                        next_state, reward, done, _ = self.env.step(np.where(action == 1)[0][0])
                        next_state = self.env.env.maze.grid.copy()
                        next_state[self.env.env.target] = 100
                        next_state[self.env.env.player] = 50
                        next_state = next_state.reshape([*list(next_state.shape), 1])
                        print('\r Step {}/{} Episode {}/{} Reward {} '.format(step, self.max_steps, episode, self.total_episodes, reward), end="")
                        if self.episode_render:
                            self.env.render()
                        if done == True:
                            print('Reached goal at step: '+str(step))
                        # Add the reward to total reward
                        episode_rewards.append(reward)

                        if step == self.max_steps-1:
                            done = True
                        # If the game is finished
                        if done:
                            # The episode ends so no next state
                            next_state = np.zeros([*list(state.shape)], dtype=np.int)
        
                            # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                            # next_state = preprocess_frame(next_state)
        
                            # Set step = max_steps to end the episode
                            step = self.max_steps
        
                            # Get the total reward of the episode
                            total_reward = np.sum(episode_rewards)
        
                            print('\r Episode: {}'.format(episode),
                                  'Total reward: {}'.format(total_reward),
                                  'Explore P: {:.4f}'.format(explore_probability),
                                  'Training Loss {:.4f}'.format(loss))
        
                            rewards_list.append((episode, total_reward))
        
                            # Store transition <st,at,rt+1,st+1> in memory D
                            self.memory.add((state, action, reward, next_state, done))
        
                        else:
                            self.memory.add((state, action, reward, next_state, done))
        
                            # st+1 is now our current state
                            state = next_state
        
                        ### LEARNING PART
                        # Obtain random mini-batch from memory
                        batch = self.memory.sample(self.batch_size)
                        states_mb = np.array([each[0] for each in batch], ndmin=3)
                        actions_mb = np.array([each[1] for each in batch])
                        rewards_mb = np.array([each[2] for each in batch])
                        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                        dones_mb = np.array([each[4] for each in batch])
        
                        target_Qs_batch = []


                        # Get Q values for next_state
                        q_next_state = sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: next_states_mb})

                        # Calculate Qtarget for all actions that state
                        q_target_next_state = sess.run(self.TargetNetwork.output, feed_dict={self.TargetNetwork.inputs_: next_states_mb})

                        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                        for i in range(0, len(batch)):
                            terminal = dones_mb[i]

                            # We got a'
                            action = np.argmax(q_next_state[i])

                            # If we are in a terminal state, only equals reward
                            if terminal:
                                target_Qs_batch.append(rewards_mb[i])
        
                            else:
                                target = rewards_mb[i] + self.gamma * q_target_next_state[i][action]
                                target_Qs_batch.append(target)
        
                        targets_mb = np.array([each for each in target_Qs_batch])
        
                        loss, _ = sess.run([self.DQNetwork.loss, self.DQNetwork.optimizer],
                                           feed_dict={self.DQNetwork.inputs_: states_mb,
                                                      self.DQNetwork.target_Q: targets_mb,
                                                      self.DQNetwork.actions_: actions_mb})
        
                        # Write TF Summaries
                        summary = sess.run(write_op, feed_dict={self.DQNetwork.inputs_: states_mb,
                                                                self.DQNetwork.target_Q: targets_mb,
                                                                self.DQNetwork.actions_: actions_mb})
                        writer.add_summary(summary, episode)
                        writer.flush()

                        if tau > self.max_tau:
                            # Update the parameters of our TargetNetwork with DQN_weights
                            update_target = self.update_target_graph()
                            sess.run(update_target)
                            tau = 0
        
                    # Save model every 5 episodes
                    if episode % 5 == 0:
                        try:
                            save_path = saver.save(sess, "./models/model_"+name+str(episode)+".ckpt")
                            print("Model Saved")
                            last_model = episode
                        except:
                            continue
        return last_model


    def test(self, env, net, model_path):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            total_test_rewards = []

            data = []
            for episode in range(1):
                total_rewards = 0
                steps = 0
                state = env.env.maze.grid.copy()
                state[env.env.target] = 100
                state[env.env.player] = 50
                state = state.reshape([*list(state.shape), 1])
                data.append(env.env.player)

                print("****************************************************")
                print("EPISODE ", episode)

                while True:
                    steps += 1
                    # Reshape the state
                    # Get action from Q-network
                    # Estimate the Qs values state
                    Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape))})

                    # Take the biggest Q value (= the best action)
                    choice = np.argmax(Qs)
                    action = self.possible_actions[choice]

                    # Perform the action and get the next_state, reward, and done information
                    while True:
                        next_state, reward, done, _ = env.step(np.where(action == 1)[0][0])
                        next_state = env.env.maze.grid.copy()
                        next_state[env.env.target] = 100
                        next_state[env.env.player] = 50
                        next_state = next_state.reshape([*list(next_state.shape), 1])
                        if reward == -1:
                            action = self.possible_actions[choice]
                            Qs[0][choice] = -10000
                            choice = np.argmax(Qs)
                            action = self.possible_actions[choice]
                        else:
                            break
                    env.render()
                    data.append(env.env.player)

                    total_rewards += reward
                    if done:
                        print("Score", total_rewards)
                        total_test_rewards.append(total_rewards)
                        return data, steps

                    state = next_state


# stack_size = 1  # We stack 4 frames

# Initialize deque with zero-images one array for each image
# stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *self.state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=16,
                                          kernel_size=[5, 5],
                                          strides=[1, 1],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")

            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=32,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")

            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=64,
                                          kernel_size=[2, 2],
                                          strides=[1, 1],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]




#env = gym.make("Maze-Img-20x20-NormalMaze-v0")
#
## Create our environment
#
#print("The size of our frame is: ", env.observation_space)
#print("The action size is : ", 4)
#game = dqn_runner(env)
#game.Initialized()
#game.train()
