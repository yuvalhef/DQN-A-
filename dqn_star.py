#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 20:12:23 2018

@author: matan
"""

from dqn import dqn_runner, DQNetwork
import astar
import qstar
import qstar_qlearn
from q_maze import q_runner
import gym
import tensorflow as tf      # Deep Learning library
import gym_maze
import pygame
from skimage import transform  # Help us to preprocess the frames
from skimage.color import rgb2gray  # Help us to gray our frames
import pickle
import numpy as np


def is_arr_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if elem is myarr), False)

MODES = {
    1: 'train',
    2: 'experiment1',  # test on 10 unseen mazes 10X10
    3: 'experiment2',  # test on 5 seen mazes 20X20
    4: 'experiment3'   # test on same maze with alternating goal locations
}
MODE = 4


def train(grids, env):
    game = dqn_runner(grids=grids, env=env)
    game.episode_render = False
    game.Initialized()
    game.train(name='train')
    return grids


def play_1(grids, env):
    model_path = './models/model_train1730.ckpt'  # change this
    dqn_r = dqn_runner(grids, env)
    tf.reset_default_graph()
    network = DQNetwork(dqn_r.state_size, dqn_r.action_size, dqn_r.learning_rate)
    astar_total = []
    qstar_total = []
    new_grids = []
    stop = 0
    # for grid in grids:
    for i in range(1000):
        env.reset()
        curr_grid = env.env.maze.grid
        start = env.env.player
        end = env.env.target
        states_qstar, counter_qstar = qstar.qstar(env, start, end, network, model_path)
        env.fixed_reset(curr_grid)
        states_astar, counter_astar = astar.astar(env, start, end)
        if counter_astar > 39:
            print('states astar: ', counter_astar, ' states qstar: ', counter_qstar)
            qstar_total.append(counter_qstar)
            astar_total.append(counter_astar)
            new_grids.append(env.env.maze.grid)
            stop += 1
        if stop == 19:
            break
    print('Average astar: ' + str(np.mean(astar_total)))
    print('Average qstar: ' + str(np.mean(qstar_total)))
    pickle.dump(grids, open("grids_20_10.p", "wb"))
    return


def play_2(grids, env):
    astar_total = []
    qstar_total = []
    dqn_total = []
    i = 0
    for i in range(5):
        # train on grid
        env.reset()
        grid = env.env.maze.grid
        game = dqn_runner(grid, env, fixed=True)
        game.max_steps = 100000
        game.total_episodes = 200
        game.episode_render = True
        game.learning_rate = 0.1
        game.decay_rate = 0.0001
        game.max_tau = 500
        game.memory_size = 10000
        game.Initialized()
        last_model = game.train(name='exp2_grid'+str(i))
        model_path = './models/model_exp2_grid'+str(i)+str(last_model)+'.ckpt'

        # Use trained model on same grid for testing
        tf.reset_default_graph()
        network = DQNetwork(game.state_size, game.action_size, game.learning_rate)
        env.fixed_reset(grid)
        start = env.env.player
        end = env.env.target
        states_astar, counter_astar = astar.astar(env, start, end)
        env.fixed_reset(grid)
        states_qstar, counter_qstar = qstar.qstar(env, start, end, network, model_path)
        env.fixed_reset(grid)
        states_dqn, counter_dqn = game.test(env, network, model_path)
        print('states astar: ', counter_astar, ' states qstar: ', counter_qstar, ' states dqn: ', counter_dqn)
        astar_total.append(counter_astar)
        qstar_total.append(counter_qstar)
        dqn_total.append(counter_dqn)
        i += 1

    print('Average astar: ' + str(np.mean(astar_total)))
    print('Average qstar: ' + str(np.mean(qstar_total)))
    print('Average dqn: ' + str(np.mean(dqn_total)))
    return


def play_3(env):
    astar_total = []
    qstar_total = []
    q_total = []

    for i in range(10):
        # train on grid
        env.reset()
        grid = env.env.maze.grid
        game = q_runner(grid, env, fixed=True)
        game.train()
        env.fixed_reset(grid)
        start = env.env.player
        end = env.env.target
        print('astar')
        states_astar, counter_astar = astar.astar(env, start, end)
        env.fixed_reset(grid)
        print('q-learn')
        states_q, counter_q = game.test()
        env.fixed_reset(grid)
        print('qstar')
        states_qstar, counter_qstar = qstar_qlearn.qstar(env, start, end, game)
        print('states astar: ', counter_astar, ' states qstar: ', counter_qstar, ' states dqn: ', counter_q)
        astar_total.append(counter_astar)
        qstar_total.append(counter_qstar)
        q_total.append(counter_q)
        i += 1

    print('Average astar: ' + str(np.mean(astar_total)))
    print('Average qstar: ' + str(np.mean(qstar_total)))
    print('Average dqn: ' + str(np.mean(q_total)))
    return


if __name__ == "__main__":
    grids_10 = pickle.load(open("grids_10.p", "rb"))
    grids_20 = pickle.load(open("grids_10.p", "rb"))
    if MODES[MODE] == "train":
        env = gym.make("Maze-Arr-10x10-NormalMaze-v0")
        train(grids_10, env)
    elif MODES[MODE] == "experiment1":
        env = gym.make("Maze-Arr-10x10-NormalMaze-v0")
        play_1(grids_10, env)
    elif MODES[MODE] == "experiment2":
        env = gym.make("Maze-Arr-8x8-NormalMaze-v0")
        play_2(grids_20, env)

    elif MODES[MODE] == "experiment3":
        env = gym.make("Maze-Arr-12x12-NormalMaze-v0")
        play_3(env)

