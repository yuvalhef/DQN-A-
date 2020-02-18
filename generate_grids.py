from dqn import dqn_runner, DQNetwork
import astar
import qstar
import dqn
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


envs = []
grids = []
env1 = gym.make("Maze-Arr-10x10-NormalMaze-v0")
env2 = gym.make("Maze-Arr-20x20-NormalMaze-v0")

# while len(envs) < 10:
#     print("continue")
#     if is_arr_in_list(env1.env.maze.grid, grids):
#         env1.reset()
#     else:
#         grids.append(env1.env.maze.grid)
#         envs.append(env1)
#         env1.reset()
# pickle.dump(grids,open("grids_10.p","wb"))

envs = []

while len(envs) < 5:
    print("continue")
    if is_arr_in_list(env2.env.maze.grid, grids):
        env1.reset()
    else:
        grids.append(env2.env.maze.grid)
        envs.append(env2)
        env2.reset()
pickle.dump(grids,open("grids_20.p","wb"))