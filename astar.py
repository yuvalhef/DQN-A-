from heapq import *
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




def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def dqn_heuristic(env):
    return

def astar(env, start, goal):
    counter = 0
    # neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    array = env.env.maze.grid
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:
        counter += 1
        current = heappop(oheap)[1]
        env.a_step(current)

        env.render()
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            # print(data)
            # print(counter)
            return data, counter

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))


    return False

#
# maze = np.array([
#     [1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
#     [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
#     [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
#     [0., 0., 1., 0., 0., 1., 0., 1., 1., 1.],
#     [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
#     [1., 1., 0., 1., 0., 1., 1., 1., 1., 1.],
#     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#     [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
#     [1., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
#     [1., 1., 1., 1., 1., 1., 1., 0., 1., 1.]
# ])

# for i in range(len(maze)):
#     for j in range(len(maze[i])):
#         if maze[i][j] == 1.:
#             maze[i][j] = 0
#         else:
#             maze[i][j] = 1
# start = (0, 0)
# end = (9, 9)
# astar(maze, start, end)

# Create our environment
# env = gym.make("Maze-Img-15x15-NormalMaze-v0")

# print("The size of our frame is: ", env.observation_space)
# print("The action size is : ", 4)
# env.reset()

# start = env.env.player
# end = env.env.target
#astar(env, start, end)
# state, reward, done, _ = env.step(0)
# env.observation_space
# print('biii')
# env.render()
