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
import astar
import pickle


def get_q(state, game):
    # Load the model
    Qs = game.qtable[state]
    minimum = min(Qs)
    if minimum < 0:
        diff = minimum*-1
        Qs = Qs + diff
    summerize = sum(Qs)
    Qs = 1-(Qs/summerize)
    return Qs


def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def qstar(env, start, goal, game):

    counter = 0
    # neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    array = env.env.maze.grid
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    actions_to_neighbors = {(0,1):0, (0,-1):1,(1,0):2,(-1,0):3}
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:

        current = heappop(oheap)[1]
        counter += 1
        env.a_step(current)
        state_current = game.states[game.env.env.player]
        actions_q = get_q(state_current, game)
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
            tentative_g_score = gscore[current] + heuristic(current, neighbor)*actions_q[actions_to_neighbors[(i,j)]]
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
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)*actions_q[actions_to_neighbors[(i,j)]]
                heappush(oheap, (fscore[neighbor], neighbor))

    return False
