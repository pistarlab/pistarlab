from gym import spaces
import gym
import mnist
import numpy as np
import random
from PIL import Image

class Memorize1(gym.Env):
    metadata = {'render.modes':['ansi']}


    def __init__(self, item_count = 10):

        self.item_count = item_count
        self.action_space = spaces.Discrete(self.item_count)
        self.observation_space = spaces.Discrete(self.item_count)
        self.index = 0
        self.episode_size = 100
        self.step_counter =0
        self.items = None
        self.names  = None
        self.new_epoch()

    def reset(self):

        if len(self.items) == self.index:
            self.new_epoch()

        self.last_index = self.index
        self.step_counter =1
        self.index +=1
        return self.items[self.last_index]

    def new_epoch(self):
        self.index = 0
        self.items = [random.randint(0,self.item_count-1) for i in range(0,self.episode_size)]
        self.names =self.items

    def step(self, action):
        if action == self.items[self.last_index]:
            reward = 1
        else:
            reward = 0
        
        ob = self.items[self.index]
        self.last_index = self.index
        self.index +=1
        self.step_counter +=1

        if self.step_counter >= self.episode_size or len(self.items) == self.index:
            done = True
        else:
            done = False

        return ob, reward, done, None


    def render(self, mode='human', close=False):
        if mode == 'ansi':
            return "{}".format(self.names[self.last_index])