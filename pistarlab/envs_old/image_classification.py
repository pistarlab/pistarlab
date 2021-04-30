from gym import spaces
import gym
import mnist
import numpy

from PIL import Image

class ImageClassification(gym.Env):


    def __init__(self, dataset_name):

        if dataset_name is "mnist":
            resolution = (28,28)
        else:
            resolution = (30,30)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(0, 255, (resolution[0], resolution[1]))
        self.index = 0
        self.episode_size = 100
        self.step_counter =0
        self.train_images = None
        self.train_names  = None
        self.new_epoch()

    def reset(self):

        if len(self.train_images) == self.index:
            self.new_epoch()

        self.last_index = self.index
        self.step_counter =1
        self.index +=1
        return self.train_images[self.last_index]

    def new_epoch(self):
        self.index = 0
        self.train_images = mnist.train_images()
        self.train_names = mnist.test_labels()

    def step(self, action):
        if action == self.train_names[self.last_index]:
            reward = 1
        else:
            reward = 0
        
        ob = self.train_images[self.index]
        self.last_index = self.index
        self.index +=1
        self.step_counter +=1

        if self.step_counter >= self.episode_size or len(self.train_images) == self.index:
            done = True
        else:
            done = False

        
        return ob, reward, done, None


    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            return Image.fromarray(self.train_images[self.last_index])