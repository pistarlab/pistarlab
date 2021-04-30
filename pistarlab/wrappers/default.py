from typing import List, OrderedDict

import gym
from gym import spaces
import numpy as np

import cv2
class ResizeImageObWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, height=42, width=42):
        super().__init__(env)
        self._height = height
        self._width = width
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._height,self._width, 3),
            dtype='uint8'
        )


    def observation(self, obs):
        
        obs = cv2.resize(obs, (self._height,self._width),  interpolation=cv2.INTER_LINEAR)
        
        return obs.astype(np.uint8)
    