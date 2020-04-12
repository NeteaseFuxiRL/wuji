from abc import ABC
from abc import abstractmethod

import numpy as np
import gym
import cv2
from gym.utils import seeding
from PIL import Image


class BaseEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second' : 3}
    reward_range = (-float('inf'), float('inf'))
    
    def __init__(self):
        self.viewer = None
        self.seed()
    
    @abstractmethod
    def step(self, action):
        pass
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def get_image(self):
        pass
    
    def render(self, mode='rgb_array', max_width=20):
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_height, img_width = img.shape[:2]
        ratio = max_width/img_width
        img = Image.fromarray(img).resize([int(ratio*img_width), int(ratio*img_height)])
        img = np.asarray(img)
        if mode == 'rgb_array':
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        elif mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            
            return self.viewer.isopen
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
