import gym
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

class CustomRewardAndDoneWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_x = 0
        self.current_time = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.current_x = 40  #posição x inicial aproximada do mario
        self.current_time = 400 #tempo inicial padrao da fase
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action) #lembrar de atualizar reward no futuro
        delta_x = info['x_pos'] - self.current_x
        self.current_x = info['x_pos']
        delta_time = self.current_time - info['time']
        self.current_time = info['time']
        custom_reward = delta_x - delta_time
        if done and not info['flag_get']:
            custom_reward -= 15

        custom_reward = np.clip(custom_reward, -15, 15)
        info.pop('x_pos', None)
        info.pop('time', None)
        info.pop('score', None)
        return obs, custom_reward, done, info

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = CustomRewardAndDoneWrapper(env)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    
    return env