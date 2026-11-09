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
        obs, reward, done, info = self.env.step(action)
        
        # 1. Diferença de posição (avançar para direita = +1, esquerda = -1 por pixel)
        delta_x = info['x_pos'] - self.current_x
        self.current_x = info['x_pos']
        
        # 2. Diferença de tempo (-1 ponto por tick do relógio)
        delta_time = self.current_time - info['time']
        self.current_time = info['time']
        
        # Calculando a recompensa base
        custom_reward = delta_x - delta_time
        
        # 3. Penalidade de morte corrigida e encerramento da tentativa
        # flag_get indica se ele tocou no mastro (ganhou a fase)
        if done and not info['flag_get']:
            custom_reward -= 15
            
        # 4. Limitar a recompensa no intervalo [-15, 15]
        custom_reward = np.clip(custom_reward, -15, 15)
        
        # Removemos os dados internos que o agente não deve receber do dicionário info
        # para garantir que ele aprenda estritamente pela imagem
        info.pop('x_pos', None)
        info.pop('time', None)
        info.pop('score', None)
        
        return obs, custom_reward, done, info

def make_env():
    # Carrega especificamente a fase 1-1 do jogo
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    
    # Aplica as 12 ações possíveis exigidas
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    # Aplica nossa lógica customizada de recompensas
    env = CustomRewardAndDoneWrapper(env)
    
    # Processamento visual: transforma em tons de cinza
    env = GrayScaleObservation(env, keep_dim=False)
    
    # Redimensiona a imagem para 84x84 pixels
    env = ResizeObservation(env, shape=84)
    
    # Agrupa os 4 últimos frames para dar noção de movimento
    env = FrameStack(env, num_stack=4)
    
    return env