import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from wrapper import make_env
from model import DQN

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    def __len__(self):
        return len(self.buffer)

def process_state(state, device):
    state = np.array(state).squeeze(-1)
    state = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
    return state

BATCH_SIZE = 32
GAMMA = 0.99            # fator de importancia das recompensas futuras
EPSILON_START = 1.0     # começa a explorar 100% das vezes
EPSILON_END = 0.02      # exploração minima de 2%
EPSILON_DECAY = 100000  # velocidade com que o epsilon cai
TARGET_UPDATE = 1000    #a cada quantos passos eh atualizada a rede alvo
MEMORY_SIZE = 50000
LR = 0.00025
EPISODES = 5000

def main():
    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"treinando em: {device}")
    
    env = make_env()
    num_actions = env.action_space.n
    input_shape = (4, 84, 84)
    policy_net = DQN(input_shape, num_actions).to(device) #policy toma decisões
    target_net = DQN(input_shape, num_actions).to(device) #target calcula recompensas esperadas
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    steps_done = 0
    
    #loop de treino
    for episode in range(EPISODES):
        state = env.reset()
        state = process_state(state, device)
        total_reward = 0
        done = False

        while not done:
            # Epsilon-Greedy: Explorar vs Explorar
            epsilon = EPSILON_END + (EPSILON_START-EPSILON_END) * \
                      np.exp(-1. * steps_done/EPSILON_DECAY)
            
            if random.random() > epsilon:
                with torch.no_grad():
                    action = policy_net(state).argmax(dim=1).item()
            else:
                action = env.action_space.sample()
                
            # Executar ação no ambiente
            next_state_raw, reward, done, _ = env.step(action)
            next_state = process_state(next_state_raw, device)
            
            # Guardar na memória (convertendo os tensores de volta para numpy para poupar RAM)
            memory.push(state.cpu().numpy()[0], action, reward, next_state.cpu().numpy()[0], done)
            
            state = next_state
            total_reward += reward
            steps_done += 1
            
            # Otimização da Rede
            if len(memory) > BATCH_SIZE:
                b_states, b_actions, b_rewards, b_next_states, b_dones = memory.sample(BATCH_SIZE)
                
                b_states = torch.FloatTensor(b_states).to(device)
                b_actions = torch.LongTensor(b_actions).unsqueeze(1).to(device)
                b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(device)
                b_next_states = torch.FloatTensor(b_next_states).to(device)
                b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(device)
                
                # Valores Q atuais
                curr_Q = policy_net(b_states).gather(1, b_actions)
                
                # Valores Q alvo
                with torch.no_grad():
                    next_Q = target_net(b_next_states).max(1)[0].unsqueeze(1)
                    target_Q = b_rewards + (GAMMA * next_Q * (1 - b_dones))
                
                # Calcular perda e otimizar
                loss = nn.SmoothL1Loss()(curr_Q, target_Q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Sincronizar a Target Net periodicamente
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episódio: {episode} | Recompensa Total: {total_reward:.1f} | Epsilon: {epsilon:.3f}")
        # Guardar checkpoint a cada 50 episódios
        if episode % 50 == 0:
            torch.save(policy_net.state_dict(), f"checkpoints/dqn_mario_ep_{episode}.pth")

if __name__ == "__main__":
    main()