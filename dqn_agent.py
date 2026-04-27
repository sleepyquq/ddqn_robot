import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 使用一个前馈神经网络(MLP)近似Q函数
        self.fc1 = nn.Linear(state_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        # 经验池
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        # 随机采样小批量进行训练
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), 
                np.array(next_states), np.array(dones, dtype=np.float32))
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 buffer_size=10000, batch_size=64, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_steps = 0
        
        # 主网络与目标网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
    def select_action(self, state, evaluate=False):
        # ε-greedy 策略
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
            
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 当前 Q 值
        q_values = self.q_network(states).gather(1, actions)
        
        # 目标 Q 值计算 (Double DQN)
        with torch.no_grad():
            # 主网络选择下一步最优动作
            best_next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            # 目标网络评估该动作的 Q 值
            max_next_q_values = self.target_network(next_states).gather(1, best_next_actions)
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
            
        # 损失计算 (均方误差)
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_steps += 1
        # 定期同步目标网络
        if self.update_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()
        
    def update_epsilon(self):
        # ε 衰减
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
