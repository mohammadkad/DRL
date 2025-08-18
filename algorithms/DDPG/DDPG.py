# by: Mohammad kadkhodaei
# 1404-05-05

'''
DDPG, or Deep Deterministic Policy Gradient, is an off-policy reinforcement learning algorithm designed for continuous action spaces. 
It combines the strengths of actor-critic methods and deep Q-networks, 
utilizing a deterministic policy (actor) and a Q-value function (critic) to learn optimal policies. 
Key features include an experience replay buffer, target networks for stability, and batch normalization. 
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import copy

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.float32)
    
    def __len__(self):
        return len(self.buffer)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
    
    def forward(self, x, u):
        x = F.relu(self.layer1(torch.cat([x, u], 1)))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay_buffer = ReplayBuffer(1000000)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        self.tau = 0.001
        self.gamma = 0.99
        self.batch_size = 64
    
    def select_action(self, state, noise=None):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise is not None:
            action += noise.sample()
        return action.clip(-self.max_action, self.max_action)
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Critic loss
        next_action = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_action)
        target_Q = reward + (1 - done) * self.gamma * target_Q.detach()
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

# Example usage
if __name__ == "__main__":
    # Environment parameters
    state_dim = 3  # Replace with your environment's state dimension
    action_dim = 1  # Replace with your environment's action dimension
    max_action = 1.0  # Replace with your environment's max action value
    
    # Initialize DDPG agent
    agent = DDPG(state_dim, action_dim, max_action)
    
    # Training loop (pseudo-code)
    for episode in range(1000):
        state = np.random.randn(state_dim)  # Replace with environment reset
        episode_reward = 0
        done = False
        
        while not done:
            # Select action with exploration noise
            noise = np.random.normal(0, 0.1, size=action_dim)  # Or use Ornstein-Uhlenbeck process
            action = agent.select_action(state, noise)
            
            # Take action in environment
            next_state = np.random.randn(state_dim)  # Replace with environment step
            reward = 0.1  # Replace with actual reward
            done = False  # Replace with actual done flag
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train agent
            agent.train()
            
            state = next_state
            episode_reward += reward
        
        print(f"Episode: {episode}, Reward: {episode_reward}")
