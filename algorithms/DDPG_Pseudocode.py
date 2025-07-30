# By: Mohammad Kadkhodaei
# Date : 1404-05-08
# -----------------------

# DDPG Pseudocode in PyTorch Style
# Key Components:
# 1. Actor-Critic Networks
# 2. Target Networks
# 3. Replay Buffer
# 4. Ornstein-Uhlenbeck Noise

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ---------------------------- #
# 1. Hyperparameters
# ---------------------------- #
LR_ACTOR = 1e-4      # Learning rate for actor
LR_CRITIC = 1e-3     # Learning rate for critic
GAMMA = 0.99         # Discount factor
TAU = 1e-3           # Soft update parameter
BATCH_SIZE = 64      # Minibatch size
BUFFER_SIZE = 1e6    # Replay buffer size
EXPLORE_NOISE = 0.1  # Exploration noise scale

# ---------------------------- #
# 2. Actor Network (Policy)
# ---------------------------- #
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # Simple 3-layer MLP
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action  # Scale output to action space bounds
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # Use tanh to bound actions between -1 and 1 (scale later)
        return self.max_action * torch.tanh(self.fc3(x))

# ---------------------------- #
# 3. Critic Network (Q-function)
# ---------------------------- #
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # State pathway
        self.fc1 = nn.Linear(state_dim, 400)
        # Action gets concatenated at second layer
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)  # Q-value output
        
    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q(s,a)

# ---------------------------- #
# 4. Replay Buffer
# ---------------------------- #
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def store(self, state, action, reward, next_state, done):
        # Store transition in buffer
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # Randomly sample batch of transitions
        batch = random.sample(self.buffer, batch_size)
        # Unpack and convert to tensors
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states),
                torch.FloatTensor(actions),
                torch.FloatTensor(rewards).unsqueeze(1),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones).unsqueeze(1))
    
    def __len__(self):
        return len(self.buffer)

# ---------------------------- #
# 5. Ornstein-Uhlenbeck Noise
# ---------------------------- #
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self):
        # Generate correlated noise for exploration
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

# ---------------------------- #
# 6. DDPG Agent
# ---------------------------- #
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # Replay buffer and noise
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.noise = OUNoise(action_dim)
        
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        if add_noise:
            action += self.noise.sample() * EXPLORE_NOISE
        return np.clip(action, -self.actor.max_action, self.actor.max_action)
    
    def train(self):
        # Skip if not enough samples
        if len(self.buffer) < BATCH_SIZE:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        
        # ---------------------------- #
        # Critic Loss
        # ---------------------------- #
        with torch.no_grad():
            # Target Q = r + γ * Q'(s', π'(s'))
            target_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, target_actions)
            target_Q = rewards + (1 - dones) * GAMMA * target_Q
        
        # Current Q estimate
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        # Update critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # ---------------------------- #
        # Actor Loss
        # ---------------------------- #
        # Policy gradient: maximize Q(s, π(s))
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # ---------------------------- #
        # Soft Update Target Networks
        # ---------------------------- #
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])

# ---------------------------- #
# 7. Training Loop
# ---------------------------- #
def train_ddpg(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        agent.noise.reset()
        
        while True:
            # Select action with exploration noise
            action = agent.select_action(state, add_noise=True)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
            agent.buffer.store(state, action, reward, next_state, done)
            
            # Train agent
            agent.train()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        print(f"Episode {episode}, Reward: {episode_reward}")
        
        # Optional: save model periodically
        if episode % 100 == 0:
            agent.save(f"ddpg_model_{episode}.pth")

# ---------------------------- #
# 8. Main Execution
# ---------------------------- #
if __name__ == "__main__":
    # Initialize environment (replace with actual env)
    env = YourEnvironment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Create agent
    agent = DDPGAgent(state_dim, action_dim, max_action)
    
    # Train the agent
    train_ddpg(env, agent)
