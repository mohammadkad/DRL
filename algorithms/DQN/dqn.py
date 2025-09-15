# 1404-06-02, 1404-06-24
# Mohammad Kadkhodaei
# ---
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym

class QNetwork(nn.Module):
    """Simple neural network for Q-function approximation"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent implementation"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_every = 4
        
        # Main network and target network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Initialize target network with same weights
        self.update_target_network()
        
    def update_target_network(self):
        """Update target network with current weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def step(self, state, action, reward, next_state, done, step_count):
        """Process one step of the environment"""
        self.remember(state, action, reward, next_state, done)
        
        # Learn every update_every steps
        if step_count % self.update_every == 0:
            self.replay()

def train_dqn(env_name='CartPole-v1', episodes=1000):
    """Training function"""
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        step_count = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            agent.step(state, action, reward, next_state, done, step_count)
            
            state = next_state
            score += reward
            step_count += 1
            
            if done:
                break
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        
        print(f'Episode {episode}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}')
        
        if avg_score >= 195:  # Solved condition for CartPole
            print(f"Solved in {episode} episodes!")
            break
    
    env.close()
    return scores

# Run training
if __name__ == "__main__":
    scores = train_dqn()
