# Basic DQN without a target network
# 1404-06-24
# Mohammad Kadkhodaei
# ---

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt

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

class BasicDQNAgent:
    """Basic DQN Agent without target network"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.learning_rate = 0.001
        
        # Single Q-network (no target network)
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():  # No gradient needed for inference
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0  # Return 0 loss if not enough samples
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Get current Q values for taken actions
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values using the same network (no target network)
        with torch.no_grad():  # No gradient for target calculation
            next_q = self.q_network(next_states).max(1)[0]
        
        # Calculate target Q values
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Compute loss
        loss = self.loss_function(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def step(self, state, action, reward, next_state, done):
        """Process one step of the environment"""
        self.remember(state, action, reward, next_state, done)
        
        # Learn from replay buffer
        loss = self.replay()
        return loss

def train_basic_dqn(env_name='CartPole-v1', episodes=500):
    """Training function for basic DQN without target network"""
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = BasicDQNAgent(state_size, action_size)
    scores = []
    losses = []
    
    print("Training Basic DQN (without target network)...")
    print("This will demonstrate why target networks are important for stability.")
    
    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        episode_losses = []
        
        while True:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            loss = agent.step(state, action, reward, next_state, done)
            if loss > 0:
                episode_losses.append(loss)
            
            state = next_state
            score += reward
            
            if done:
                break
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        scores.append(score)
        losses.append(avg_loss)
        
        # Calculate moving average of last 100 episodes
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        print(f'Episode {episode:3d}, Score: {score:6.1f}, '
              f'Avg Score: {avg_score:6.1f}, Epsilon: {agent.epsilon:.3f}, '
              f'Loss: {avg_loss:.4f}')
        
        # Early stopping if solved
        if avg_score >= 195 and len(scores) >= 100:
            print(f"Solved in {episode} episodes!")
            break
    
    env.close()
    return scores, losses

def plot_results(scores, losses):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot scores
    ax1.plot(scores, alpha=0.6, label='Episode Score')
    ax1.plot(np.convolve(scores, np.ones(100)/100, mode='valid'), 
             'r-', label='Moving Average (100 episodes)')
    ax1.set_ylabel('Score')
    ax1.set_title('DQN without Target Network - Training Performance')
    ax1.legend()
    ax1.grid(True)
    
    # Plot losses
    ax2.plot(losses, alpha=0.6, label='Training Loss')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_no_target_network.png')
    plt.show()

# Run training and show results
if __name__ == "__main__":
    scores, losses = train_basic_dqn(episodes=300)
    plot_results(scores, losses)
    
    print(f"\nFinal Results:")
    print(f"Best score: {max(scores)}")
    print(f"Average of last 100 episodes: {np.mean(scores[-100:]):.1f}")
    print(f"Final epsilon: {1.0 * (0.995 ** len(scores)):.3f}")
