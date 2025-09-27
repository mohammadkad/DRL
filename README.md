# DRL
Deep Reinforcement Learning, Life doesn’t give you its MDP; life is uncertain.

# Definition
Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.
In AI, particularly reinforcement learning (RL), a control task refers to a problem where an agent must make sequential decisions to achieve a goal. These are often modeled as Markov Decision Processes (MDPs) or Partially Observable MDPs (POMDPs).
The Markov Property implies that our agent needs only the current state to decide what action to take and not the history of all the states and actions they took before.

- state, action, reward and next state
- S0, A0, R1, S1
- The agent’s goal is to maximize its *cumulative reward*, called the **expected return**.
- The main goal of Reinforcement learning is to find the optimal policy π∗ that will maximize the expected cumulative reward.

### Observations/States Space
- State s: is a complete description of the state of the world (there is no hidden information). In a fully observed environment.
- Observation o: is a partial description of the state. In a partially observed environment.

### Action Space
The Action space is the set of all possible actions in an environment. The actions can come from a discrete or continuous space.

### Reward
![pics/Reward.jpg](https://github.com/mohammadkad/DRL/blob/main/pics/rewards.jpg)

### Type of tasks
- Episodic task : In this case, we have a starting point and an ending point (a terminal state). This creates an episode: a list of States, Actions, Rewards, and new States.
- Continuing tasks : These are tasks that continue forever (no terminal state). In this case, the agent must learn how to choose the best actions and simultaneously interact with the environment.

### The Exploration/Exploitation trade-off
- Exploration is exploring the environment by trying **random actions** in order to find more information about the environment.
- Exploitation is exploiting known information to maximize the reward.

### Main approaches for solving RL problems
The Policy π: the agent’s brain
This Policy is the function we want to learn, our goal is to find the optimal policy π*, the policy that maximizes expected return when the agent acts according to it. We find this π* through training.

- Policy-based methods (Directly) : we learn a policy function directly. We aim to optimize the policy directly without using a value function.
  1- Deterministic
  2- Stochastic
  - Subclasses:
    - Policy-Gradient methods : optimizes the policy directly by estimating the weights of the optimal policy using Gradient Ascent
      - Reinforce :
        - use Monte-Carlo sampling to estimate return (we use an entire episode to calculate the return). we have significant variance in policy gradient estimation.
        - The Monte Carlo variance, leads to slower training since we need a lot of samples to mitigate it.
- Value-based methods (Indirectly) : we learn a value function that maps a state to the expected value of being at that state. The idea is that an optimal value function leads to an optimal policy.
  - state-value function
  - action-value function
- Actor-critic method (hybrid architecture), which is a combination of value-based and policy-based methods.
  - We learn two function approximations (two neural networks) : 1- A policy that controls how our agent acts 2- A value function to assist the policy update by measuring how good the action taken is
  - stabilize the training by reducing the variance using:
    - An Actor that controls how our agent behaves (Policy-Based method)
    - A Critic that measures how good the taken action is (Value-Based method)
    - Algorithms:
      -  Advantage Actor Critic (A2C)
      -  

### TD vs MC
- Monte Carlo vs Temporal Difference Learning:
- Monte Carlo uses an entire episode of experience before learning
- Temporal Difference (TD) uses only a step (St, At, Rt+1, St+1) to learn.
1. TD (Temporal difference) methods, Temporal Difference learning combines ideas from Monte Carlo methods (learning from experience) and Dynamic Programming (bootstrapping).:
 - TD learning, TD(0)/One-Step TD, TD(λ)/Forward View
 - SARSA (On-Policy)
 - SARSE
 - Q-Learning (Off-Policy)
 - Value-Based Methods: DQN (Deep Q-Network), Double DQN (DDQN)
 - Policy-Based Methods: REINFORCE (Monte-Carlo policy-gradient), Actor-Critic
 - Advanced Actor-Critic Methods: A3C (Asynchronous Advantage Actor-Critic), A2C (Advantage Actor-Critic), DDPG (Deep Deterministic Policy Gradient), TD3 (Twin Delayed DDPG), SAC (Soft Actor-Critic)
2. MC (Monte Carlo)
 - 

### The difference between policy-based and policy-gradient methods
Policy-gradient methods, what we’re going to study in this unit, is a subclass of policy-based methods. In policy-based methods, the optimization is most of the time on-policy since for each update, we only use data (trajectories) collected by our most recent version of πθ.

The difference between these two methods lies on how we optimize the parameter θ:

- In policy-based methods, we search directly for the optimal policy. We can optimize the parameter
θ indirectly by maximizing the local approximation of the objective function with techniques like hill climbing, simulated annealing, or evolution strategies.

- In policy-gradient methods, because it is a subclass of the policy-based methods, we search directly for the optimal policy. But we optimize the parameter
θ directly by performing the gradient ascent on the performance of the objective function J(θ).

### Policy Gradient Theorem (PGT)
- The objective function gives us the performance of the agent given a trajectory (state action sequence without considering reward (contrary to an episode)), and it outputs the expected cumulative reward.

# Reinforcement Learning Algorithms Hierarchy

## 1. Policy-Based Methods (Direct Policy Optimization)
### 1.1. Gradient-Free Methods
- Cross-Entropy Method (CEM)
- Evolutionary Strategies
- Covariance Matrix Adaptation (CMA-ES)

### 1.2. Gradient-Based Methods
- REINFORCE (Monte Carlo Policy Gradient)
- Natural Policy Gradient
- Trust Region Policy Optimization (TRPO)
- Proximal Policy Optimization (PPO)

## 2. Value-Based Methods (Learn Value Function First)
### 2.1. Prediction Methods (Value Estimation)
- **Monte Carlo (MC) Methods**
  - First-Visit MC
  - Every-Visit MC
- **Temporal-Difference (TD) Methods**
  - TD(0)
  - TD(λ)
  - **SARSA** (On-Policy TD Control)
  - **Q-learning** (Off-Policy TD Control)
  - Expected SARSA

### 2.2. Approximate Methods
- Linear Function Approximation
- Deep Q-Network (DQN) and variants

## 3. Actor-Critic Methods (Hybrid Approach)
### 3.1. Basic Actor-Critic
- Advantage Actor-Critic (A2C)
- Asynchronous Advantage Actor-Critic (A3C)

### 3.2. Advanced Actor-Critic
- Soft Actor-Critic (SAC)
- Twin Delayed DDPG (TD3)
- Deep Deterministic Policy Gradient (DDPG)

## 4. Model-Based Methods
### 4.1. Learn the Model
- Dyna-Q
- Model-Based Value Expansion

### 4.2. Use the Model
- Monte Carlo Tree Search (MCTS)
- iLQR (Iterative LQR)

## 5. Classification by Update Mechanism
### 5.1. Full Returns
- **Monte Carlo** (uses complete episode returns)

### 5.2. Bootstrapping
- **Temporal-Difference** (uses estimated returns)
- Dynamic Programming (uses full model)

### 5.3. Direct Policy Search
- **Cross-Entropy Method** (optimizes policy directly)
- Policy Gradient methods

## Key Relationships:
- **CEM** ↔ Policy-Based (gradient-free)
- **MC** ↔ Value-Based (full returns)
- **TD** ↔ Value-Based (bootstrapping)
- **Actor-Critic** = Policy-Based (Actor) + Value-Based (Critic)
