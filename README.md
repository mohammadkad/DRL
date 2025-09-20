# DRL
Deep Reinforcement Learning

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

- Policy-based methods (Directly) : we learn a policy function directly.
  - Deterministic
  - Stochastic
- Value-based methods (Indirectly) : we learn a value function that maps a state to the expected value of being at that state. The idea is that an optimal value function leads to an optimal policy.
  - state-value function
  - action-value function
- Actor-critic method, which is a combination of value-based and policy-based methods.

### TD vs MC
- Monte Carlo vs Temporal Difference Learning:
- Monte Carlo uses an entire episode of experience before learning
Temporal Difference (TD) uses only a step (St, At, Rt+1, St+1) to learn.
TD (Temporal difference) methods, Temporal Difference learning combines ideas from Monte Carlo methods (learning from experience) and Dynamic Programming (bootstrapping).:
 - TD learning, TD(0)/One-Step TD, TD(λ)/Forward View
 - SARSA (On-Policy)
 - SARSE
 - Q-Learning (Off-Policy)
 - Value-Based Methods: DQN (Deep Q-Network), Double DQN (DDQN)
 - Policy-Based Methods: REINFORCE, Actor-Critic
 - Advanced Actor-Critic Methods: A3C (Asynchronous Advantage Actor-Critic), A2C (Advantage Actor-Critic), DDPG (Deep Deterministic Policy Gradient), TD3 (Twin Delayed DDPG), SAC (Soft Actor-Critic)
MC (Monte Carlo)
 - 

### The difference between policy-based and policy-gradient methods
Policy-gradient methods, what we’re going to study in this unit, is a subclass of policy-based methods. In policy-based methods, the optimization is most of the time on-policy since for each update, we only use data (trajectories) collected by our most recent version of πθ.

The difference between these two methods lies on how we optimize the parameter θ:

- In policy-based methods, we search directly for the optimal policy. We can optimize the parameter
θ indirectly by maximizing the local approximation of the objective function with techniques like hill climbing, simulated annealing, or evolution strategies.

- In policy-gradient methods, because it is a subclass of the policy-based methods, we search directly for the optimal policy. But we optimize the parameter
θ directly by performing the gradient ascent on the performance of the objective function J(θ).
