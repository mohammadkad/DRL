# DRL
Deep Reinforcement Learning

# Definition
Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.
In AI, particularly reinforcement learning (RL), a control task refers to a problem where an agent must make sequential decisions to achieve a goal. These are often modeled as Markov Decision Processes (MDPs) or Partially Observable MDPs (POMDPs).
The Markov Property implies that our agent needs only the current state to decide what action to take and not the history of all the states and actions they took before.

state, action, reward and next state
S0, A0, R1, S1
The agent’s goal is to maximize its *cumulative reward*, called the **expected return**.

### Observations/States Space
- State s: is a complete description of the state of the world (there is no hidden information). In a fully observed environment.
- Observation o: is a partial description of the state. In a partially observed environment.
