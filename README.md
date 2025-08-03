# DRL
Deep Reinforcement Learning

# Definition
Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.
In AI, particularly reinforcement learning (RL), a control task refers to a problem where an agent must make sequential decisions to achieve a goal. These are often modeled as Markov Decision Processes (MDPs) or Partially Observable MDPs (POMDPs).
The Markov Property implies that our agent needs only the current state to decide what action to take and not the history of all the states and actions they took before.

- state, action, reward and next state
- S0, A0, R1, S1
The agent’s goal is to maximize its *cumulative reward*, called the **expected return**.

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

### Two main approaches for solving RL problems
The Policy π: the agent’s brain
This Policy is the function we want to learn, our goal is to find the optimal policy π*, the policy that maximizes expected return when the agent acts according to it. We find this π* through training.

- Policy-based methods (Directly) : we learn a policy function directly.
* Deterministic
* Stochastic
- Value-based methods (Indirectly)
