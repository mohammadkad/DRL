- 1404-06-22
- Mohammad Kadkhodaei Elyaderani

- The agent’s decision-making process is called the policy π: given a state, a policy will output an action or a probability distribution over actions.
- Our goal is to find an optimal policy π* , aka., a policy that leads to the best expected cumulative reward.

- There are two main types of RL methods:
  - Policy-based methods: Train the policy directly to learn which action to take given a state
  - Value-based methods: Train a value function to learn which state is more valuable and use this value function to take the action that leads to it.
    - we have two types of value-based functions:
      - The state-value function
      - The action-value function


- This can be a computationally expensive process, and that’s where the Bellman equation comes in to help us.

- Monte Carlo vs Temporal Difference Learning:
  - Monte Carlo uses an entire episode of experience before learning
  - Temporal Difference (TD) uses only a step (St, At, Rt+1, St+1) to learn.
 
  What is Q-Learning?
  - Q-Learning is an off-policy value-based method that uses a TD approach to train its action-value function
  - TD approach: updates its action-value function at each step instead of at the end of the episode.
  - Q-Learning is the algorithm we use to train our Q-function (action-value function)
  - The Q comes from “the Quality” (the value) of that action at that state.
  - Q-function is encoded by a Q-table, a table where each cell corresponds to a state-action pair value. (Q-table as the memory or cheat sheet)
  - Trains a Q-function (an action-value function), which internally is a Q-table that contains all the state-action pair values.
  - When the training is done, we have an optimal Q-function, which means we have optimal Q-table.
  - And if we have an optimal Q-function, we have an optimal policy since we know the best action to take at each state.
  
