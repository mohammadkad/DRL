<!--
1404-05-26
1404-06-26
Mohammad Kadkhodaei
-->

- MDP
- The general framework of MDPs allows us to model virtually any complex sequential decision-making problem under uncertainty in a way that RL agents can interact with and learn to solve solely through experience.

First-Order Markov Property in RL If the environment is not inherently Markovian (e.g., partial observability), we often use state representations (like LSTM, attention, or frame stacking) to approximate it.
Higher-Order Markov Models? DRL typically avoids them by instead augmenting the state (e.g., stacking frames in Atari games) to restore the Markov property.
When is the Markov Assumption Violated? In Partially Observable MDPs (POMDPs), the agent does not have full state information (e.g., a robot with noisy sensors).
Solutions:
 - Frame stacking (e.g., last 4 frames in Atari DQN).
 - Recurrent policies (e.g., DRQN with LSTMs).
 - Memory-based architectures (Transformers, NTM).
