<!--
1404-05-26
1404-06-26
Mohammad Kadkhodaei
-->

- MDP
- The general framework of MDPs allows us to model virtually any complex sequential decision-making problem under uncertainty in a way that RL agents can interact with and learn to solve solely through experience.

- The Markov property:
  - The probability of the next state, given the current state and action, is independent of the history of interactions.

First-Order Markov Property in RL If the environment is not inherently Markovian (e.g., partial observability), we often use state representations (like LSTM, attention, or frame stacking) to approximate it.
Higher-Order Markov Models? DRL typically avoids them by instead augmenting the state (e.g., stacking frames in Atari games) to restore the Markov property.
When is the Markov Assumption Violated? In Partially Observable MDPs (POMDPs), the agent does not have full state information (e.g., a robot with noisy sensors).
Solutions:
 - Frame stacking (e.g., last 4 frames in Atari DQN).
 - Recurrent policies (e.g., DRQN with LSTMs).
 - Memory-based architectures (Transformers, NTM).

Extensions to MDPs:
• Partially observable Markov decision process (POMDP): When the agent cannot fully observe the environment state
• Factored Markov decision process (FMDP): Allows the representation of the transition and reward function more compactly so that we can represent large MDPs
• Continuous [Time|Action|State] Markov decision process: When either time, action, state or any combination of them are continuous
• Relational Markov decision process (RMDP): Allows the combination of probabilistic and relational knowledge
• Semi-Markov decision process (SMDP): Allows the inclusion of abstract actions that can take multiple time steps to complete
• Multi-agent Markov decision process (MMDP): Allows the inclusion of multiple agents in the same environment
• Decentralized Markov decision process (Dec-MDP): Allows for multiple agents to collaborate and maximize a common reward
