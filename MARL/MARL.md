<!-- 1404-07-09 -->
### Basics of the fascinating topic of multi-agents reinforcement learning (MARL):
- goal is to create agents that can interact with other humans and other agents.
- When our agent was alone in its environment: it was not **cooperating** or **collaborating** with other agents.
- we have multiple agents that share and interact in a common environment.
- we have multiple agents interacting in the environment and with the other agents.

### Different types of multi-agent environments:
- Cooperative environments: where your agents need to maximize the common benefits.
- Competitive/Adversarial environments: in this case, your agent wants to maximize its benefits by minimizing the opponent’s.
- Mixed of both adversarial and cooperative

### Designing Multi-Agents systems:
- Decentralized system : In decentralized learning, each agent is trained independently from the others.
  - each vacuum learns to clean as many places as it can without caring about what other vacuums (agents) are doing.
  - The benefit is that since no information is shared between agents, these vacuums can be designed and trained like we train single agents.
  - The idea here is that our training agent will consider other agents as part of the environment dynamics. Not as agents.
  - we treat all agents independently without considering the existence of the other agents.
  - However, the big drawback of this technique is that it will make the environment non-stationary since the underlying Markov decision process changes over time as other agents are also interacting in the environment.
- Centralized approach
  - we have a high-level process that collects agents’ experiences: the experience buffer
  - And we’ll use these experiences to learn a common policy.
  - A single policy is learned from all the agents.
  - The reward is global.

### divided into three main types:
- centralized training and execution (CTE)
- centralized training for decentralized execution (CTDE)*
- Decentralized training and execution (DTE)
