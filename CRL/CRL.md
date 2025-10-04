<!-- 1404-07-10 -->
### Continual Reinforcement Learning (CRL)
- Continual Reinforcement Learning (CRL, a.k.a. Lifelong Reinforcement Learning, LRL) in  in **dynamic**, **non-stationary** environments
- Enabling agents to learn continuously, adapt to new tasks, and retain (keep) previously acquired knowledge.
- Exploring methods to enable RL agents to avoid **catastrophic forgetting** and effectively transfer knowledge
- Also referred to as lifelong learning or incremental learning
- The central challenge in CL lies in achieving a balance between **stability** and **plasticity** -> stability-plasticity dilemma
- The overarching goal is to build intelligent systems that are capable of learning and adapting throughout their lifetimes, rather than starting anew for each task.
- Current research in CL primarily focuses on two key aspects: addressing **catastrophic forgetting** and enabling **knowledge transfer**.

### Objectives: 
- minimizing the forgetting of knowledge from previously learned tasks
- leveraging prior experiences to learn new tasks more efficiently

### Definition:
- Catastrophic forgetting: degradation of model performance on previous tasks upon learning new ones
- Knowledge transfer: involves leveraging knowledge from previous tasks to facilitate learning on new tasks

### Strategies:
- Replay-based methods
- Regularization-based methods
- Parameter isolation methods

### CRL methods: What knowledge is stored and/or transferred?
- Policy-focused
  - Policy Reuse
  - Policy Decomposition :PNN, OWL, ...
  - Policy Merging
- Experience-focused
  - Direct Replay : 3RL, ...
  - Generative Replay
- Dynamic-focused
  - Direct Modeling
  -  Indirect Modeling
- Reward-focused



