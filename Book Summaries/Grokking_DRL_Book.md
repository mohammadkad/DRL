<!--
Grokking Deep Reinforcement Learning book Summary
Mohammad Kadkhodaei
1404-06-23
-->

- The distinct property of DRL programs is learning through trial and error from feedback that’s simultaneously sequential, evaluative, and sampled by leveraging powerful non-linear function approximation
- Reinforcement learning (RL) is the task of learning through trial and error. In this type of task, no human labels data, and no human collects or explicitly designs the collection of data.
- Deep learning (DL), involves using multi-layered non-linear function approximation, typically neural networks.


<!-- Chapter 3, 1404-07-01 -->
### Chapter 3:
- MDPs are the motors moving RL environments.
- Two fundamental algorithms for solving MDPs under a technique called dynamic programming: value iteration (VI) and policy iteration (PI).
- VI and PI are the foundations from which virtually every other RL (and DRL) algorithm originates.
- What we need is a plan for every possible state, a universal plan, a policy. (Policy), Policies are universal plans.
- Policies cover all possible states.
- Policies can be stochastic or deterministic. The policy can return action-probability distributions or single actions for a given state (or observation)
- Functions:
  - V : state-value function V, V-function, value function, V(s)
  - Q : The action-value function Q, Q-function, Q(s, a)
  - A : The action-advantage function A, advantage function, A-function, a(s, a) = q(s , a) - v(s)

<!-- Chapter 4, 1404-07-02 -->
### Chapter 4:
- exploration-exploitation dilemma : Every decision you make is a trade-off between information gathering (exploration) and information exploitation.
- challenges of learning from evaluative feedback
- Remember, in DRL, agents learn from feedback that’s simultaneously sequential (as opposed to one shot), evaluative (as opposed to supervised), and sampled (as opposed to exhaustive)
- Greedy
- Random
- Epsilon-greedy: Almost always greedy and sometimes random
- Decaying epsilon-greedy: First maximize exploration, then exploitation
- Optimistic initialization strategy
- softmax strategy
- upper confidence bound (UCB) strategy
- Thompson sampling: Balancing reward and risk

<!-- 1404-07-05 -->
### Chapter 5, Evaluating agents’ behaviors:
#### Prediction Problem (Policy Evaluation)
 - Goal: Estimate how good a given policy is
 - Input: Policy π
 - Output: Value function Vπ(s) or Qπ(s,a)
 - Key Challenge: Estimating future returns from experience
#### Control Problem (Policy Optimization)
 - Goal: Find the optimal policy π*
 - Input: Environment interaction
 - Output: Optimal policy π* and optimal value function
 - Key Challenge: Balancing exploration vs exploitation
- Learning to estimate the value of policies

