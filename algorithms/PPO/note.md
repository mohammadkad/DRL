<!-- 1404-07-07 -->
### proximal policy optimization
- Think of PPO as an algorithm with the same underlying architecture as A2C
- The critical innovation in PPO is a surrogate objective function that allows an on-policy algorithm to perform multiple gradient steps on the same mini-batch of experiences.
- A2C, being an on-policy method, cannot reuse experiences for the optimization steps.
- In general, on-policy methods need to discard experience samples immediately after stepping the optimizer.
- However, PPO introduces a clipped objective function that prevents the policy from getting too different after an optimization step
