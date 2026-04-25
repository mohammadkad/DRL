# Why Actor-Critic Methods? (Bias vs. Variance Trade-off)

## Overview

In Deep Reinforcement Learning, there are two primary approaches:

- **Value-based (e.g., DQN):** Learn a value function \( Q(s,a) \) or \( V(s) \), then derive a policy implicitly (e.g., \( \epsilon \)-greedy). **Low variance, high bias** (when using function approximation).
- **Policy-based (e.g., REINFORCE):** Directly learn a policy \( \pi(a|s) \) by gradient ascent on expected return. **High variance, low bias** (in principle).

**Actor-critic methods** emerge as a way to **balance bias and variance**, getting the best of both worlds.

---

## Why do we need Actor-Critic?

### 1. The problem with pure Policy Gradients (REINFORCE)

The standard policy gradient is:

\[
\nabla J(\theta) = \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \, G_t \right]
\]

where \( G_t = \sum_{k=t}^T \gamma^{k-t} r_k \) is the **actual return from time t**.

- **High variance**: \( G_t \) can vary wildly depending on random actions and environment stochasticity. The gradient estimate has huge noise, leading to slow, unstable learning.
- **Why?** Because \( G_t \) is an **unbiased estimate** of the value of being in state \( s_t \), but it's a Monte Carlo sample — it includes all future randomness.

### 2. Reducing variance with a baseline

A standard trick: subtract a baseline \( b(s_t) \) from \( G_t \):

\[
\nabla J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \, (G_t - b(s_t)) \right]
\]

This **reduces variance** without changing the expected gradient (if \( b(s_t) \) doesn't depend on action).

Choosing \( b(s_t) = V^\pi(s_t) \) (the true state-value function) minimizes variance optimally. But we don't know \( V^\pi(s_t) \) — we have to **estimate it**.

---

## Enter Actor-Critic: The explicit bias-variance trade-off

In actor-critic, we maintain **two networks**:

- **Actor** \( \pi_\theta(a|s) \): the policy we are learning.
- **Critic** \( V_\phi(s) \) or \( Q_\phi(s,a) \): estimates the value function.

Instead of \( G_t \), we use an **estimate** of the advantage:

\[
\hat{A}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \quad \text{(TD error)}
\]

or more generally the \( Q \) value minus the baseline:

\[
\hat{A}_t = Q_\phi(s_t, a_t) - V_\phi(s_t)
\]

The policy gradient becomes:

\[
\nabla J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a|s) \, \hat{A}(s,a) \right]
\]

---

## Bias vs. Variance in Actor-Critic

| Method | Gradient Estimate | Variance | Bias | Why? |
|--------|----------------|----------|------|------|
| **REINFORCE** | \( G_t \) (actual return) | Very high | Zero | Unbiased but noisy due to environment randomness. |
| **Actor-Critic** | \( \hat{A}_t \) (bootstrapped) | Lower | Non-zero | Uses learned \( V_\phi \), which may be inaccurate. |

### Variance reduction:
- Bootstrapping from \( V_\phi(s_{t+1}) \) cuts off long tails of randomness → **much lower variance** than Monte Carlo.
- Learn faster, more stable updates.

### Bias introduced:
- If \( V_\phi \) is wrong, the advantage estimate is **biased** — the gradient direction is no longer exactly correct.
- This is called **approximation bias**.

---

## The trade-off visualized

- **Pure value-based (DQN):** Low variance, high bias (due to max-operator and function approximation errors).
- **Pure policy-based (REINFORCE):** High variance, zero bias (but impractical for complex/deep networks).
- **Actor-Critic:** Tuneable middle ground. By improving the critic, we reduce bias toward zero; by using bootstrapping, we keep variance low.

---

## How modern methods manage this trade-off

- **Advantage Actor-Critic (A2C):** Uses \( n \)-step returns (\( G_t^{(n)} \)) — interpolates between Monte Carlo (low bias, high variance) and 1-step TD (higher bias, lower variance).
- **GAE (Generalized Advantage Estimation):** Exponentially weights \( n \)-step returns — a smooth, tunable bias-variance knob.
- **PPO/TRPO:** Constraints on policy update to avoid destructive bias from a bad critic.
- **DDPG/SAC:** Use target networks (delayed updates) to reduce bias in the critic.

---

## Summary

We need actor-critic methods because:

1. Pure policy gradients are **too high variance** for deep neural networks to learn efficiently.
2. Bootstrapping with a learned value function (the critic) **drastically reduces variance**.
3. The price is **potential bias** — but we can manage this with careful design (GAE, target networks, etc.).
4. Actor-critic provides a **flexible framework** to navigate the bias-variance frontier, which is the central challenge in modern deep RL.

That balance is why virtually all state-of-the-art DRL algorithms (PPO, SAC, TD3, A2C) are actor-critic methods.
