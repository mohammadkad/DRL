<!--
Date : 1404-06-04, 1404-06-24
Mohammad Kadkhodaei Elyaderani
---
youtube : https://www.youtube.com/@CodeEmporium
-->

- Instead of using a Q-table, Deep Q-Learning uses a Neural Network that takes a state and approximates Q-values for each action based on that state.
- Finally, we have a couple of fully connected layers that output a Q-value for each possible action at that state.
- Deep Q-Learning uses a deep neural network to approximate the different Q-values for each possible action at a state (value-function estimation).
- The difference is that, during the training phase, instead of updating the Q-value of a state-action pair directly as we have done with Q-Learning, in Deep Q-Learning, we create a loss function that compares our Q-value prediction and the Q-target and uses gradient descent to update the weights of our Deep Q-Network to approximate our Q-values better.

The Deep Q-Learning training algorithm has two phases:
 - Sampling: we perform actions and store the observed experience tuples in a replay memory.
 - Training: Select a small batch of tuples randomly and learn from this batch using a gradient descent update step.

To help us stabilize the training, we implement three different solutions:
 - Experience Replay to make more efficient use of experiences.
 - Fixed Q-Target to stabilize the training.
 - Double Deep Q-Learning, to handle the problem of the overestimation of Q-values.

