# Side-by-Side Code Comparison

# ---
# Old v0.21 Code
import gym

# Environment creation and seeding
env = gym.make("LunarLander-v3", options={})
env.seed(123)
observation = env.reset()

# Training loop
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render(mode="human")

env.close()
# ---

# ---
# New v0.26+ Code (Including v1.0.0)
import gymnasium as gym  # Note: 'gymnasium' not 'gym'

# Environment creation with render mode specified upfront
env = gym.make("LunarLander-v3", render_mode="human")

# Reset with seed parameter
observation, info = env.reset(seed=123, options={})

# Training loop with terminated/truncated distinction
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Episode ends if either terminated OR truncated
    done = terminated or truncated

env.close()
# ---
