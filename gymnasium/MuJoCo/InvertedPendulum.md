<!-- 1405-04-30 -->

# InvertedPendulum-v5 Reward System

## Reward Structure
The `InvertedPendulum-v5` environment uses a simple reward function:

- **Reward**: `+1` for every timestep the pole remains upright
- **Upright Definition**: Pole angle is within ±0.2 radians (≈ ±11.5°) from vertical

## Design Philosophy
The reward follows a **goal-oriented** approach:
- Rewards success, not specific actions
- Agent must learn optimal balancing strategy independently
- Maximizing total reward equals maximizing survival time

## Episode Details

| Feature | Specification |
|---------|---------------|
| **Termination Conditions** | Pole angle exceeds ±0.2 radians **OR** episode reaches 1000 timesteps |
| **Maximum Possible Reward** | 1000 (perfect balance for entire episode) |
| **Reward Access** | Available in `info` dictionary under key `"reward_survive"` |

## Key Takeaways
1. Constant positive reward encourages survival
2. Agent must discover balancing policy through exploration
3. Total reward directly correlates with balancing duration
