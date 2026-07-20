<!-- 1405-04-29 -->

### MuJoCo stands for Multi-Joint dynamics with Contact
- There are eleven MuJoCo environments (in roughly increasing complexity)

### ENVs:
- CartPoles
  - InvertedPendulum
  - InvertedDoublePendulum
- Arms
  - Reacher
  - Pusher
- 2D Runners
  - HalfCheetah
  - Hopper
  - Walker2d
- Swimmers
  - Swimmer
-Quadruped
  - Ant
- Humanoid Bipeds
  - Humanoid
  - HumanoidStandup

### Installation:
- pip install gymnasium[mujoco]


### Number of steps to see improvements:
- MK >  200,000 <!-- 1405-04-29 -->
- Default: parser.add_argument("--max_timesteps", default=3e6, type=int) # Max time steps to run environment, 3,000,000 (3 million)
- Default: parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
- parser.add_argument("--max_timesteps", default=200000, type=int)
- parser.add_argument("--eval_freq", default=300, type=int)
