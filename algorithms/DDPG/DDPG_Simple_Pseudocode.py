# 1404-05-26,
# Mohammad Kadkhodaei
# ---

for episode in episodes:
    state = env.reset()
    while not done:
        # 1. Select action with exploration noise
        action = actor(state) + noise()
        
        # 2. Execute action, store experience
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        
        # 3. Sample batch and train
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            
            # Update Critic (Q-function)
            target_actions = actor_target(next_state)
            target_Q = reward + γ * critic_target(next_state, target_actions)
            critic_loss = MSE(critic(state, action), target_Q)
            
            # Update Actor (policy)
            actor_loss = -mean(critic(state, actor(state)))
            
            # Soft update target networks
            soft_update(actor_target, actor, τ)
            soft_update(critic_target, critic, τ)
        
        state = next_state
