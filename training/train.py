import numpy as np
import torch
from envs.fish_env import FishEnv
from models.ppo_marl import PPO
from memory import Memory

# Hyperparameters
NUM_AGENTS = 2
OBS_DIM = 8         # From fish_env.py
ACT_DIM = 2         # [thrust_x, thrust_y]
GLOBAL_STATE_DIM = NUM_AGENTS * 4  # 2 pos + 2 vel per agent
MAX_TIMESTEPS = 200
EPISODES = 1000
UPDATE_EVERY = 2000  # How many steps before PPO update
LOG_INTERVAL = 10

def main():
    env = FishEnv(num_agents=NUM_AGENTS)
    memory = Memory(NUM_AGENTS)
    ppo = PPO(
        num_agents=NUM_AGENTS,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        global_state_dim=GLOBAL_STATE_DIM
    )

    timestep_counter = 0
    rewards_per_episode = []

    for episode in range(EPISODES):
        local_obs, global_state = env.reset()
        ep_reward = np.zeros(NUM_AGENTS)

        for t in range(MAX_TIMESTEPS):
            actions = []
            for agent_id in range(NUM_AGENTS):
                action = ppo.select_action(local_obs[agent_id], memory, agent_id)
                actions.append(action)

            next_local_obs, next_global_state, rewards, dones, _ = env.step(actions)

            for agent_id in range(NUM_AGENTS):
                memory.rewards[agent_id].append(rewards[agent_id])
                memory.is_terminals[agent_id].append(dones[agent_id])

            local_obs = next_local_obs
            global_state = next_global_state
            ep_reward += np.array(rewards)
            timestep_counter += 1

            if all(dones):
                break

            # Update PPO if enough steps have passed
            if timestep_counter % UPDATE_EVERY == 0:
                flat_global_states = np.tile(global_state, (len(memory.rewards[0]), 1))
                ppo.update(memory, flat_global_states)
                memory.clear()
                timestep_counter = 0

        rewards_per_episode.append(ep_reward.mean())

        if episode % LOG_INTERVAL == 0:
            print(f"Episode {episode}, avg reward: {rewards_per_episode[-1]:.2f}")

    print("Training complete!")

if __name__ == "__main__":
    main()
