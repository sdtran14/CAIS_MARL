import numpy as np
import torch
from envs.fish_env import FishEnv
from models.ppo_marl import PPO
from memory import Memory
import glob
import matplotlib.pyplot as plt


# Hyperparameters
NUM_AGENTS = 2
OBS_DIM = 8         # From fish_env.py
ACT_DIM = 2         # [thrust_x, thrust_y]
GLOBAL_STATE_DIM = NUM_AGENTS * 4  # 2 pos + 2 vel per agent
MAX_TIMESTEPS = 200
EPISODES = 2001
UPDATE_EVERY = 2000  # How many steps before PPO update
LOG_INTERVAL = 10
SAVE_INTERVAL = 50

RENDER = False

plt.figure(figsize=(8, 5))

def main():
    render = None
    if RENDER:
        render = "human"
    env = FishEnv(num_agents=NUM_AGENTS, render_mode=render)
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
        first_reset = (timestep_counter == 0)

        local_obs, global_state = env.reset()
        ep_reward = np.zeros(NUM_AGENTS)
        if first_reset:
            memory.global_states.append(global_state)   # only once per window
            
        
        for t in range(MAX_TIMESTEPS):
            actions = []
            for agent_id in range(NUM_AGENTS):
                action = ppo.select_action(local_obs[agent_id], memory, agent_id)
                actions.append(action)

            next_local_obs, next_global_state, rewards, dones, _ = env.step(actions)
            
            if(RENDER):
                env.render()

            for agent_id in range(NUM_AGENTS):
                memory.rewards[agent_id].append(rewards[agent_id])
                memory.is_terminals[agent_id].append(dones[agent_id])
            memory.global_states.append(next_global_state)

            local_obs = next_local_obs
            global_state = next_global_state
            ep_reward += np.array(rewards)
            timestep_counter += 1

            

            # Update PPO if enough steps have passed
            if timestep_counter % UPDATE_EVERY == 0:
                #   all but last for critic input…
                prev_states = np.stack(memory.global_states[:-1], axis=0)

                #   the very last one for bootstrapping
                boot_state = memory.global_states[-1]
                ppo.update(
                    memory,
                    prev_states,      # shape (T, D)
                    boot_state        # shape (D,)
                )
                memory.clear()
                timestep_counter = 0
                break
            
            if all(dones):
                break

        rewards_per_episode.append(ep_reward.mean())

        if episode % LOG_INTERVAL == 0:
            avg = rewards_per_episode[-1]
            print(f"Episode {episode}, avg reward: {avg:.2f}")

            # Prepare data
            episodes = list(range(1, len(rewards_per_episode) + 1))

            # Clear and redraw
            plt.clf()
            plt.plot(episodes, rewards_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.title("Training Progress")
            plt.grid(True)

            # Save (overwrites the same file each time)
            plt.savefig("training_progress.png")

        if episode % SAVE_INTERVAL == 0:
            print(f"Saving model at episode {episode}")
            ppo.save(directory="checkpoints", filename=f"ppo_ep{episode}")

    print("Training complete!")

if __name__ == "__main__":
    main()
