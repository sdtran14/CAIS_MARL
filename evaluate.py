#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Make sure your project root is on PYTHONPATH so imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.fish_env import FishEnv
from models.ppo_marl import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_dir, model_name, episodes, render):
    # Hyperparams must match training
    NUM_AGENTS       = 2
    OBS_DIM          = 8
    ACT_DIM          = 2
    GLOBAL_STATE_DIM = NUM_AGENTS * 4

    # 1) Load environment
    env = FishEnv(num_agents=NUM_AGENTS,
                  render_mode="human" if render else None)

    # 2) Instantiate PPO and load
    ppo = PPO(NUM_AGENTS, OBS_DIM, ACT_DIM, GLOBAL_STATE_DIM)
    ppo.load(directory=model_dir, filename=model_name)
    for actor in ppo.actors:
        actor.eval()  # put in eval mode

    all_episode_rewards = []

    # 3) Run episodes
    for ep in range(episodes):
        local_obs, _ = env.reset()
        episode_reward = np.zeros(NUM_AGENTS)
        dones = [False] * NUM_AGENTS

        while not all(dones):
            actions = []
            for agent_id, obs in enumerate(local_obs):
                obs_t = torch.FloatTensor(obs).to(device)
                with torch.no_grad():
                    mean = ppo.actors[agent_id](obs_t)
                actions.append(mean.cpu().numpy())

            local_obs, _, rewards, dones, _ = env.step(actions)
            episode_reward += np.array(rewards)

            if render:
                env.render()

        all_episode_rewards.append(episode_reward)
        print(f"Episode {ep+1} rewards: {episode_reward}")

    # 4) Summary
    avg_rewards = np.mean(all_episode_rewards, axis=0)
    print(f"\nAverage rewards over {episodes} episodes: {avg_rewards}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained multi-agent PPO on FishEnv"
    )
    parser.add_argument(
        "--model-dir", type=str, default="checkpoints",
        help="Directory where actor_*.pth and critic.pth live"
    )
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="Base filename (e.g. 'ppo_ep50')"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="How many episodes to run"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="If set, render each step in a window (requires pygame)"
    )
    args = parser.parse_args()

    evaluate(
        model_dir  = args.model_dir,
        model_name = args.model_name,
        episodes   = args.episodes,
        render     = args.render
    )
