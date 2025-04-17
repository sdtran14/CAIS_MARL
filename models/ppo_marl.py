import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.policy(x)

class Critic(nn.Module):
    def __init__(self, global_state_dim):
        super(Critic, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(global_state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.value(state)

class PPO:
    def __init__(self, num_agents, obs_dim, act_dim, global_state_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.num_agents = num_agents
        self.gamma = gamma
        self.eps_clip = eps_clip

        # One actor per agent (decentralized)
        self.actors = [Actor(obs_dim, act_dim).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]

        # One centralized critic (shared for simplicity)
        self.critic = Critic(global_state_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Shared action std (assumes continuous action space)
        self.action_var = torch.full((act_dim,), 0.5).to(device)

    def select_action(self, obs, memory, agent_id):
        obs = torch.FloatTensor(obs).to(device)
        mean = self.actors[agent_id](obs)
        dist = MultivariateNormal(mean, torch.diag(self.action_var))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states[agent_id].append(obs)
        memory.actions[agent_id].append(action)
        memory.logprobs[agent_id].append(action_logprob)

        return action.detach().cpu().numpy()

    def update(self, memory, global_states, next_global_state):
        # Flatten all agent memory (assumes shared timesteps for now)
        rewards, dones = memory.rewards, memory.is_terminals
        old_states = memory.states
        old_actions = memory.actions
        old_logprobs = memory.logprobs


        # Convert to torch        
        global_states = torch.FloatTensor(global_states).to(device)
        next_global_state = torch.FloatTensor(next_global_state).to(device)

        # Estimate values from critic
        values = self.critic(global_states).squeeze()
       
        with torch.no_grad():
            next_value = self.critic(next_global_state.unsqueeze(0)).squeeze()
        agent_returns = self._compute_returns(rewards, dones, next_value)
        team_returns = torch.stack(agent_returns, dim=0).mean(dim=0)  

        # Critic update
        value_loss = nn.MSELoss()(values, team_returns)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # Actor update (per agent)
        for agent_id in range(self.num_agents):
            obs = torch.stack(old_states[agent_id]).to(device)
            actions = torch.stack(old_actions[agent_id]).to(device)
            logprobs_old = torch.stack(old_logprobs[agent_id]).to(device).detach()

            mean = self.actors[agent_id](obs)
            dist = MultivariateNormal(mean, torch.diag(self.action_var))
            logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()

            ratios = torch.exp(logprobs - logprobs_old)
            advantages = agent_returns[agent_id] - values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()

            self.actor_optimizers[agent_id].zero_grad()
            loss.backward()
            self.actor_optimizers[agent_id].step()

    def _compute_returns(self, rewards, dones, last_value, gamma=0.99):
        """
        rewards:    list of lists, rewards[i][t] for agent i at timestep t
        dones:      same shape, done flags
        last_value: scalar V(s_{T+1}) from the critic
        returns:    list of length num_agents, each a tensor of shape (T,)
        """
        all_returns = []
        for i in range(self.num_agents):
            R = last_value
            ret_i = []
            for step in reversed(range(len(rewards[i]))):
                # only that agent's reward here
                R = rewards[i][step] + gamma * R * (1 - dones[i][step])
                ret_i.insert(0, float(R))    # ensure Python float
            all_returns.append(torch.tensor(ret_i, dtype=torch.float32, device=device))

        return all_returns

    
    def save(self, directory="checkpoints", filename="ppo_ep0", keep_last=3):
        """
        Save the actors and critic for `filename` (e.g. 'ppo_ep650'),
        then prune to keep only the `keep_last` most recent episodes.
        """
        os.makedirs(directory, exist_ok=True)

        # 1) Save the new checkpoint files
        for agent_id, actor in enumerate(self.actors):
            torch.save(
                actor.state_dict(),
                os.path.join(directory, f"{filename}_actor_{agent_id}.pth")
            )
        torch.save(
            self.critic.state_dict(),
            os.path.join(directory, f"{filename}_critic.pth")
        )

        # 2) Prune old episodes
        #    Look for files named like "<prefix>_ep<NUM>_critic.pth"
        prefix_match = re.match(r"(.*)_ep\d+$", filename)
        prefix = prefix_match.group(1) if prefix_match else filename

        crit_pattern = os.path.join(directory, f"{prefix}_ep*_critic.pth")
        critic_files = glob.glob(crit_pattern)

        # Extract episode numbers
        eps = []
        for path in critic_files:
            name = os.path.basename(path)
            m = re.match(rf"{re.escape(prefix)}_ep(\d+)_critic\.pth$", name)
            if m:
                eps.append(int(m.group(1)))

        # Sort descending (newest first)
        eps_sorted = sorted(eps, reverse=True)

        # Delete all but the first `keep_last`
        for ep in eps_sorted[keep_last:]:
            # actor files
            for agent_id in range(len(self.actors)):
                actor_path = os.path.join(
                    directory, f"{prefix}_ep{ep}_actor_{agent_id}.pth"
                )
                if os.path.exists(actor_path):
                    print(f"Removing old actor checkpoint: {actor_path}")
                    os.remove(actor_path)

            # critic file
            critic_path = os.path.join(directory, f"{prefix}_ep{ep}_critic.pth")
            if os.path.exists(critic_path):
                print(f"Removing old critic checkpoint: {critic_path}")
                os.remove(critic_path)
    
    def load(self, directory="checkpoints", filename="ppo_checkpoint"):
        for agent_id, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor_{agent_id}.pth")))

        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic.pth")))