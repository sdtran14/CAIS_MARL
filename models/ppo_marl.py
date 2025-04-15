import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

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

    def update(self, memory, global_states):
        # Flatten all agent memory (assumes shared timesteps for now)
        rewards, dones = memory.rewards, memory.is_terminals
        old_states = memory.states
        old_actions = memory.actions
        old_logprobs = memory.logprobs

        # Convert to torch
        global_states = torch.FloatTensor(global_states).to(device)

        # Estimate values from critic
        values = self.critic(global_states).squeeze()
        returns = self._compute_returns(rewards, dones, values.detach())

        # Critic update
        value_loss = nn.MSELoss()(values, returns)
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
            advantages = returns - values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()

            self.actor_optimizers[agent_id].zero_grad()
            loss.backward()
            self.actor_optimizers[agent_id].step()

    def _compute_returns(self, rewards, dones, last_value, gamma=0.99):
        returns = []
        R = last_value
        for step in reversed(range(len(rewards[0]))):
            R = sum([rewards[i][step] for i in range(self.num_agents)]) / self.num_agents + gamma * R * (1 - dones[0][step])
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(device)
