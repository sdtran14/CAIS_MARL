class Memory:
    def __init__(self, num_agents):
        self.states = [[] for _ in range(num_agents)]       # Local obs per agent
        self.actions = [[] for _ in range(num_agents)]      # Actions taken
        self.logprobs = [[] for _ in range(num_agents)]     # Log probs of actions
        self.rewards = [[] for _ in range(num_agents)]      # Rewards received
        self.is_terminals = [[] for _ in range(num_agents)] # Done flags
        self.global_states = []

    def clear(self):
        for agent_data in [
            self.states,
            self.actions,
            self.logprobs,
            self.rewards,
            self.is_terminals
        ]:
            for entry in agent_data:
                entry.clear()
        self.global_states.clear()