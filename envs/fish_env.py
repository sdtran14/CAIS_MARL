# python3.10# -*- coding: utf-8 -*-

# use Gym 0.21.0
# pip install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac

# + Fix typo with opencv 3.?? in init.py
# + pip install pyglet==1.5.27 PyOpenGL PyOpenGL-accelerate
# + modify timelimit reset (without seed)
# ------------------------------------------------------------------------
# required pacckages: NumPy, SciPy, openAI gym
# written in the framwork of gym
# ------------------------------------------------------------------------
import gymnasium as gym
# from gym import spaces
# from gym.utils import seeding
from gymnasium import spaces
import numpy as np
from os import path
from scipy import integrate
#from colorama import Fore, Back, Style

class FishEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, num_agents=2, world_size=10.0, render_mode=None):
        super(FishEnv, self).__init__()
        
        # Render settings
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 512


        self.num_agents = num_agents
        self.world_size = world_size
        self.max_steps = 200
        self.timestep = 0

        # Action: thrust in x and y
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for _ in range(num_agents)]

        # Observation: x, y, vx, vy, flow_x, flow_y, goal_dx, goal_dy
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32) for _ in range(num_agents)]

        self.reset()

    def reset(self):
        self.timestep = 0
        self.agent_states = []
        self.goal = np.array([self.world_size, self.world_size])

        for _ in range(self.num_agents):
            x, y = np.random.uniform(0, self.world_size, size=2)
            vx, vy = np.zeros(2)
            self.agent_states.append({'pos': np.array([x, y]), 'vel': np.array([vx, vy])})

        local_obs = self._get_local_obs()
        global_state = self._get_global_state()
        return local_obs, global_state

    def step(self, actions):
        self.timestep += 1
        rewards = []
        dones = []

        for i, action in enumerate(actions):
            agent = self.agent_states[i]

            # Sample flow at agent's location (placeholder)
            flow = self._get_flow_at(agent['pos'])

            # Apply action as thrust + flow
            total_force = action + flow
            agent['vel'] += total_force * 0.1
            agent['pos'] += agent['vel'] * 0.1

            # Bound within world
            agent['pos'] = np.clip(agent['pos'], 0, self.world_size)

            # Reward: negative distance to goal
            dist_to_goal = np.linalg.norm(agent['pos'] - self.goal)
            rewards.append(-dist_to_goal)

            # Done if close to goal or max steps
            done = dist_to_goal < 1.0 or self.timestep >= self.max_steps
            dones.append(done)

        local_obs = self._get_local_obs()
        global_state = self._get_global_state()
        return local_obs, global_state, rewards, dones, {}

    def _get_flow_at(self, pos):
        # Placeholder for vector field lookup
        return np.array([np.sin(pos[0]), np.cos(pos[1])])

    def _get_local_obs(self):
        obs = []
        for agent in self.agent_states:
            pos = agent['pos']
            vel = agent['vel']
            flow = self._get_flow_at(pos)
            goal_vec = self.goal - pos
            obs.append(np.concatenate([pos, vel, flow, goal_vec]))
        return obs

    def _get_global_state(self):
        # Flat array of all agents' positions and velocities
        state = []
        for agent in self.agent_states:
            state.extend(agent['pos'])
            state.extend(agent['vel'])
        return np.array(state)
    
    def render(self):
        import pygame
        import numpy as np

        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Fish Environment")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background

        scale = self.window_size / self.world_size
    
        # Draw goal
        goal_pos = self.goal * scale
        pygame.draw.circle(canvas, (0, 255, 0), goal_pos.astype(int), 10)

        # Draw agents
        for agent in self.agent_states:
            pos = agent['pos'] * scale
            pygame.draw.circle(canvas, (0, 0, 255), pos.astype(int), 5)
        
        # Sample the vector field on a grid and draw arrows
        grid_spacing = self.world_size / 10  # adjust resolution here
        arrow_color = (100, 100, 100)  # light grey
        arrow_scale = 0.5  # how long the arrows are drawn (scaled)
        
        for i in range(10):
            for j in range(10):
                x = i * grid_spacing + grid_spacing / 2
                y = j * grid_spacing + grid_spacing / 2
                flow = self._get_flow_at(np.array([x, y]))
                
                start_px = np.array([x, y]) * scale
                end_px = start_px + flow * scale * arrow_scale

                pygame.draw.line(canvas, arrow_color, start_px.astype(int), end_px.astype(int), width=1)
                pygame.draw.circle(canvas, arrow_color, start_px.astype(int), 1)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
        
            

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
