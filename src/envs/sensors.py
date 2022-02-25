from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from smac.env.multiagentenv import MultiAgentEnv


class SensorEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
            self,
            n_preys=3,
            episode_limit=10,
            array_height=3,
            array_width=5,
            catch_reward=2,
            scan_cost=1,
            obs_last_action=False,
            state_last_action=True,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = array_height * array_width
        self.n_preys = n_preys
        self.episode_limit = episode_limit
        self.array_width = array_width
        self.array_height = array_height
        self.map_height = 2 * array_height - 1
        self.map_width = 2 * array_width - 1
        self.catch_reward = catch_reward
        self.scan_cost = scan_cost

        # Observations and state
        self.obs_last_action = obs_last_action
        self.state_last_action = state_last_action

        # Other
        self._seed = seed

        # Actions
        self.n_actions = 9

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.neighbors = [(1, 1), (1, -1), (1, 0), (-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1)]

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Configuration initialization
        self.prey_positions = np.zeros((self.map_height, self.map_width))
        self.occ = np.zeros((self.map_height, self.map_width)).astype(int)
        self.occ[0:self.map_height:2, 0:self.map_width:2] = 1
        self.prey_positions_idx = np.zeros((self.n_preys, 2)).astype(int)
        self.agent_positions_idx = np.zeros((self.n_agents, 2)).astype(int)

        for prey_i in range(self.n_preys):
            prey_h = np.random.randint(low=0, high=self.map_height)
            prey_w = np.random.randint(low=0, high=self.map_width)

            while self.occ[prey_h, prey_w]:
                prey_h = np.random.randint(low=0, high=self.map_height)
                prey_w = np.random.randint(low=0, high=self.map_width)

            self.prey_positions[prey_h, prey_w] = prey_i + 1
            self.occ[prey_h, prey_w] = 1
            self.prey_positions_idx[prey_i, 0] = prey_h
            self.prey_positions_idx[prey_i, 1] = prey_w

        for agent_y in range(self.array_height):
            for agent_x in range(self.array_width):
                self.agent_positions_idx[agent_y * array_width + agent_x, 0] = agent_y * 2
                self.agent_positions_idx[agent_y * array_width + agent_x, 1] = agent_x * 2

        self.obs_size = 11
        self.avail_actions = []
        for agent_i in range(self.n_agents):
            agent_y = self.agent_positions_idx[agent_i, 0]
            agent_x = self.agent_positions_idx[agent_i, 1]
            _avail_actions = [] # size 9

            for delta in self.neighbors:
                if 0 <= agent_x + delta[0] < self.map_width and 0 <= agent_y + delta[1] < self.map_height:
                    _avail_actions.append(1)
                else:
                    _avail_actions.append(0)
            _avail_actions.append(1)
            self.avail_actions.append(_avail_actions)

        self._episode_scaned = 0

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        reward = 0
        terminated = False
        info['battle_won'] = False

        prey_scaned = np.array([0 for _ in range(self.n_preys)])

        # map = np.zeros((self.map_height, self.map_width))
        # map[0:self.map_height:2, 0:self.map_width:2] = 1
        # for prey_i in range(self.n_preys):
        #     map[self.prey_positions_idx[prey_i, 0], self.prey_positions_idx[prey_i, 1]] = 2

        for agent_i, action in enumerate(actions):
            if action < 8:
                reward -= self.scan_cost

                agent_y = self.agent_positions_idx[agent_i, 0]
                agent_x = self.agent_positions_idx[agent_i, 1]

                scan_x = agent_x + self.neighbors[action][0]
                scan_y = agent_y + self.neighbors[action][1]

                # map[scan_y, scan_x] += 10

                if 0 <= scan_y < self.map_height and 0 <= scan_x < self.map_width:
                    for prey_i in range(self.n_preys):
                        if scan_x == self.prey_positions_idx[prey_i, 1] and scan_y == self.prey_positions_idx[prey_i, 0]:
                            prey_scaned[prey_i] += 1

        # print(map)

        for _prey_scaned in prey_scaned:
            if _prey_scaned >= 2:
                reward += self.catch_reward
                self._episode_scaned += 1
            # elif _prey_scaned == 3:
            #     reward += self.catch_reward * 1.5
            #     self._episode_scaned += 1
            # elif _prey_scaned == 4:
            #     reward += self.catch_reward * 2
            #     self._episode_scaned += 1

        info['scaned'] = self._episode_scaned

        # Prey move
        for prey_i in range(self.n_preys):
            h, w = self.prey_positions_idx[prey_i, 0], self.prey_positions_idx[prey_i, 1]

            delta_h = np.random.randint(low=-2, high=3)
            delta_w = np.random.randint(low=-2, high=3)

            target_w = min(max(w + delta_w, 0), self.map_width - 1)
            target_h = min(max(h + delta_h, 0), self.map_height - 1)

            while self.occ[target_h, target_w]:
                delta_h = np.random.randint(low=-2, high=3)
                delta_w = np.random.randint(low=-2, high=3)

                target_w = min(max(w + delta_w, 0), self.map_width - 1)
                target_h = min(max(h + delta_h, 0), self.map_height - 1)

            self.occ[h, w] = 0
            self.occ[target_h, target_w] = 1
            self.prey_positions_idx[prey_i, 0] = target_h
            self.prey_positions_idx[prey_i, 1] = target_w

        if self._episode_steps >= self.episode_limit:
            terminated = True
            self._episode_count += 1
            self.battles_game += 1

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        agent_h = self.agent_positions_idx[agent_id, 0]
        agent_w = self.agent_positions_idx[agent_id, 1]
        occ_temp = np.pad(self.occ, ((1,1),(1,1)), 'constant', constant_values=(-1,-1))
        agent_h = agent_h + 1
        agent_w = agent_w + 1
        obs = occ_temp[agent_h - 1: agent_h + 2, agent_w - 1: agent_w + 2]
        obs[1, 1] = 0
        return np.concatenate([obs.flatten(), self.agent_positions_idx[agent_id]])

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_size

    def get_state(self):
        """Returns the global state."""
        return np.concatenate(self.get_obs())

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.n_agents * self.get_obs_size()

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return self.avail_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.prey_positions = np.zeros((self.map_height, self.map_width))
        self.occ = np.zeros((self.map_height, self.map_width)).astype(int)
        self.occ[0:self.map_height:2, 0:self.map_width:2] = 1
        self.prey_positions_idx = np.zeros((self.n_preys, 2)).astype(int)
        self.agent_positions_idx = np.zeros((self.n_agents, 2)).astype(int)

        for prey_i in range(self.n_preys):
            prey_h = np.random.randint(low=0, high=self.map_height)
            prey_w = np.random.randint(low=0, high=self.map_width)

            while self.occ[prey_h, prey_w]:
                prey_h = np.random.randint(low=0, high=self.map_height)
                prey_w = np.random.randint(low=0, high=self.map_width)

            self.prey_positions[prey_h, prey_w] = prey_i + 1
            self.occ[prey_h, prey_w] = 1
            self.prey_positions_idx[prey_i, 0] = prey_h
            self.prey_positions_idx[prey_i, 1] = prey_w

        for agent_y in range(self.array_height):
            for agent_x in range(self.array_width):
                self.agent_positions_idx[agent_y * self.array_width + agent_x, 0] = agent_y * 2
                self.agent_positions_idx[agent_y * self.array_width + agent_x, 1] = agent_x * 2

        self._episode_scaned = 0

        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game
        }
        return stats
