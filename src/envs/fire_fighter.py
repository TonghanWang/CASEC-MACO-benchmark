
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging
import random


class FireFighterEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
            self,
            n_agents=3,
            n_houses=10,
            episode_limit=5,
            max_fire_level=5,
            catch_prob=0.4,
            increase_prob_w=0.8,
            increase_prob_wo=0.4,
            lower_prob_w=0.6,
            win_reward=0,
            sight_range=1,
            obs_last_action=False,
            state_last_action=True,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents
        self.n_houses = n_houses
        self.episode_limit = episode_limit
        self.max_fire_level = max_fire_level
        self.catch_prob = catch_prob
        self.increase_prob_w = increase_prob_w
        self.increase_prob_wo = increase_prob_wo
        self.lower_prob_w = lower_prob_w
        self.win_reward = win_reward

        # Observations and state
        self.obs_last_action = obs_last_action
        self.state_last_action = state_last_action

        # Other
        self._seed = seed

        # Actions
        self.n_actions = self.n_houses

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Configuration initialization
        self.fire_levels = np.random.randint(low=0, high=self.max_fire_level, size=(self.n_houses))
        self.positions = np.random.randint(low=0, high=self.n_houses, size=(self.n_agents)).astype(np.int)

        self.eye = np.eye(self.n_houses)
        self.sight_range = sight_range

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        actions_numpy = actions.detach().cpu().numpy()
        self.positions = deepcopy(actions_numpy)

        # Fight
        for house_id in range(self.n_houses):
            o_fire_level = self.fire_levels[house_id]
            a_fire_level = o_fire_level

            num_fighters = (self.positions == int(house_id)).sum()
            if num_fighters >= 2:
                a_fire_level = 0
            elif num_fighters == 1:
                if self.fire_levels[house_id - 1] > 0 or self.fire_levels[(house_id + 1) % self.n_houses] > 0:
                    random_number = np.random.rand()
                    if random_number < self.lower_prob_w:
                        a_fire_level = max(0, a_fire_level - 1)
                else:
                    a_fire_level = max(0, a_fire_level - 1)

            self.fire_levels[house_id] = a_fire_level

        # Increase
        for house_id in range(self.n_houses):
            o_fire_level = self.fire_levels[house_id]
            a_fire_level = o_fire_level

            if self.fire_levels[house_id - 1] > 0 or self.fire_levels[(house_id + 1) % self.n_houses] > 0:
                if self.fire_levels[house_id] == 0:
                    random_number = np.random.rand()
                    if random_number < self.catch_prob:
                        a_fire_level = 1
                else:
                    random_number = np.random.rand()
                    if random_number < self.increase_prob_w:
                        a_fire_level = min(self.max_fire_level - 1, a_fire_level + 1)
            else:
                if self.fire_levels[house_id] > 0:
                    random_number = np.random.rand()
                    if random_number < self.increase_prob_wo:
                        a_fire_level = min(self.max_fire_level - 1, a_fire_level + 1)

            self.fire_levels[house_id] = a_fire_level

        # Reward
        reward = -np.sum(self.fire_levels)

        terminated = False
        info['battle_won'] = False

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if reward > -1e-4:
            terminated = True
            info['battle_won'] = True
            self.battles_won += 1
            reward += self.win_reward

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        self.obs = np.concatenate([self.fire_levels, self.fire_levels, self.fire_levels])
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        pos = self.positions[agent_id] + self.n_houses
        fire_levels = self.obs[pos - self.sight_range: pos + self.sight_range + 1]
        return np.concatenate([fire_levels, self.eye[agent_id]])

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.sight_range * 2 + 1 + self.n_houses

    def get_state(self):
        """Returns the global state."""
        return np.concatenate(self.get_obs())

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.n_agents * self.get_obs_size()

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.fire_levels = np.random.randint(low=0, high=self.max_fire_level, size=(self.n_houses))
        self.positions = np.random.randint(low=0, high=self.n_houses, size=(self.n_agents)).astype(np.int)

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
