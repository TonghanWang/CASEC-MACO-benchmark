

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


class AlohaEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
            self,
            n_agents=10,
            episode_limit=20,
            max_list_length=5,
            obs_last_action=False,
            state_last_action=True,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents

        # Observations and state
        self.obs_last_action = obs_last_action
        self.state_last_action = state_last_action

        # Rewards args
        self.max_list_length = max_list_length

        # Other
        self._seed = seed

        # Actions
        self.n_actions = 2
        self.reward_scale = 10.

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.episode_limit = episode_limit

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Initialize backlogs
        self.backlogs = np.ones(self.n_agents)
        self.transmitted = 0
        self.adj = np.array([[0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                             [1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                             [0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 1., 0., 1., 0., 0., 0., 1., 0.],
                             [0., 0., 0., 1., 0., 1., 0., 0., 0., 1.],
                             [1., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                             [0., 0., 1., 0., 0., 0., 1., 0., 1., 0.],
                             [0., 0., 0., 1., 0., 0., 0., 1., 0., 1.],
                             [0., 0., 0., 0., 1., 0., 0., 0., 1., 0.]])

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        actions_numpy = actions.detach().cpu().numpy()
        reward = 0

        for agent_i, action in enumerate(actions):
            if action == 1 and self.backlogs[agent_i] > 0:
                if (self.adj[agent_i] * actions_numpy).sum() < 0.01:
                    self.backlogs[agent_i] = self.backlogs[agent_i] - 1
                    self.transmitted += 1
                    reward += 0.1
                else:
                    reward -= 10

        terminated = False
        info['trans'] = self.transmitted
        info['left'] = self.backlogs.sum()
        info['battle_won'] = False

        # Add new packages
        self.backlogs += np.random.choice([0., 1.], p=[0.4, 0.6], size=[self.n_agents])
        self.backlogs = np.clip(self.backlogs, a_min=0, a_max=self.max_list_length)

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

            if self.transmitted > self.n_agents / 2 * self.episode_limit * 0.9:
                info['battle_won'] = True
                self.battles_won += 1

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return np.array([self.backlogs[agent_id]])

    def get_obs_size(self):
        """Returns the size of the observation."""
        return 1

    def get_state(self):
        """Returns the global state."""
        return self.backlogs

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.n_agents

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
        self.transmitted = 0
        self.backlogs = np.ones(self.n_agents)

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
