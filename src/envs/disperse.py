from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import numpy as np
import random


class DisperseEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
            self,
            n_agents=12,
            n_actions=4,
            initial_need=[0, 0, 0, 0],
            episode_limit=10,
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

        # Other
        self._seed = seed

        # Actions
        self.n_actions = n_actions
        self.initial_need = initial_need

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.episode_limit = episode_limit
        self.needs = initial_need
        self.actions = np.random.randint(0, n_actions, n_agents)
        self._match = 0

    def _split_x(self, x, n):
        result = np.zeros(n)
        p = np.random.randint(low=0, high=n)
        low = x // 2
        result[p] = np.random.randint(low=low, high=x+1)
        return result

    def step(self, actions):
        """Returns reward, terminated, info."""
        # print(self.needs)
        # print(actions)
        self._total_steps += 1
        self._episode_steps += 1
        info = {}
        self.actions = actions

        terminated = False
        info['battle_won'] = False
        # actions_numpy = actions.detach().cpu().numpy()

        delta = []
        for action_i in range(self.n_actions):
            supply = float((actions == action_i).sum())
            need = float(self.needs[action_i])

            if supply >= need:
                self._match += 1

            delta.append(min(supply - need, 0))
        reward = float(np.array(delta).sum()) / self.n_agents

        # print('step', self._episode_steps, ':')
        # print(self.needs)
        # print(self.actions)
        # print(reward)

        self.needs = self._split_x(self.n_agents, self.n_actions)
        info['match'] = self._match

        if self._episode_steps >= self.episode_limit:
            # print(self._match)
            # print(reward)
            terminated = True
            self._episode_count += 1
            self.battles_game += 1

            if self._match == self.n_actions * self.episode_limit:
                info['battle_won'] = True
                self.battles_won += 1

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        agent_action = self.actions[agent_id]
        # print([agent_action, self.needs[agent_action]])
        # print([float(x) for x in (self.actions == agent_action)])
        action_one_hot = np.zeros(self.n_actions)
        action_one_hot[agent_action] = 1.
        return np.concatenate((action_one_hot, [self.needs[agent_action]], [float(x) for x in (self.actions == agent_action)]))
        # return np.array([agent_action, self.needs[agent_action], (self.actions == agent_action).sum()])

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.n_actions + 1 + self.n_agents

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

        self.needs = self.initial_need
        self.actions = np.random.randint(0, self.n_actions, self.n_agents)
        self._match = 0

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