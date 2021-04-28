from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from smac.env.multiagentenv import MultiAgentEnv
import copy


class GatherEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
            self,
            n_agents=7,
            episode_limit=20,
            map_height=3,
            map_width=7,
            catch_reward=10,
            catch_fail_reward=-5,
            target_reward=0.000,
            other_reward=5,
            obs_last_action=False,
            state_last_action=True,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents
        self.episode_limit = episode_limit
        self.map_height = map_height
        self.map_width = map_width
        self.catch_reward = catch_reward
        self.catch_fail_reward = catch_fail_reward
        self.other_reward = other_reward
        self.target_reward = target_reward

        # Observations and state
        self.obs_last_action = obs_last_action
        self.state_last_action = state_last_action

        # Other
        self._seed = seed

        # Actions
        self.n_actions = 5

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Configuration initialization
        self.agent_positions_idx = np.zeros((self.n_agents, 2)).astype(int)
        target_count = [0, 0, 0]
        self.agent_target = [np.zeros(2) for _ in range(self.n_agents)]

        for agent_i in range(self.n_agents):
            agent_x = np.random.randint(low=0, high=self.map_width)
            agent_y = np.random.randint(low=0, high=self.map_height)

            self.agent_positions_idx[agent_i, 0] = agent_y
            self.agent_positions_idx[agent_i, 1] = agent_x

            if self._distance(agent_x, 0, agent_y, 1) < self._distance(agent_x, 2, agent_y, 1):
                self.agent_target[agent_i] = np.array([1, 0])
                target_count[0] += 1
            else:
                if self._distance(agent_x, 4, agent_y, 1) <= self._distance(agent_x, 2, agent_y, 1):
                    self.agent_target[agent_i] = np.array([1, 4])
                    target_count[2] += 1
                else:
                    self.agent_target[agent_i] = np.array([1, 2])
                    target_count[1] += 1

        if target_count[0] >= target_count[1] and target_count[0] >= target_count[2]:
            self.target = np.array([1, 0])
            self.n_target = np.array([1, 2])
            self.n2_target = np.array([1, 4])
        else:
            if target_count[1] >= target_count[0] and target_count[1] >= target_count[2]:
                self.target = np.array([1, 2])
                self.n_target = np.array([1, 0])
                self.n2_target = np.array([1, 4])
            else:
                self.target = np.array([1, 4])
                self.n_target = np.array([1, 2])
                self.n2_target = np.array([1, 0])

        for agent_i in range(self.n_agents):
            if self.agent_target[agent_i][1] != self.target[1] or self.agent_target[agent_i][0] != self.target[0]:
                self.agent_target[agent_i] = np.array([-1, -1])



    def _distance(self, x1, x2, y1, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        reward = 0
        terminated = False
        info['battle_won'] = False

        occ_count = 0
        n_occ_count = 0
        n2_occ_count = 0

        # map = np.zeros((self.map_height, self.map_width))

        for agent_i, action in enumerate(actions):
            y, x = self.agent_positions_idx[agent_i, 0], self.agent_positions_idx[agent_i, 1]

            target_x = x
            target_y = y

            if action == 0:
                target_x, target_y = x, min(self.map_height - 1, y + 1)
            elif action == 1:
                target_x, target_y = min(x + 1, self.map_width - 1), y
            elif action == 2:
                target_x, target_y = x, max(0, y - 1)
            elif action == 3:
                target_x, target_y = max(0, x - 1), y

            self.agent_positions_idx[agent_i, 0], self.agent_positions_idx[agent_i, 1] = target_y, target_x
            # map[target_y, target_x] += 1

            if target_x == self.target[1] and target_y == self.target[0]:
                occ_count += 1
            elif target_x == self.n_target[1] and target_y == self.n_target[0]:
                n_occ_count += 1
            elif target_x == self.n2_target[1] and target_y == self.n2_target[0]:
                n2_occ_count += 1

        # print(map)

        if occ_count == self.n_agents:
            terminated = True
            info['battle_won'] = True
            self.battles_won += 1
            reward += self.catch_reward

        if self._episode_steps >= self.episode_limit:
            terminated = True

            if occ_count + n_occ_count+ n2_occ_count == self.n_agents:
                if occ_count == 0:
                    reward += self.other_reward
                elif occ_count < self.n_agents:
                    reward += self.catch_fail_reward

        if terminated:
            #print("terminated")
            #print(reward)
            #print(occ_count)
            #print(n_occ_count)
            #print(n2_occ_count)
            self._episode_count += 1
            self.battles_game += 1

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        # return self.agent_positions_idx[agent_id]
        return np.concatenate([self.agent_positions_idx[agent_id], self.agent_target[agent_id]])

    def get_obs_size(self):
        """Returns the size of the observation."""
        return 4

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
        return [1 for _ in range(self.n_actions)]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Configuration initialization
        self.agent_positions_idx = np.zeros((self.n_agents, 2)).astype(int)
        target_count = [0, 0, 0]
        self.agent_target = [np.zeros(2) for _ in range(self.n_agents)]

        for agent_i in range(self.n_agents):
            agent_x = np.random.randint(low=0, high=self.map_width)
            agent_y = np.random.randint(low=0, high=self.map_height)

            self.agent_positions_idx[agent_i, 0] = agent_y
            self.agent_positions_idx[agent_i, 1] = agent_x

            if self._distance(agent_x, 0, agent_y, 1) < self._distance(agent_x, 2, agent_y, 1):
                self.agent_target[agent_i] = np.array([1, 0])
                target_count[0] += 1
            else:
                if self._distance(agent_x, 4, agent_y, 1) <= self._distance(agent_x, 2, agent_y, 1):
                    self.agent_target[agent_i] = np.array([1, 4])
                    target_count[2] += 1
                else:
                    self.agent_target[agent_i] = np.array([1, 2])
                    target_count[1] += 1

        if target_count[0] >= target_count[1] and target_count[0] >= target_count[2]:
            self.target = np.array([1, 0])
            self.n_target = np.array([1, 2])
            self.n2_target = np.array([1, 4])
        else:
            if target_count[1] >= target_count[0] and target_count[1] >= target_count[2]:
                self.target = np.array([1, 2])
                self.n_target = np.array([1, 0])
                self.n2_target = np.array([1, 4])
            else:
                self.target = np.array([1, 4])
                self.n_target = np.array([1, 2])
                self.n2_target = np.array([1, 0])

        for agent_i in range(self.n_agents):
            if self.agent_target[agent_i][1] != self.target[1] or self.agent_target[agent_i][0] != self.target[0]:
                self.agent_target[agent_i] = np.array([-1, -1])

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
