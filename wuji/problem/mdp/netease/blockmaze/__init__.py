"""
Copyright (C) 2018--2020, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import operator
import types

import gym
import numpy as np

import wuji
from wuji.problem.mdp.netease.blockmaze.blockmaze import Env

NAME = os.path.basename(os.path.dirname(os.path.splitext(__file__)[0]))
LENGTH = 40
gym.envs.register(id='RandomMaze-v0', entry_point=Env, max_episode_steps=LENGTH)


class Problem(object):
    class Controller(object):
        def __init__(self, problem, kind, state):
            self.problem = problem
            self.kind = kind
            self.state = state
            self.config = problem.config
            self.context = problem.context
            self.env = problem.env
            self.bug_set = set()

        def get_state(self):
            return self.state

        async def __call__(self, action):
            input, self.reward, done, self.info = self.env.step(action)
            self.state = self.problem.make_state(input, self.info)
            self.problem.frame += 1
            exp = dict(
                done=done,
            )
            if self.info['bug'] is not None:
                self.bug_set.add(self.info['bug'])
                self.problem.bug_set.add(self.info['bug'])
            return exp

        def get_rewards(self, **kwargs):
            return np.array([])

        def get_reward(self, **kwargs):
            return np.mean(kwargs['rewards'] * self.problem.get_weight_reward())

        def get_final_rewards(self, **kwargs):
            return np.array([])

        def get_final_reward(self, **kwargs):
            return np.mean(kwargs['rewards'] * self.problem.get_weight_final_reward())

        def update(self, exp):
            if exp['done']:
                exp['rewards'] = self.get_final_rewards(**exp)
                exp['reward'] = self.get_final_reward(**exp)
            else:
                exp['rewards'] = self.get_rewards(**exp)
                exp['reward'] = self.get_reward(**exp)

        def get_result(self):
            return dict(
                fitness=0,
                objective=[],
                point=[],
            )

    def __init__(self, config):
        self.config = config
        self.env = gym.make('RandomMaze-v0')
        self.context = dict(
            encoding=dict(
                blob=dict(
                    init=[dict(
                        kwargs=self.env.context,
                    )],
                ),
            ),
            name_reward=[],
            weight_reward=np.array([]),
            name_final_reward=[],
            weight_final_reward=np.array([]),
            length=LENGTH,
        )
        self.bug_set = set()

    def close(self):
        self.env.close()

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key in {'context'}}

    def __setstate__(self, state):
        return self.__dict__.update(state)

    def __len__(self):
        return 1

    def make_state(self, input, info):
        state = dict(
            inputs=(input,),
            info=info,
        )
        return state

    def reset(self, *args):
        self.frame = 0
        state = self.make_state(*self.env.reset())
        return [self.Controller(self, kind, state) for kind in args], []

    def evaluating(self, *args, **kwargs):
        return types.SimpleNamespace(close=lambda: None)

    def evaluate_reduce(self, results):
        result = {key: wuji.nanmean(list(map(operator.itemgetter(key), results)), 0) for key in results[0]}
        result['bug_set'] = self.bug_set
        result['bug_found'] = len(self.bug_set)
        return result

    def set_name_reward(self, value):
        self.context['name_reward'] = value

    def get_name_reward(self):
        return self.context['name_reward']

    def set_weight_reward(self, value):
        self.context['weight_reward'] = value

    def get_weight_reward(self):
        return self.context['weight_reward']

    def set_name_final_reward(self, value):
        self.context['name_final_reward'] = value

    def get_name_final_reward(self):
        return self.context['name_final_reward']

    def set_weight_final_reward(self, value):
        self.context['weight_final_reward'] = value

    def get_weight_final_reward(self):
        return self.context['weight_final_reward']

    def render(self):
        return self.env.render()
