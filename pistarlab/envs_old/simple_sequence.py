import os
import random

import gym
from gym import spaces

os.environ["TERM"] = 'xterm'

clear = lambda: os.system('clear')


# TODO: should extend PEnv
class SimpleSequence(gym.Env):
    """
    # Modeled after gym env

    This is intended to be a general template for a function learning environment

    Currently only one function implemented.
    Note Game never ends

    **Sum is Even Game**
    HOW TO PLAY:
    The user is presented with a series of symbols, one per turn.
    The point is to correctly answer the question, is the sum of the last two numbers even or odd.

    Valid Actions: integers 0,1, or 2.
    0 = no action
    1 = odd
    2 = even

    TODO: delayed reward/ reward offset
    """

    def __init__(self, enable_render=False):
        self.char_render_max = 150
        self.max_value = 5

        self.noise_token = "_"
        self.ready_token = "?"
        self.noise_thres = 0.2
        self.num_thres = 0.2
        self.state = {}
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(self.max_value + 2)
        self.current_obs = self.noise_token
        self.frame_counter = 0
        self.add_number_counter = 0
        self.full_seq = []
        self.current_action = 0
        self.action_seq = []
        self.current_reward = 0
        self.reward_seq = []
        self.mode_seq = []
        self.states = [str(x) for x in range(self.max_value)] + [self.noise_token, self.ready_token]
        self.score = 0
        self.reward_early_resp = 0
        self.reward_success_resp = 1
        self.reward_unsuccess_resp = -1
        self.reward_default = 0
        self.reward_no_resp = -2
        self.state_map = {t: i for i, t in enumerate(self.states)}
        self.state['mode'] = 'P'
        self.enable_render = enable_render

    def get_output_value(self, v):
        return self.state_map[v]

    def check_response(self, action_index):
        if self.state['mode'] != 'R':
            return self.reward_early_resp
        self.state['mode'] = 'A'
        return self.is_even_or_odd(action_index)

    def is_even_or_odd(self, action_index):
        sum = self.state['vals'][-1] + self.state['vals'][-2]
        even = (sum % 2 == 0)
        if (action_index == 1 and not even) or (action_index == 2 and even):
            return self.reward_success_resp
        else:
            return self.reward_unsuccess_resp

    def step(self, action, human=False):
        self.current_action = action
        self.frame_counter += 1
        self.current_reward = self.reward_default

        if action != 0:
            self.current_reward = self.check_response(action)

        if self.state['mode'] == 'A':
            self.current_obs = self.noise_token
            self.state['mode'] = 'P'
            self.add_number_counter = 0
        elif random.random() <= self.noise_thres:
            self.current_obs = self.noise_token
        elif self.state['mode'] == 'R':
            self.current_obs = self.noise_token
            self.state['mode'] = 'P'
            self.current_reward = self.reward_no_resp
            self.add_number_counter = 0
        elif random.random() <= self.num_thres or self.add_number_counter < 2:
            self.add_number_counter += 1
            val = int(random.randint(0, self.max_value - 1))
            self.state['vals'].append(val)
            self.current_obs = str(val)
        else:
            self.current_obs = self.ready_token
            self.state['mode'] = 'R'

        self.score += self.current_reward

        result_obs = self.current_obs
        if not human:
            result_obs = self.get_output_value(result_obs)

        # TODO, should be able to exit game cleanly

        return result_obs, self.current_reward, False, {}

    def reset(self, human=False):
        self.state['mode'] = 'P'
        self.state['vals'] = []
        self.current_obs = self.noise_token
        self.frame_counter = 0
        self.add_number_counter = 0
        self.current_action = 0
        self.current_reward = 0
        self.score = 0
        result_obs = self.current_obs
        if not human:
            result_obs = self.get_output_value(result_obs)
        return result_obs

    def render(self, mode='human', close=False):
        if not self.enable_render:
            return
        print("\n" * 100)
        clear()

        start_idx = max(0, len(self.full_seq) - self.char_render_max)
        print(" ")
        self.full_seq.append(self.current_obs)
        print("O: " + "".join([str(x) for x in self.full_seq[start_idx:]]))

        self.action_seq.append(self.current_action)
        print("A: " + "".join([str(x) for x in self.action_seq[start_idx:]]))

        def reward_st(i):
            if i == -1:
                return "D"
            elif i < -1:
                return "F"
            else:
                return str(i)

        self.reward_seq.append(self.current_reward)
        print("R: " + "".join([reward_st(x) for x in self.reward_seq[start_idx:]]))

        self.mode_seq.append(self.state['mode'])
        print("M: " + "".join(self.mode_seq[start_idx:]))

        print("Score: %s" % self.score)
