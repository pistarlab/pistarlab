import sys

import numpy
import numpy as np
from gym.envs.toy_text import discrete
from PIL import Image, ImageDraw
numpy.random.seed(1)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# TODO: improvements
# - Add version which uses image and text and uses full observation.  also, perhaps graph version
# - randomly places target and cliff in different arrangements, but guarentees path


# original source: https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py

class CliffWalkingEnv(discrete.DiscreteEnv):
    """
    Point of game is to get to T, with out touching C
    User starts at top left of grid.

    This is a perfect information game

    C is a cliff
    T is the target

    o  o  o  o  o  o  o  o  o  o  o  x
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T


    Rewards
    C = -100
    o = -1
    T = goal

    transition probablities


    # NOTES FROM SOURCE
    This is a simple implementation of the Gridworld Cliff
    reinforcement learning task.
    Adapted from Example 6.6 (page 106) from Reinforcement Learning: An Introduction
    by Sutton and Barto:
    http://incompleteideas.net/book/bookdraft2018jan1.pdf
    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py
    The board is a 4x12 matrix, with (using Numpy matrix indexing):
        [3, 0] as the start at bottom-left
        [3, 11] as the goal at bottom-right
        [3, 1..10] as the cliff at bottom-center
    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward
    and a reset to the start. An episode terminates when the agent reaches the goal.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        reward = -100.0 if (self._cliff[tuple(new_position)]) else -1.0
        is_done = self._cliff[tuple(new_position)] or (tuple(new_position) == (3, 11))
        return [(1.0, new_state, reward, is_done)]

    def __init__(self):
        self.shape = (4, 12)
        self.step_counter = 0
        self.max_steps = 100

        nS = np.prod(self.shape)  # number of states
        nA = 4  # Number of actions

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3, 0), self.shape)] = 1.0

        super(CliffWalkingEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        str_output = self._render(mode, close)
        if mode == 'rgb_array':
            img = Image.new('RGB', (250, 80), color=(0, 0, 0))
            d = ImageDraw.Draw(img)
            d.text((20, 20), str_output, fill=(255, 255, 255))
            return numpy.array(img)

    def reset(self):
        self.step_counter = 0
        return super(CliffWalkingEnv,self).reset()
        
    def step(self,a):
        self.step_counter += 1
        time_out  = self.step_counter > self.max_steps
        s, r, d, p =  super(CliffWalkingEnv,self).step(a)
        if not d and time_out:
            r = -100
            d = True
        return s, r, d, p

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = sys.stdout
        output_st = ""

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " X "
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
            output_st += output
        outfile.write("\n")
        return output_st + "\n"
