"""
It's an agent playing in the simulation.
It also learns the optimal behavior through many trials.
"""

import numpy as np
import MDP as mdp

class Agent:
    def __init__(self, mdp):
        print("========================= Agent INIT ==========================")
        self.s      = 0
        self.a      = 0
        self.next_s = 0
        self.reward = 0

        self.Q      = np.zeros((mdp.state_size, mdp.action_size))
        self.policy = np.zeros(mdp.state_size)

    def init_state(self, init_s):
        self.s      = init_s
        self.a      = 0
        self.next_s = 0
        self.reward = 0

    def set_policy(self, policy):
        policy = policy.astype(int)
        self.policy = policy

    def GLIE_policy(self, s, t):
        op = np.random.choice(2, 1, p=[1/t, 1-1/t])
        if op == 0: # random
            a = np.random.choice(self.action_size, 1)
        else: # greedy
            a = np.argmax(self.Q[s])
        return a