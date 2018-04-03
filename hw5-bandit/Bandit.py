import numpy as np

class Bandit:
    def __init__(self, n):
        self.num_arms = n
        self.param_arms = np.zeros((n,2))
        self.opt_reward = 0

    def NumArms(self):
        return(self.num_arms)

    def SetParams(self, a, r, p):
        self.param_arms[a][0] = r
        self.param_arms[a][1] = p
        self.opt_reward = np.max(self.param_arms[:,0] * self.param_arms[:,1])
        #print("opt_reward", self.opt_reward)

    def Pull(self, a):
        reward = self.SBRD(self.param_arms[a][0], self.param_arms[a][1])
        return reward[0]

    def SBRD(self, r, prob):
        r_list = [r, 0.0]
        reward = np.random.choice(r_list, 1, p=[prob, 1-prob])
        return reward