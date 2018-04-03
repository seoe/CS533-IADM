"""
MDP class builds a MDP model given parameters (n, rewards, probabilities)
It also provides the initial state or the next state to an agent based on its MDP
"""

import numpy as np

class MDP:
    def __init__(self, n):
        print("========================= MDP INIT ==========================")
        self.n              = n
        self.state_size     = n * 8
        self.action_size    = 2
        self.T              = np.zeros((self.action_size, self.state_size, self.state_size))
        self.R              = np.zeros(self.state_size)

    def make_MDP(self, filename, rewards, probabilities):
        penalty_driving     = rewards[0]
        penalty_handicap    = rewards[1]
        penalty_collision   = rewards[2]
        best_reward         = rewards[3]

        prob_avail_handicap = probabilities[0]
        prob_T              = probabilities[1]

        self.build_MDP( self.n,
                        penalty_driving,
                        penalty_handicap,
                        penalty_collision,
                        best_reward,
                        prob_avail_handicap,
                        prob_T,
                        filename )

    def build_MDP(self, n, penalty_driving, penalty_handicap, penalty_collision,
                        best_reward, prob_avail_handicap, prob_T, filename):
        # Define trainsition functions for each action - Drive, Park, and Exit
        T_drive = np.zeros((self.state_size,self.state_size))
        T_park = np.zeros((self.state_size,self.state_size))
        T_exit = np.identity(self.state_size)
        # Define a reward function
        R = np.zeros(self.state_size)

        # Set the probability of availability of each parking spot
        prob_availability = np.zeros(2*n)
        prob_availability[0] = prob_availability[2*n-1] = prob_avail_handicap # the probability of availability of handicap spots
        for i in range(1,n):
            prob_availability[i] = prob_availability[2*n-1-i] = 1.0 - (1.0/(i+prob_T))

        # Trainsition functions for terminal states, in which parking status is true
        sub_T_drive_parked = np.zeros((4,4))
        sub_T_drive_parked[1,1] = sub_T_drive_parked[3,3] = 1.0

        # Sub trainsition functions for Park action
        sub_T_park = np.zeros((4,4))
        sub_T_park[0,1] = sub_T_park[1,1] = sub_T_park[2,3] = sub_T_park[3,3] = 1.0

        # Define trainsition functions for Drive and Park actions, and the reward function
        for i in range(0, n * 2):
            # T_drive
            T_drive[4*i+0:4*i+4,4*i+0:4*i+4] = sub_T_drive_parked # self transition in parking status no matter the type of actions

            sub_T_drive_spot = np.zeros((4,4))
            sub_T_drive_spot[0,0] = sub_T_drive_spot[2,0] = prob_availability[i] # Probability that a parking spot is available
            sub_T_drive_spot[0,2] = sub_T_drive_spot[2,2] = 1 - prob_availability[i] # Probability that a parking spot is not availabe

            if i < n * 2 - 1:
                T_drive[4*i+0:4*i+4,4*i+4:4*i+8] = sub_T_drive_spot
            else: # transition from the last state (Sn=A2) to the first state (S1=A1)
                T_drive[4*i+0:4*i+4,0:4] = sub_T_drive_spot

            # T_park
            T_park[4*i+0:4*i+4,4*i+0:4*i+4] = sub_T_park

            # R
            if i < 2:
                R[4*i+0:4*i+4] = [penalty_driving, penalty_handicap, penalty_driving, penalty_collision]
            else:
                R[4*i+0:4*i+4] = [penalty_driving, best_reward / i, penalty_driving, penalty_collision]

            R1 = R.reshape(self.state_size/4,4)
            R1[n+1:,1] = R1[2:n+1,1][::-1]
            R = R1.reshape(1,self.state_size)[0]

        self.T[0] = T_drive
        self.T[1] = T_park
        #self.T[2] = T_exit
        self.R = R

        self.write_MDP(filename, T_drive, T_park, T_exit, R)

    def write_MDP(self, filename, T_drive, T_park, T_exit, R):
        R = np.reshape(R, (1, self.state_size))
        f = open(filename, "w")
        f.write("{} {}".format(self.state_size,self.action_size))
        f.write("\n\n")
        np.savetxt(f, T_drive, fmt='%-7.2f')
        f.write("\n")
        np.savetxt(f, T_park, fmt='%-7.2f')
        f.write("\n")
        np.savetxt(f, T_exit, fmt='%-7.2f')
        f.write("\n")
        np.savetxt(f, R, fmt='%-7.2f')
        f.close()

    def init_state(self):
        init_s_list = [4,6, 4+4*self.n, 6+4*self.n]
        init_s = np.random.choice(init_s_list, 1, p = [0.4, 0.1, 0.4, 0.1]) #TODO
        return init_s

    def next_state(self, s, a):
        next_states = np.where(self.T[a, s][0] > 0)[0]
        prob = self.T[a, s][0][next_states]
        next_s = np.random.choice(next_states, 1, p = prob)
        return next_s