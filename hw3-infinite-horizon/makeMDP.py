# -*- coding: utf-8 -*-
# CS533 Homework #3 - Spring 2017
# Eugene Seo (OSUID: 932981978)
# Due: 04-26-2017
# Description: Produce MDPs from certain parameters that characterize a parking problem (rewards, probabilities)

import sys
import numpy as np

def write_MDP(filename, state_size, T_drive, T_park, T_exit, R):
    R = np.reshape(R, (1, state_size))
    f = open(filename, "w")
    f.write("{} {}".format(state_size,3))
    f.write("\n\n")
    np.savetxt(f, T_drive, fmt='%-7.2f')
    f.write("\n")
    np.savetxt(f, T_park, fmt='%-7.2f')
    f.write("\n")
    np.savetxt(f, T_exit, fmt='%-7.2f')
    f.write("\n")
    np.savetxt(f, R, fmt='%-7.2f')
    f.close()


def build_MDP(  n, penalty_driving, penalty_handicap, penalty_collision,
                best_reward, prob_avail_handicap, prob_T):

    state_size = 8 * n

    # Define trainsition functions for each action - Drive, Park, and Exit
    T_drive = np.zeros((state_size,state_size))
    T_park = np.zeros((state_size,state_size))
    T_exit = np.identity(state_size)
    # Define a reward function
    R = np.zeros(state_size)

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
        #sub_T_drive_spot[1,1] = sub_T_drive_spot[1,3] = 1
        #sub_T_drive_spot[3,1] = sub_T_drive_spot[3,3] = 1
        
        if i < n * 2 - 1:
            T_drive[4*i+0:4*i+4,4*i+4:4*i+8] = sub_T_drive_spot
        else: # transition from the last state (Sn=A2) to the first state (S1=A1)
            T_drive[4*i+0:4*i+4,0:4] = sub_T_drive_spot

        # T_park
        T_park[4*i+0:4*i+4,4*i+0:4*i+4] = sub_T_park

        # R
        if i < 2:
            R[4*i+0:4*i+4] = [penalty_driving, penalty_handicap, penalty_driving, penalty_collision]
        elif i > 1 and i < n + 1:
            R[4*i+0:4*i+4] = [penalty_driving, best_reward - i, penalty_driving, penalty_collision]
        else:
            R[4*i+0:4*i+4] = [penalty_driving, i - best_reward + 1, penalty_driving, penalty_collision]

    return T_drive, T_park, T_exit, R


def main():
    n = int(sys.argv[1])
    state_size = 8 * n

    filename = "MDParking1.txt"
    # ==================== 1st Parameter Setting =================== #
    # Parameters for rewards
    penalty_driving = -1
    penalty_handicap = -10
    penalty_collision = -100
    best_reward = n + 1

    # Parameters of probabilities that a parking spot if available
    prob_avail_handicap = 0.9
    prob_T = 1.0 # probability temperature
    # ============================================================== #

    T_drive, T_park, T_exit, R = build_MDP( n,
                                            penalty_driving,
                                            penalty_handicap,
                                            penalty_collision,
                                            best_reward,
                                            prob_avail_handicap,
                                            prob_T )

    write_MDP(filename, state_size, T_drive, T_park, T_exit, R)

    filename = "MDParking2.txt"
    # ==================== 2nd Parameter Setting =================== #
    # Parameters for rewards
    penalty_driving = -100
    penalty_handicap = -1
    penalty_collision = -100
    best_reward = n + 1

    # Parameters of probabilities that a parking spot if available
    prob_avail_handicap = 0.9
    prob_T = 5.0 # probability temperature
    # ============================================================== #

    T_drive, T_park, T_exit, R = build_MDP( n,
                                            penalty_driving,
                                            penalty_handicap,
                                            penalty_collision,
                                            best_reward,
                                            prob_avail_handicap,
                                            prob_T )

    write_MDP(filename, state_size, T_drive, T_park, T_exit, R)


if __name__ == "__main__":
    main()