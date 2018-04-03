# -*- coding: utf-8 -*-
# CS533 Homework #3 - Spring 2017
# Eugene Seo (OSUID: 932981978)
# Due: 04-26-2017
# Description: Optimizing Infinite-Horizon Discounted Reward with Application to Optimal Parking

import sys
import numpy as np
from numpy.linalg import inv
from copy import deepcopy

def bellman_backup(state_size, action_size, T, s, V):
    future_values_by_actions = [] # list of expected future values with each actions
    for a in range(0, action_size):
        future_values = 0 # accumulated future value by an action
        for s_next in range(0, state_size):
            value_s_next = T[a,s,s_next] * V[s_next] # value by the next state
            future_values = future_values + value_s_next
        future_values_by_actions.append(future_values)

    max_value = np.max(future_values_by_actions)
    action = np.argmax(future_values_by_actions) + 1 # action number starting from 1

    return max_value, action


def policy_evaluation(state_size, T, R, P, df): # for infinite
	I = np.identity(state_size)
	return np.dot(inv(I-np.dot(df,T)),R)


def policy_optimization(state_size, action_size, T, R, df): # by value iteraction
    V = np.zeros(state_size)
    policy = np.zeros(state_size)
    T_opt = np.zeros((state_size,state_size)) # for infinite

    V_pre = 100000
    V[:] = R # initialize to R(s)
    iteration = 0

    # compute stopping condition
    expected_diff = 0.01
    stop_point = (expected_diff * np.square(1-df)) / (2 * np.square(df))
    print("stop_point", stop_point)

    # 1. Valute Interaction
    while ( max(abs(V_pre-V)) > stop_point ): # until convergence
        iteration += 1
        print("iteration", iteration, max(abs(V_pre-V)))
        V_pre = deepcopy(V)
        for s in range(0, state_size):
            # 2. Greedy Policy
            max_value, action = bellman_backup(state_size, action_size, T, s, V)
            V[s] = R[s] + df * max_value # discounted expected value
            policy[s] = action
            T_opt[s,:] = T[action-1,s,:] # for matrix computation
    
    # 3. Policy Evaluation
    V_opt = policy_evaluation(state_size, T_opt, R, policy, df) # for infinite

    return V_opt, policy


def build_states(n):
    s = []
    for i in range(0,2):
        for j in range(0,2):
            s_name = "A1" + str(i) + str(j)
            s.append(s_name)
    for k in range(1,n+1):
        for i in range(0,2):
            for j in range(0,2):
                s_name = "B" + str(k) + str(i) + str(j)
                s.append(s_name)
    for k in range(0,n-1):
        for i in range(0,2):
            for j in range(0,2):
                s_name = "A" + str(n-k) + str(i) + str(j)
                s.append(s_name)
    return(s)


def read_file(filename):
    blankLines = []
    f = open(filename)
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
            if line == '\r\n' or line == '\n':
                blankLines.append(i)

    input_size = lines[0].split()
    state_size = int(input_size[0].decode('utf-8-sig'))
    action_size = int(input_size[1])

    T = np.zeros((action_size, state_size, state_size))
    for i in range(0, action_size):
        T_A = lines[blankLines[i]:blankLines[i+1]-1]
        T_A = map(str.split, T_A)
        T_A = np.array(T_A)
        T[i] = T_A

    R = lines[len(lines)-1].split()
    R = map(float, lines[len(lines)-1].split())

    return state_size, action_size, T, R


def write_file(filename, H, V, P):
    output_filename = "./output/output_" + str(H) + "_" + filename
    with file(output_filename, 'w') as outfile:
        outfile.write('V\n')
        np.savetxt(outfile, V, delimiter=' ', fmt='%s')
        outfile.write('\nPolicy\n')
        np.savetxt(outfile, P, delimiter=' ', fmt='%s')


def main():
    filename = sys.argv[1]
    discount_factor = float(sys.argv[2])
    if discount_factor < 0 or discount_factor >= 1:
        print("[Error] The discount factor should be >= 0 and < 1\n")
        sys.exit(0)
    state_size, action_size, T, R = read_file(filename)

    V, P = policy_optimization(state_size, action_size, T, R, discount_factor)
    print(np.max(V))
    
    if "Parking" in filename:
        s = build_states(state_size/8)
        s = np.array(s)
        s = s.reshape((state_size, 1))
        V = V.reshape((state_size, 1))
        P = P.reshape((state_size, 1))
        V = np.concatenate((s,V), axis=1)
        P = np.concatenate((s,P), axis=1)
    
    write_file(filename, discount_factor, V, P)


if __name__ == "__main__":
    main()