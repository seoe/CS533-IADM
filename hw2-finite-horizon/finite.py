# -*- coding: utf-8 -*-
# CS533 Homework #2 - Spring 2017
# Eugene Seo (OSUID: 932981978)
# Due: 04-17-2017
# Optimizing Finite-Horizon Expected Total Reward

import sys
import numpy as np

def bellman_backup(state_size, action_size, T, s, V, t):
    future_values_by_actions = [] # list of expected future values with each actions    
    for a in range(0, action_size):
        future_values = 0 # accumulated future value by an action
        for s_next in range(0, state_size):
            value_s_next = T[a,s,s_next] * V[s_next,t-1] # value by the next state
            future_values = future_values + value_s_next
        future_values_by_actions.append(future_values)
    
    max_value = np.max(future_values_by_actions)
    action = np.argmax(future_values_by_actions) + 1 # action number starting from 1
    
    return max_value, action
      
def policy_optimization(state_size, action_size, T, R, H):
    V = np.zeros((state_size, H))
    policy = np.zeros((state_size, H))
    
    V[:,0] = R # rewards for 0 stage-to-go
    
    for t in range(1, H):
        for s in range(0, state_size):
            max_value, action = bellman_backup(state_size, action_size, T, s, V, t)
            V[s,t] = R[s] + max_value
            policy[s,t] = action
    
    print("V")
    print(V)
    print("\nPolicy")
    print(policy)
    
    return V, policy              
            
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
    
def write_file(filename, H, V, policy):
    output_filename = "./output/output_" + str(H) + "_" + filename
    with file(output_filename, 'w') as outfile:
        outfile.write('V\n')
        np.savetxt(outfile, V, delimiter=' ', fmt='%-7.4f')
        outfile.write('\nPolicy\n')
        np.savetxt(outfile, policy, delimiter=' ', fmt='%d')    
    
def main():
    filename = sys.argv[1]
    H = int(sys.argv[2])
    state_size, action_size, T, R = read_file(filename)    
    V, policy = policy_optimization(state_size, action_size, T, R, H)
    write_file(filename, H, V, policy)        
    
if __name__ == "__main__":
    main()