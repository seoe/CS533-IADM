import numpy as np
from collections import *
import csv

def load_MDP(MDP_id, state_size):
    action_size = 4   
    T = np.zeros((action_size, state_size, state_size))
    
    action_num = -1
    i = 0
    with open(MDP_id, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if "Action" in row:
                action_num += 1
                i = 0
            else:
                T[action_num, i,] = row
                i += 1
    print(T)
    return T