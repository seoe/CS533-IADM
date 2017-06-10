# Final Project, Laurel Hopkins & Eugene Seo
# CS 533, Spring 2017

from random import randint
from collections import *
import numpy as np
import read_file as rf


def expert0_init(states):
	ex0 = defaultdict(list)    
	'''
	# Action 1
	act1 = np.zeros((states, states))
	for i in range(0, states):
		act1[i][i] = 1
	ex0[1] = act1

	# Action 2
	act2 = np.zeros((states, states))
	for i in range(0, states):
		if i == (states - 1):
			act2[i][0] = 0.5
			act2[i][i] = 0.5
		else:
			act2[i][i] = 0.5
			act2[i][i+1] = 0.5
	ex0[2] = act2
	'''
	T = rf.load_MDP('MDP1.csv', states)
	for i in range(len(T)):	
		ex0[i+1] = T[i]
	return ex0


def expert1_init(states):
	ex1 = defaultdict(list)
	'''
	# Action 1
	act1 = np.zeros((states, states))
	for i in range(0, states):
		if i == (states - 1):
			act1[i][0] = 0.5
			act1[i][i] = 0.5
		else:
			act1[i][i] = 0.5
			act1[i][i + 1] = 0.5
	ex1[1] = act1

	# Action 2
	act2 = np.zeros((states, states))
	for i in range(0, states):
		act2[i][i] = 1
	ex1[2] = act2
	'''
	T = rf.load_MDP('MDP2.csv', states)
	for i in range(len(T)):
		ex1[i+1] = T[i]
	return ex1


def expert_init(filename, states):
	ex = defaultdict(list)
	T = rf.load_MDP(filename, states)
	for i in range(len(T)):
		ex[i+1] = T[i]
	return ex
	

def policy_init(states):
	num_policies = 8
	policy = np.zeros((num_policies, states))
	'''
	policy[0] = [1, 1, 1, 1]
	policy[1] = [2, 2, 2, 2]
	policy[2] = [1, 2, 1, 2]
	'''
	policy[0] = [4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1] # if size is small, then first enlarge. if size is large, then plant.
	policy[1] = [1, 1, 1, 1, 1, 1, 4, 1, 4, 1, 4, 1] # if plant is low, then first plant. Otherwise, enlarge and plant.
	policy[2] = [4, 1, 4, 1, 4, 1, 3, 3, 2, 2, 2, 2] # if plant is low then, try to make it bigger either size and flw. Otherwise treat and control
	policy[3] = [3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2] # if there are few pollinators, do control to increase them. Otherwise treat
	policy[4] = [3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3] # if there are few pollinators with a lot flw, then do treat plants. Otherwise control	
	policy[5] = [2, 2, 1, 4, 3, 3, 1, 4, 1, 3, 1, 3] # something missing 1
	policy[6] = [1, 4, 2, 2, 1, 4, 1, 2, 3, 4, 3, 4] # something missing 2
	policy[7] = [3, 3, 2, 2, 4, 1, 4, 1, 4, 1, 4, 1] # if there are few pollinators with a lot flw, then do treat plants. Otherwise E/P
	return policy


def next_state(transition):
	'''
	states, threshold, count = len(transition), [], 0
	choose = randint(1, 100)
	for i in range(0, states):
		if count == 100.0:
			threshold.append(0)
		else:
			count += transition[i]*100
			threshold.append(count)
	for i in range(0, states):
		if choose <= threshold[i]:
			return i
	'''
	states, threshold, count = len(transition), [], 0
	return np.random.choice(range(states), 1, p=transition)[0]


# return reward and next state from taking action defined by the policy for the given state
def simulator(s, policy, expert, reward):
	action = policy[s]
	transition = expert[action][s]   # returns transition function [1, 0, 0, 0]
	s_next = next_state(transition)
	return s_next, reward[s_next]


def reward_init(states):
	rewards = np.zeros(states)
	'''
	for i in range(0, states):
		rewards[i] = 5 + i*5
	'''
	#rewards = [350,250,250,150,100,100,80,8060,40,40,20,20]
	rewards = [2,3,4,5,6,7,10,11,12,13,14,15]
	rewards = np.array(rewards)
	return rewards


def init_models(num_models, num_states):
	expert_models = defaultdict(list)
	ex0 = expert_init('MDP1.csv', num_states)
	ex1 = expert_init('MDP2.csv', num_states)
	ex2 = expert_init('MDP3.csv', num_states)
	ex3 = expert_init('MDP4.csv', num_states)
	expert_models[0] = ex0
	expert_models[1] = ex1
	expert_models[2] = ex2
	expert_models[3] = ex3
	policy = policy_init(num_states)
	rewards = reward_init(num_states)
	belief = np.zeros(num_models)

	for i in range(0, num_models):
		belief[i] = 1/float(num_models)

	return expert_models, policy, rewards, belief


if __name__ == '__main__':
	init_models(4, 12)
