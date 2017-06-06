# Final Project, Laurel Hopkins & Eugene Seo
# CS 533, Spring 2017

from random import randint
from collections import *
import numpy as np


def expert0_init(states):
	ex0 = defaultdict(list)

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
	return ex0


def expert1_init(states):
	ex1 = defaultdict(list)

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
	return ex1


def policy_init(states):
	num_policies = 3
	policy = np.zeros((num_policies, states))
	policy[0] = [1, 1, 1, 1]
	policy[1] = [2, 2, 2, 2]
	policy[2] = [1, 2, 1, 2]
	return policy


def next_state(transition):
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


def simulator(s, policy, h, expert, reward):
	# return value from taking action defined by the policy for the given state and horizon
	action = policy[s]
	transition = expert[action][s]   # returns transition function [1, 0, 0, 0]
	s_next = next_state(transition)
	return s_next, reward[s_next]


def reward_init(states):
	rewards = np.zeros(states)
	for i in range(0, states):
		rewards[i] = 5 + i*5
	return rewards


def init_models(num_models, num_states):
	expert_models = defaultdict(list)
	ex0 = expert0_init(num_states)
	ex1 = expert1_init(num_states)
	expert_models[0] = ex0
	expert_models[1] = ex1
	policy = policy_init(num_states)
	rewards = reward_init(num_states)
	belief = np.zeros(num_models)

	for i in range(0, num_models):
		belief[i] = 1/float(num_models)

	return expert_models, policy, rewards, belief


if __name__ == '__main__':
	init_models(2, 4)
