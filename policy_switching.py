# Final Project, Laurel Hopkins & Eugene Seo
# CS 533, Spring 2017

from collections import *
import numpy as np
from random import randint
import simulator as sim
import xlsxwriter
import csv
import matplotlib.pyplot as plt

'USER DEFINED PARAMETERS'
num_models = 2  # number of expert models
num_states = 4  # number of states
policy_length = 8
w = 4  # run sim(s, policy, h) w times for each policy


# chooses expert based on belief state distribution
def _belief(b, num_models):
	threshold, count = [], 0
	choose = randint(1, 100)
	for i in range(0, num_models):
		count += b[i] * 100
		threshold.append(count)
	for i in range(0, num_models):
		if choose <= threshold[i]:
			return i


def policy_switching(state, expert, w, policy, horizon, rewards):
	num_policies = policy.shape[0]
	value = np.zeros(num_policies)
	for i in range(0, num_policies):  # iterate through each policy
		val = 0
		for j in range(0, w):  # perform n
			val += sim.simulator(state, policy[i], horizon, expert, rewards)[1]
		value[i] = val

	return np.argmax(value)


def update_models(state, expert_models, p, horizon, running_value, num_models):
	value = np.zeros(num_models)
	next_state = np.zeros(num_models)
	for i in range(0, num_models):
		next_state[i], value[i] = sim.simulator(state, p, horizon, expert_models[i], rewards)
		# running_value[i] += value[i]
	opt_model = np.argmax(value)
	return next_state[opt_model], opt_model #, running_value


def update_belief(belief, opt_model):
	if optimal_model == 0:
		belief[0] += 0.05
		belief[1] -= 0.05
	return belief


if __name__ == '__main__':
	state, output_policy, true_expert = 0, [], []  # start in state 0 -- could change to another state
	running_value = np.zeros(num_models)
	expert_models, policy, rewards, belief = sim.init_models(num_models, num_states)

	for horizon in range(0, policy_length):
		ex = _belief(belief, num_models)  # returns expert model to use
		p = policy_switching(state, expert_models[ex], w, policy, horizon, rewards)  # best policy from policy switching
		next_state, optimal_model = update_models(state, expert_models, policy[p], horizon, running_value, num_models)  # applies policy to all experts
		output_policy.append((state, p, int(next_state)))
		true_expert.append(optimal_model)  # returns expert model with highest value after applying action from best policy
		update_belief(belief, optimal_model)  # update belief
		state = next_state
	# how do we want to update belief state? increment best expert and decrement others equally?

	print 'best_expert = ', true_expert
	print
	print 'output_policy = ', output_policy

