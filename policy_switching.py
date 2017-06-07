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
num_states = 12 #4  # number of states
horizon = 20
w = 4  # run sim(s, policy, h) w times for each policy


# chooses expert model based on belief state distribution
# returns expert model
def _belief(b, num_models):
	threshold, count = [], 0
	choose = randint(1, 100)
	for i in range(0, num_models):
		count += b[i] * 100
		threshold.append(count)
	for i in range(0, num_models):
		if choose <= threshold[i]:
			return i


# performs policy switching on one expert model (chosen from belief distribution)
# returns best performing policy
def policy_switching(state, expert, w, policy, state_rewards):
	num_policies = policy.shape[0]
	value = np.zeros(num_policies)
	for i in range(0, num_policies):  # iterate through each policy
		val = 0
		for j in range(0, w):  # perform n
			val += sim.simulator(state, policy[i], expert, state_rewards)[1]
		value[i] = val

	return np.argmax(value)


# apply action from chosen policy (determined in policy switching) to all expert models
# returns next state, action taken to get to the next state & the optimal model
def update_models(state, expert_models, p, num_models, state_rewards):
	_rewards = np.zeros(num_models)
	next_state = np.zeros(num_models)
	for i in range(0, num_models):
		next_state[i], _rewards[i] = sim.simulator(state, p, expert_models[i], state_rewards)

	opt_model = np.argmax(_rewards)
	return int(next_state[opt_model]), opt_model, _rewards


# updates belief distribution based on rewards returned from update_models()
# returns updated belief distribution
def update_belief(belief, _rewards):
	norm_rewards = _rewards
	sum = np.sum(norm_rewards)
	norm_rewards = norm_rewards / sum
	belief = (belief + norm_rewards) / 2
	return belief


if __name__ == '__main__':
	state, output_policy, true_expert, final_policy = 0, [], [], []  # start in state 0 -- could change to another state
	reward_from_act = np.zeros(num_models)
	expert_models, policy, state_rewards, belief = sim.init_models(num_models, num_states)

	for h in range(0, horizon):
		ex = _belief(belief, num_models)  # returns expert model to use -- based on belief distribution
		p = policy_switching(state, expert_models[ex], w, policy, state_rewards)  # finds best policy from policy switching
		next_state, optimal_model, reward_from_act = update_models(state, expert_models, policy[p], num_models, state_rewards)  # applies policy to all experts
		update_belief(belief, reward_from_act)  # update belief

		output_policy.append((state, (p, int(policy[p][state])), int(next_state)))  # (state, (policy, action), next state)
		true_expert.append(optimal_model)  # returns expert model with highest reward after applying action from chosen policy
		final_policy.append(int(policy[p][state]))
		state = next_state

	print 'true_expert = ', true_expert
	print
	print 'output_policy = ', output_policy
	print
	print 'final policy = ', final_policy
