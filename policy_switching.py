# Final Project, Laurel Hopkins & Eugene Seo
# CS 533, Spring 2017

from collections import *
import numpy as np
from random import randint
import simulator as sim


'USER DEFINED PARAMETERS'
num_models = 2  # number of expert models
num_states = 12  # number of states
horizon = 20
w = 4  # run sim(s, policy, h) w times for each policy
h = 5
real_world_expert = 0  # must be a one of the expert models

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
def policy_switching(start_state, experts, num_models, w, h, policy, state_rewards, _belief):
	num_policies = policy.shape[0]
	reward_pol_ex = np.zeros((num_policies, num_models))
	for pol in range(0, num_policies):  # iterate through each policy
		for m in range(0, num_models):  # iterate through each expert
			ave_reward = 0  # holds average reward of expert after h policy evaluations
			for _h in range(0, h):  # run each policy on expert h times
				val, val_temp, state = 0, 0, start_state
				for j in range(0, w):  # perform w steps of policy rollout
					next_s, val_temp = sim.simulator(state, policy[pol], experts[m], state_rewards)
					val += val_temp
					state = next_s
				ave_reward += val
			reward_pol_ex[pol][m] = ave_reward / h  # ave reward after h calls

	# determine best policy based on p(m1)*R11 + p(m2)*R12 > or < p(m1)*R21 + p(m2)*R22
	scaled_val = np.zeros(num_policies)
	for i in range(0, num_policies):
		scaled_val[i] = np.sum((reward_pol_ex[i] * _belief))
	return np.argmax(scaled_val)


# apply action from chosen policy (determined in policy switching) to all expert models
# returns next state, action taken to get to the next state & the optimal model
def update_models(state, expert_models, p, num_models, state_rewards, real_world_expert):
	_rewards = np.zeros(num_models)
	next_state = np.zeros(num_models)
	for i in range(0, num_models):
		next_state[i], _rewards[i] = sim.simulator(state, p, expert_models[i], state_rewards)

	# opt_model = np.argmax(_rewards)
	# return int(next_state[opt_model]), opt_model, _rewards
	return int(next_state[real_world_expert]), _rewards


# updates belief distribution based on rewards returned from update_models()
# returns updated belief distribution
def update_belief(belief, _rewards):
	norm_rewards = _rewards
	sum = np.sum(norm_rewards)
	norm_rewards = norm_rewards / sum
	belief = (belief + norm_rewards) / 2
	return belief


if __name__ == '__main__':
	state, output_policy, final_policy = 0, [], []  # start in state 0 -- could change to another state
	reward_from_act = np.zeros(num_models)
	expert_models, policy, state_rewards, belief = sim.init_models(num_models, num_states)

	for hor in range(0, horizon):
		# ex = _belief(belief, num_models)  # returns expert model to use -- based on belief distribution
		p = policy_switching(state, expert_models, num_models, w, h, policy, state_rewards, belief)  # finds best policy from policy switching
		next_state, reward_from_act = update_models(state, expert_models, policy[p], num_models, state_rewards, real_world_expert)  # applies policy to all experts
		belief = update_belief(belief, reward_from_act)  # update belief

		output_policy.append((state, (p, int(policy[p][state])), int(next_state)))  # (state, (policy, action), next state)
		final_policy.append(int(policy[p][state]))
		state = next_state


	print
	print 'output_policy = ', output_policy
	print
	print 'final policy = ', final_policy
