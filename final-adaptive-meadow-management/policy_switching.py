# Final Project, Laurel Hopkins & Eugene Seo
# CS 533, Spring 2017

from collections import *
import numpy as np
from random import randint
import simulator as sim
import matplotlib.pyplot as plt


'USER DEFINED PARAMETERS'
num_models = 4  # number of expert models
num_states = 12  # number of states
horizon = 15
w = 20  # run sim(s, policy, h) w times for each policy
h = 10
real_world_expert = 1  # must be a one of the expert models

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
			ave_reward = 0  # holds average reward of expert after w policy evaluations
			for _h in range(0, w):  # run each policy on expert w times
				val, val_temp, state = 0, 0, start_state
				for j in range(0, h):  # perform h steps of policy rollout
					next_s, val_temp = sim.simulator(state, policy[pol], experts[m], state_rewards)
					val += val_temp
					state = next_s
				ave_reward += val
			reward_pol_ex[pol][m] = ave_reward / w  # ave reward after w calls

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

	return int(next_state[real_world_expert]), _rewards


def update_models2(belief, state, expert_models, p, num_models, state_rewards):
	_rewards = np.zeros(num_models)
	next_state = np.zeros(num_models)
	for i in range(0, num_models):
		next_state[i], _rewards[i] = sim.simulator(state, p, expert_models[i], state_rewards)
	return int(np.random.choice(next_state, 1, p=belief)[0])


def update_models3(belief, state, expert_models, p, num_models, state_rewards):
	transitions = np.zeros((num_models, num_states))
	for i in range(0, num_models):
		action = p[state]
		transitions[i] = expert_models[i][action][state] * belief[i]
	transition = sum(transitions)
	return int(np.random.choice(range(num_states), 1, p=transition)[0])


# updates belief distribution based on rewards returned from update_models()
# returns updated belief distribution
def update_belief(belief, expert_models, num_models, state, action, next_state):
	expert_models_prob = np.zeros(num_models)
	for i in range(0, num_models):
		expert_models_prob[i] = expert_models[i][action][state][next_state]
	accu_blief = belief + np.multiply(expert_models_prob,belief)	
	belief = accu_blief / sum(accu_blief)
	return belief


if __name__ == '__main__':
	state, output_policy, final_policy, transition_policy = 0, [], [], []  # start in state 0 -- could change to another state
	reward_from_act = np.zeros(num_models)
	belief_history = np.zeros((horizon+1, num_models))	
	expert_models, policy, state_rewards, belief = sim.init_models(num_models, num_states)
	belief_history[0] = belief
	trajectory = np.zeros(horizon+1)
	trajectory[0] = state+1
	
	for hor in range(0, horizon):
		p = policy_switching(state, expert_models, num_models, w, h, policy, state_rewards, belief)  # finds best policy from policy switching		
		action = policy[p][state]
		next_state = update_models2(belief, state, expert_models, policy[p], num_models, state_rewards)
		trajectory[hor+1] = next_state+1
		belief = update_belief(belief, expert_models, num_models, state, action, next_state)  # update belief
		belief_history[hor+1] = belief		
		
		output_policy.append((state+1, (p+1, int(policy[p][state])), int(next_state+1)))  # (state, (policy, action), next state)
		transition_policy.append((state+1, int(policy[p][state]), int(next_state+1)))
		final_policy.append(int(action))		
		state = next_state


	print
	print 'output_policy = ', output_policy
	print
	print 'transition_policy = ', transition_policy
	print
	print 'final policy = ', final_policy
	print
	print 'trajectory = ', trajectory
	print
	print 'belief_history = \n', belief_history
	
	
	plt.figure(1)
	plt.title('The belief in each expert model over time')
	plt.xlabel('Time')
	plt.ylabel('Belief in expert model')
	plt.plot(belief_history[:,0], label='Expert 1', marker='*')
	plt.plot(belief_history[:,1], label='Expert 2', marker='o')
	plt.plot(belief_history[:,2], label='Expert 3', marker='^')
	plt.plot(belief_history[:,3], label='Expert 4', marker='s')	
	plt.legend(loc='upper left')
	plt.grid(True)

	plt.figure(2)
	plt.title('Real world state transition via action')
	plt.xlabel('Action')
	plt.ylabel('State')
	x = range(horizon+1)
	my_xticks = [0]+final_policy
	plt.xticks(x, my_xticks)
	plt.ylim(0,12)
	plt.plot(x, trajectory, marker='o')	
	plt.grid(True)
	plt.show()

	plt.close()