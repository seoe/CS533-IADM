# -*- coding: utf-8 -*-
# CS533 Homework #3 - Spring 2017
# Eugene Seo (OSUID: 932981978)
# Due: 04-26-2017
# Description: Optimizing Infinite-Horizon Discounted Reward with Application to Optimal Parking

import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import MDP as mdp
import Agent as agent

# random policy 1
def policy_generator_1(state_size, T, R):
    prob_park = 0.2
    policy = np.random.choice(2, state_size, p=[1-prob_park, prob_park])
    return policy

# partical random policy 2
def policy_generator_2(state_size, T, R):
    prob_park = 0.2
    policy = np.zeros([state_size/4,4])
    policy[:, 0:2]= np.random.choice(2, state_size/2, p=[1-prob_park, prob_park]).reshape(state_size/4,2)
    policy = policy.reshape(1,state_size)[0]
    return policy

# my policy 1
def policy_generator_3(state_size, T, R): 
    ratio = (state_size/4 - 2)/5
    policy = np.zeros([state_size/4,4])
    policy[2:2+ratio, 0] = 1
    policy[policy.shape[0]-ratio:policy.shape[0], 0] = 1
    policy = policy.reshape(1,state_size)[0]
    return policy

# my policy 2
def policy_generator_4(state_size, T, R): 
    policy = np.zeros([state_size/4,4])
    policy[2:, 0] = 1 # policy 2
    policy = policy.reshape(1,state_size)[0]
    return policy

def basic_policies_simulation(myMDPs, max_trial, max_steps):
    for myMDP in myMDPs:
        myAgent = agent.Agent(myMDP)
        policy_list = []
        policy_list.append(policy_generator_1(myMDP.state_size, myMDP.T, myMDP.R))
        policy_list.append(policy_generator_2(myMDP.state_size, myMDP.T, myMDP.R))
        policy_list.append(policy_generator_3(myMDP.state_size, myMDP.T, myMDP.R))
        policy_list.append(policy_generator_4(myMDP.state_size, myMDP.T, myMDP.R))
        for policy in policy_list:
            myAgent.set_policy(policy)
            avg_reward = evaluation(myMDP, myAgent, max_trial, max_steps)
            print(avg_reward)

def evaluation(myMDP, myAgent, max_trial, max_steps):
    reward_list = []
    for i in range(0,max_trial):
        reward = simulation(myMDP, myAgent, max_steps)
        reward_list.append(reward)
    return np.average(reward_list)

def simulation(myMDP, myAgent, max_steps):
    myAgent.init_state(myMDP.init_state())
    myAgent.a = myAgent.policy[myAgent.s]

    trial_num = 1
    while(myAgent.a != 1 and trial_num < max_steps):
        myAgent.reward = myAgent.reward + myMDP.R[myAgent.s]
        myAgent.s = myMDP.next_state(myAgent.s, myAgent.a)
        myAgent.a = myAgent.policy[myAgent.s]
        trial_num = trial_num + 1

    if trial_num < max_steps:
        myAgent.reward = myAgent.reward + myMDP.R[myAgent.s]
        myAgent.next_s = myMDP.next_state(myAgent.s, myAgent.a)
        myAgent.reward = myAgent.reward + myMDP.R[myAgent.next_s] # when Park

    return myAgent.reward

def Q_learning(myMDP, myAgent, max_steps, T):
    beta = 1    
    lr = 0.9
    
    # step 2. take action from explore/exploit policy giving new state s'    
    trial_num = 1
    myAgent.a = myAgent.GLIE_policy(myAgent.s, T)
    while(myAgent.a != 1 and trial_num < max_steps):
        myAgent.next_s = myMDP.next_state(myAgent.s, myAgent.a)
        # step 3. perform TD update
        s = myAgent.s
        a = myAgent.a
        myAgent.Q[s,a] = myAgent.Q[s,a] + lr * (myMDP.R[s] + beta * max(myAgent.Q[myAgent.next_s[0]]) - myAgent.Q[s,a])
        myAgent.s = myAgent.next_s
        myAgent.a = myAgent.GLIE_policy(myAgent.s, T)
        trial_num = trial_num + 1

    if trial_num < max_steps:
        myAgent.next_s = myMDP.next_state(myAgent.s, myAgent.a)
        s = myAgent.s
        a = myAgent.a
        myAgent.Q[s,a] = myAgent.Q[s,a] + lr * (myMDP.R[s] + beta * max(myAgent.Q[myAgent.next_s[0]]) - myAgent.Q[s,a])
        
        myAgent.s = myAgent.next_s
        myAgent.a = myAgent.GLIE_policy(myAgent.s, T)
        myAgent.next_s = myMDP.next_state(myAgent.s, myAgent.a)
        s = myAgent.s
        a = myAgent.a
        myAgent.Q[s,a] = myAgent.Q[s,a] + lr * (myMDP.R[s] + beta * max(myAgent.Q[myAgent.next_s[0]]) - myAgent.Q[s,a])
        
    policy = np.argmax(myAgent.Q, axis=1)
    myAgent.set_policy(policy) # greedy policy for evaluation
    return myAgent

def reinforcement_learning(myMDPs, learning_num, max_trial, max_steps):
    name = ["myMDP1", "myMDP2"]
    name_idx = 0
    for myMDP in myMDPs:
        myAgent = agent.Agent(myMDP)
        myAgent.init_state(myMDP.init_state())
        improvement = []
        for i in range(2,learning_num):
            for j in range(0, max_trial):
                myAgent = Q_learning(myMDP, myAgent, max_steps, i) # learning phase
            avg_reward = evaluation(myMDP, myAgent, max_trial, max_steps)
            if i % 100 == 0:
                print("learning_num", i, "avg_reward", avg_reward)
            improvement.append(avg_reward)
        print(np.average(improvement))
        plt.plot(improvement)
        plt.title(name[name_idx])
        plt.xlabel('Learning Times')
        plt.ylabel('Performance (Average Reward)')
        plt.savefig(name[name_idx])
        plt.show()
        name_idx = name_idx + 1

def buildTwoMDPs():
    n = 10
    rewards = np.zeros(4)
    probabilities = np.zeros(2)
    myMDPs = []

    rewards[1] = penalty_handicap = -10
    rewards[2] = penalty_collision = -100
    probabilities[0] = prob_avail_handicap = 0.9
    probabilities[1] = prob_T = 5.0 # probability temperature: lower - most occupied, higher - most available

    """ 1st set of parameter values for less cost of driving and high reward for closest parking spot"""
    myMDP1 = mdp.MDP(n)
    MDPname = "myMDP1.txt"
    rewards[0] = penalty_driving = -1
    rewards[3] = best_reward = 100
    myMDP1.make_MDP(MDPname, rewards, probabilities)
    myMDPs.append(myMDP1)

    """ 2nd set of parameter values for high cost of driving and less reward for closest parking spot"""
    myMDP2 = mdp.MDP(n)
    MDPname = "myMDP2.txt"
    rewards[0] = penalty_driving = -10
    rewards[3] = best_reward = 10
    myMDP2.make_MDP(MDPname, rewards, probabilities)
    myMDPs.append(myMDP2)

    return myMDPs
    
def main():
    myMDPs = buildTwoMDPs()
    """ PART 2 """
    max_trial = 1000
    max_steps = 100
    basic_policies_simulation(myMDPs, max_trial, max_steps)
    """ PART 3 """
    learning_num = 500
    max_trial = 10
    reinforcement_learning(myMDPs, learning_num, max_trial, max_steps)


if __name__ == "__main__":
    main()