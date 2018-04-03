import numpy as np
import time

def IncrementalUniform(bandit, num_pulls, T):
    # variable inits
    num_arms = bandit.NumArms()
    num_pulls_arm = np.zeros(num_arms)
    avg_rewards = np.zeros(num_arms)
    total_rewards = 0
    best_arm = 0
    best_avg_reward = 0
    cr_list = []
    sr_list = []
    cr = 0
    sr = 0

    t0 = float(time.clock())
    for n in range(1, num_pulls+1):
        arm_index = n % num_arms - 1
        num_pulls_arm[arm_index] += 1
        r = bandit.Pull(arm_index)
        total_rewards += r

        old_mean = avg_rewards[arm_index]
        new_mean = old_mean + (r-old_mean)/num_pulls_arm[arm_index]
        avg_rewards[arm_index] = new_mean
        if new_mean > best_avg_reward:
            best_avg_reward = new_mean
            best_arm = arm_index

        sr = bandit.opt_reward - best_avg_reward
        sr_list.append(sr)

        cr = n * bandit.opt_reward - total_rewards
        cr_list.append(cr)

        if n%200000 == 0:
            t1 = float(time.clock())
            print("T", T, "Uni", "pull", n, "time",  round(t1-t0,2))
            t0 = t1

    if best_arm == -1:
        best_arm = num_arms
    return cr_list, sr_list, best_arm+1


def UCB(bandit, num_pulls, T):
    num_arms = bandit.NumArms()
    num_pulls_arm = np.zeros(num_arms) #n(a)
    avg_rewards = np.zeros(num_arms) #Q(a)
    total_rewards = 0
    best_arm = 0
    best_avg_reward = 0
    cr_list = []
    sr_list = []
    cr = 0
    sr =0

    t0 = float(time.clock())
    for i in range(1, num_arms+1):
        i = i - 1 # arm_index
        num_pulls_arm[i] += 1
        r = bandit.Pull(i)
        total_rewards += r

        old_mean = avg_rewards[i]
        new_mean = old_mean + (r-old_mean)/num_pulls_arm[i]
        avg_rewards[i] = new_mean
        if new_mean > best_avg_reward:
            best_avg_reward = new_mean
            best_arm = i

        sr = bandit.opt_reward - best_avg_reward
        sr_list.append(sr)

        cr = i * bandit.opt_reward - total_rewards
        cr_list.append(cr)

    for n in range(num_arms, num_pulls):
        ucb = avg_rewards + np.sqrt(2*np.log(n)/num_pulls_arm)
        arm_index = np.argmax(ucb)
        num_pulls_arm[arm_index] += 1
        r = bandit.Pull(arm_index)
        total_rewards += r

        old_mean = avg_rewards[arm_index]
        new_mean = old_mean + (r-old_mean)/num_pulls_arm[arm_index]
        avg_rewards[arm_index] = new_mean
        if new_mean > best_avg_reward:
            best_avg_reward = new_mean
            best_arm = arm_index

        sr = bandit.opt_reward - best_avg_reward
        sr_list.append(sr)

        cr = n * bandit.opt_reward - total_rewards
        cr_list.append(cr)

        if n%200000 == 0:
            t1 = float(time.clock())
            print("T", T, "UCB", "pull", n, "time",  round(t1-t0,2))
            t0 = t1

    return cr_list, sr_list, best_arm+1


def EGreedy(bandit, num_pulls, E, T):
    num_arms = bandit.NumArms()
    num_pulls_arm = np.zeros(num_arms)
    avg_rewards = np.zeros(num_arms)
    total_rewards = 0
    best_arm = 0
    other_arms = range(1,num_arms)
    best_avg_reward = 0
    op_list = ["best", "random"]
    cr_list = []
    sr_list = []
    cr = 0
    sr = 0
    other_probs = [E/(num_arms-1)] * (num_arms-2)
    prob = [E] + other_probs + [1-E-sum(other_probs)]

    t0 = float(time.clock())
    for n in range(1, num_pulls+1):
        arms_order = [best_arm] + other_arms
        arm_index = np.random.choice(arms_order, 1, p=prob)[0]
        num_pulls_arm[arm_index] += 1
        r = bandit.Pull(arm_index)
        total_rewards += r

        old_mean = avg_rewards[arm_index]
        new_mean = old_mean + (r-old_mean)/num_pulls_arm[arm_index]
        avg_rewards[arm_index] = new_mean
        if new_mean > best_avg_reward:
            best_avg_reward = new_mean
            best_arm = arm_index
            other_arms = range(num_arms)
            other_arms.remove(best_arm)

        sr = bandit.opt_reward - best_avg_reward
        sr_list.append(sr)

        cr = n * bandit.opt_reward - total_rewards
        cr_list.append(cr)

        if n%200000 == 0:
            t1 = float(time.clock())
            print("T", T, "Gre", "pull", n, "time",  round(t1-t0,2))
            t0 = t1

    return cr_list, sr_list, best_arm+1