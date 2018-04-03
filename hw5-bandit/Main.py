import numpy as np
import Bandit as bd
import BanditAlg as alg
import matplotlib.pyplot as plt
import sys
import time
from collections import Counter

def generateBandit_1(num_arms):
    bandit = bd.Bandit(num_arms)
    for a in range(0,num_arms):
        if a < 9:
            bandit.SetParams(a, 0.05, 1)
        else:
            bandit.SetParams(a, 1, 0.1)
    print(bandit.param_arms)
    return bandit

def generateBandit_2(num_arms):
    bandit = bd.Bandit(num_arms)
    for a in range(0,num_arms):
        bandit.SetParams(a, (a+1)/float(num_arms), 0.1)
    print(bandit.param_arms)
    return bandit

def generateBandit_3(num_arms):
    bandit = bd.Bandit(num_arms)
    for a in range(0,num_arms):
        bandit.SetParams(a, 1, (a+1)/float(num_arms))
    print(bandit.param_arms)
    return bandit

def main():
    P_init = float(time.clock())
    bandit_num = int(sys.argv[1])

    if bandit_num == 1:
        bandit = generateBandit_1(10)
    elif bandit_num == 2:
        bandit = generateBandit_2(20)
    else:
        bandit = generateBandit_3(10)

    print("bandit", bandit_num)

    # variables initialization
    uniform_cr1_T = []
    uniform_sr1_T = []
    UCB_cr2_T = []
    UCB_sr2_T = []
    greedy_cr3_T = []
    greedy_sr3_T = []
    best_arm1_T = []
    best_arm2_T = []
    best_arm3_T = []

    # setting trials nums
    N = 2000000
    T1 = 5  # trials for bandit 1
    T2 = 10 # trials for bandit 2
    T3 = 10 # trials for bandit 3

    # plot variables initialization
    fig_num = 0
    x = np.arange(1,N+1,1)

    # Run Incremental Uniform algorithm
    for t in range(0, T1):
        T_init = float(time.clock())
        cumulative_regret1, simple_regret1, best_arm1 = alg.IncrementalUniform(bandit, N, t)
        uniform_cr1_T.append(cumulative_regret1)
        uniform_sr1_T.append(simple_regret1)
        best_arm1_T.append(best_arm1)
        T_end = float(time.clock())
        print("T", t, "time",  round(T_end-T_init,2))
        T_init = T_end
        '''
        fig_num += 1
        plt.figure(fig_num)
        plt.title(str(bandit_num) + "Bandit cumulative regret (Unifrom-" + str(t) +")")
        plt.plot(x, cumulative_regret1)
        plt.xlabel('Number of pulls')
        plt.ylabel('Cumulative Regret')
        plt.savefig("cumulative_regret_uniform_"+str(t)+".png", format='png')

        fig_num += 1
        plt.figure(fig_num)
        plt.title(str(bandit_num) + "Simple regret (Unifrom-" + str(t) +")")
        plt.plot(x, simple_regret1)
        plt.ylim([np.min(simple_regret1)-1,np.max(simple_regret1)+1])
        plt.xlabel('Number of pulls')
        plt.ylabel('Simple Regret')
        plt.savefig("simple_regret_uniform_"+str(t)+".png", format='png')
        '''
    print(best_arm1_T)
    cr1 = [round(np.mean(e),2) for e in zip(*uniform_cr1_T)]
    sr1 = [round(np.mean(e),2) for e in zip(*uniform_sr1_T)]

    # Plot the final graphs
    fig_num += 1
    plt.figure(fig_num)
    plt.title(str(bandit_num) + "Bandit final cumulative regret vs. number of pulls (Unifrom-final)")
    plt.plot(x, cr1)
    plt.xlabel('Number of pulls')
    plt.ylabel('Cumulative Regret')
    plt.savefig("cumulative_regret_uniform_"+str(bandit_num)+".png", format='png')

    fig_num += 1
    plt.figure(fig_num)
    plt.title(str(bandit_num) + "Bandit final simple regret vs. number of pulls (Unifrom-final)")
    plt.plot(x, sr1)
    plt.ylim([np.min(sr1)-1,np.max(sr1)+1])
    plt.xlabel('Number of pulls')
    plt.ylabel('Simple Regret')
    plt.savefig("simple_regret_uniform_"+str(bandit_num)+".png", format='png')


    # Run UCB algorithm
    for t in range(0, T2): #75s per T
        T_init = float(time.clock())
        cumulative_regret2, simple_regret2, best_arm2 = alg.UCB(bandit, N, t)
        UCB_cr2_T.append(cumulative_regret2)
        UCB_sr2_T.append(simple_regret2)
        best_arm2_T.append(best_arm2)
        T_end = float(time.clock())
        print("T", t, "time",  round(T_end-T_init,2))
        T_init = T_end
        '''
        fig_num += 1
        plt.figure(fig_num)
        plt.title(str(bandit_num) + "Bandit cumulative regret vs. number of pulls (UCB-" + str(t) +")")
        plt.plot(x, cumulative_regret2)
        plt.xlabel('Number of pulls')
        plt.ylabel('Cumulative Regret')
        plt.savefig("cumulative_regret_ucb_"+str(t)+".png", format='png')

        fig_num += 1
        plt.figure(fig_num)
        plt.title(str(bandit_num) + "Bandit simple regret vs. number of pulls (UCB-" + str(t) +")")
        plt.plot(x, simple_regret2)
        plt.ylim([np.min(simple_regret2)-1,np.max(simple_regret2)+1])
        plt.xlabel('Number of pulls')
        plt.ylabel('Simple Regret')
        plt.savefig("simple_regret_ucb_"+str(t)+".png", format='png')
        '''
    print(best_arm2_T)
    cr2 = [round(np.mean(e),2) for e in zip(*UCB_cr2_T)]
    sr2 = [round(np.mean(e),2) for e in zip(*UCB_sr2_T)]

    # Plot the final graphs
    fig_num += 1
    plt.figure(fig_num)
    plt.title(str(bandit_num) + "Bandit final cumulative regret vs. number of pulls (UCB-final)")
    plt.plot(x, cr2)
    plt.xlabel('Number of pulls')
    plt.ylabel('Cumulative Regret')
    plt.savefig("cumulative_regret_ucb_"+str(bandit_num)+".png", format='png')

    fig_num += 1
    plt.figure(fig_num)
    plt.title(str(bandit_num) + "Bandit final simple regret vs. number of pulls (UCB-final)")
    plt.plot(x, sr2)
    plt.ylim([np.min(sr2)-1,np.max(sr2)+1])
    plt.xlabel('Number of pulls')
    plt.ylabel('Simple Regret')
    plt.savefig("simple_regret_ucb_"+str(bandit_num)+".png", format='png')


    # Run E-Greedy algorithm
    for t in range(0, T3): #2m per T
        T_init = float(time.clock())
        cumulative_regret3, simple_regret3, best_arm3 = alg.EGreedy(bandit, N, 0.5, t)
        greedy_cr3_T.append(cumulative_regret3)
        greedy_sr3_T.append(simple_regret3)
        best_arm3_T.append(best_arm3)
        T_end = float(time.clock())
        print("T", t, "time",  round(T_end-T_init,2))
        T_init = T_end
        '''
        fig_num += 1
        plt.figure(fig_num)
        plt.title(str(bandit_num) + "Bandit cumulative regret vs. number of pulls (0.5-Greedy-" + str(t) +")")
        plt.plot(x, cumulative_regret3)
        plt.xlabel('Number of pulls')
        plt.ylabel('Cumulative Regret')
        plt.savefig("cumulative_regret_greedy_"+str(t)+".png", format='png')

        fig_num += 1
        plt.figure(fig_num)
        plt.title(str(bandit_num) + "Bandit simple regret vs. number of pulls (0.5-Greedy-" + str(t) +")")
        plt.plot(x, simple_regret3)
        plt.ylim([np.min(simple_regret3)-1,np.max(simple_regret3)+1])
        plt.xlabel('Number of pulls')
        plt.ylabel('Simple Regret')
        plt.savefig("simple_regret_greedy_"+str(t)+".png", format='png')
        '''
    print(best_arm3_T)

    cr3 = [round(np.mean(e),2) for e in zip(*greedy_cr3_T)]
    sr3 = [round(np.mean(e),2) for e in zip(*greedy_sr3_T)]

    # Plot the final graphs
    fig_num += 1
    plt.figure(fig_num)
    plt.title("Final cumulative regret vs. number of pulls (0.5-Greedy-final)")
    plt.plot(x, cr3)
    plt.xlabel('Number of pulls')
    plt.ylabel('Cumulative Regret')
    plt.savefig("cumulative_regret_greedy_"+str(bandit_num)+".png", format='png')

    fig_num += 1
    plt.figure(fig_num)
    plt.title("Simple regret vs. number of pulls (0.5-Greedy-final)")
    plt.plot(x, sr3)
    plt.ylim([np.min(sr3)-1,np.max(sr3)+1])
    plt.xlabel('Number of pulls')
    plt.ylabel('Simple Regret')
    plt.savefig("simple_regret_greedy_"+str(bandit_num)+".png", format='png')

    # plot graphs
    fig_num += 1
    plt.figure(fig_num)
    plt.title("Cumulative_regret vs. number of pulls")
    plt.gca().set_color_cycle(['black', 'blue', 'red'])
    plt.plot(x, cr1)
    plt.plot(x, cr2)
    plt.plot(x, cr3)
    plt.ylim([np.min(cr1+cr2+cr3)-1,np.max(cr1+cr2+cr3)+1])
    plt.xlabel('Number of pulls')
    plt.ylabel('Cumulative Regret')
    plt.legend(['IncrementalUniform', 'UCB', 'e-Greedy'], loc='upper right')
    plt.savefig("cumulative_regret_"+str(bandit_num)+".png", format='png')

    fig_num += 1
    plt.figure(fig_num)
    plt.title("Simple_regret vs. number of pulls")
    plt.gca().set_color_cycle(['black', 'blue', 'red'])
    plt.plot(x, sr1)
    plt.plot(x, sr2)
    plt.plot(x, sr3)
    plt.ylim([np.min(sr1+sr2+sr3)-1,np.max(sr1+sr2+sr3)+1])
    plt.xlabel('Number of pulls')
    plt.ylabel('Simple Regret')
    plt.legend(['IncrementalUniform', 'UCB', 'e-Greedy'], loc='upper right')
    plt.savefig("simple_regret_"+str(bandit_num)+".png", format='png')

    plt.close()

    P_end = float(time.clock())
    print("Total time",  round(P_end-P_init,2))


if __name__ == "__main__":
    main()