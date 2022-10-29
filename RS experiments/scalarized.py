'''NSW Q learning without R included in argmax, same initial values, with penalty environment v2'''
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np 
import time, math, random
from typing import Tuple
from collections import defaultdict
import itertools
import importlib
import json
import argparse

import gym
import mo_gym

def run_NSW_Q_learning(episodes, alpha, epsilon, gamma, nsw_lambda, init_val, dim_factor, non_stationary, run):
    Q = np.zeros([fair_env.observation_space, fair_env.action_space.n, 3], dtype=float)
    Q = Q + init_val
    nsw_data, total_data = [], []
    hist_data = []
        
    for i in range(1, episodes+1):
        R_acc = np.zeros(3)
        state = fair_env.reset()
        done = False
        avg = []
        c = 0
        
        while not done:
            if np.random.uniform(0,1) < epsilon:
                action = fair_env.action_space.sample()
            else:
                if non_stationary == True:
                    action = argmax_linear(R_acc, np.power(gamma,c)*Q[state], nsw_lambda)
                else:   # if stationary policy, then Racc doesn't affect action selection
                    action = argmax_linear(0, Q[state], nsw_lambda)
            next, reward, done, info = fair_env.step(action)
            
            max_action = argmax_linear(0, gamma*Q[next], nsw_lambda)

            Q[state, action] = Q[state, action] + alpha*(reward + gamma*Q[next, max_action] - Q[state, action])
            
            epsilon *= dim_factor  # epsilon diminish over time
            state = next
            R_acc += np.power(gamma,c)*reward
            c += 1

        R_acc = np.where(R_acc < 0, 0, R_acc) # Replace the negatives with 0
        nsw_score = np.power(np.product(R_acc), 1/len(R_acc))
        nsw_data.append(nsw_score)
        total = np.sum(R_acc)
        total_data.append(total)
        hist_data.append(R_acc)

    if non_stationary == False:
        np.save(file='/Users/oliviafan/Desktop/RG_histdata/NSW_Penalty_V2_run{}_noscalarized_gamma_nsw'.format(run),
                arr=nsw_data)
        np.save(file='/Users/oliviafan/Desktop/RG_histdata/NSW_Penalty_V2_run{}_noscalarized_gamma_hist'.format(run),
                arr=hist_data)
    else:
        np.save(file='/Users/oliviafan/Desktop/RG_histdata/NSW_Penalty_V2_run{}_scaled_gamma_nsw'.format(run),
                arr=nsw_data)
        np.save(file='/Users/oliviafan/Desktop/RG_histdata/NSW_Penalty_V2_run{}_scaled_gamma_hist'.format(run),
                arr=hist_data)
    print('FINISH TRAINING NSW Q LEARNING')

def argmax_linear(R, gamma_Q, nsw_lambda):
    sum = R + gamma_Q
    nsw_vals = [scalarized(sum[i], nsw_lambda) for i in range(fair_env.action_space.n)]
    if np.all(nsw_vals == nsw_vals[0]) == True: # if all values are same, random action
        # numpy argmax always return first element when all elements are same
        action = fair_env.action_space.sample()
    else:
        action = np.argmax(nsw_vals)
    return action

def scalarized(vec, nsw_lambda): 
    vec = vec + nsw_lambda
    vec = np.where(vec <= 0, nsw_lambda, vec)  # replace any negative values or zeroes with lambda
    return np.sum(vec)

if __name__ == '__main__':
    
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""NSW Q-learning on Taxi""")
    prs.add_argument("-ep", dest="episodes", type=int, default=10000, required=False, help="episodes.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.7, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-aN", dest="alpha_N", type=bool, default=False, required=False, help="Whether use 1/N for alpha\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.1, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.999, required=False, help="Discount rate\n")
    prs.add_argument("-nl", dest="nsw_lambda", type=float, default=1e-4, required=False, help="Smoothing factor\n")
    prs.add_argument("-i", dest="init_val", type=int, default=0.1, required=False, help="Initial values\n")
    prs.add_argument("-d", dest="dim_factor", type=float, default=0.9, required=False, help="Diminish factor for epsilon\n")
    prs.add_argument("-ns", dest="non_stat", type=bool, default=True, required=False, help="Whether non-stationary policy\n")
    args = prs.parse_args()
    
    # size = args.size
    # loc_coords = args.loc_coords
    # dest_coords = args.dest_coords
    # fuel = args.fuel
    
    fair_env = gym.make('resource-gathering-v0')
    obs = fair_env.reset()   

    # fair_env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel, 
    #                         output_path='Taxi_MDP/NSW_Q_learning/run_', fps=4)
    # fair_env.seed(1122)
    
    for i in range(1,51):   # Do 50 runs, and then take the average
        run_NSW_Q_learning(episodes=args.episodes, alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma, 
                           nsw_lambda=args.nsw_lambda, init_val=args.init_val, non_stationary=args.non_stat, 
                           dim_factor=args.dim_factor, run=i)
    
    
    


    