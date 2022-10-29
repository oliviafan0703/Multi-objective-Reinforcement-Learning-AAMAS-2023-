'''Mixture policy approach baseline implementation: https://core.ac.uk/download/pdf/212996663.pdf'''
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np 
import time, math, random
from typing import Tuple
from collections import defaultdict
import itertools
import importlib
import json
import argparse
import time

import gym
import mo_gym

def run_Q_learning(episodes, alpha, epsilon, gamma, init_val, objective):
    """
    Q Learning to optimize ONE objective
    params: objective (int) index of the objective to maximize from
    """
    Q_table = np.zeros([env.observation_space, env.action_space.n])
    Q_table = Q_table + init_val
    r_acc = np.zeros(3)
    
    for i in range(1, episodes+1):
        state = env.reset()
        done = False
        
        while not done:
            if np.random.uniform(0, 1) < epsilon: action = env.action_space.sample()
            else: action = np.argmax(Q_table[state])
            
            next, reward, done, info= env.step(action)
            Q_table[state, action] = (1 - alpha)*Q_table[state, action] + alpha*(reward[objective]+gamma*np.max(Q_table[next]))
            state = next
            r_acc += reward
        print('Episode {}: {}'.format(i, r_acc))
    return Q_table

def greedy(vec, weights):
    arr = []
    for val in vec: arr.append(np.dot(weights, val))    # linear scalarization
    return np.argmax(arr)

def mixture_policy(episodes, timesteps, alpha, epsilon, gamma, init_val, dimension, weights_arr, interval, run, save=True):
    dims = [i for i in range(len(weights_arr))]
    policies = []
    for dim in dims:    # Obtain set of policies
        q = np.full([env.observation_space, env.action_space.n, dimension], init_val, dtype=float)
        policies.append(q)
    
    nsw_data, total_data = [], []
    for i in range(1, episodes+1):
        R_acc = np.zeros(dimension)
        state = env.reset()
        done = False
        count, dim, c = 0, 0, 0
        Q = policies[dim]
        weights = weights_arr[dim]
        
        while not done:
            if count > int(timesteps/dimension/interval):   # determines the period of changing policies
                dim += 1
                if dim >= dimension: dim = 0  # back to first objective after a "cycle"
                Q = policies[dim]
                weights = weights_arr[dim]
                count = 0   # change policy after t/d timesteps
            if np.random.uniform(0, 1) < epsilon: action = env.action_space.sample()
            else: action = greedy(Q[state], weights)
            
            next, reward, done, info = env.step(action)
            count += 1
            next_action = greedy(Q[next], weights)
            for j in range(len(Q[state, action])):
                # Q[state,action][j] = Q[state,action][j] + alpha*(reward[j]+gamma*Q[next,next_action][j]-Q[state,action][j])
                Q[state,action][j] = Q[state,action][j] + alpha*(reward[j]+gamma*Q[next,next_action][j]-Q[state,action][j])
            state = next
            R_acc += np.power(gamma, c)*reward
            c += 1
        
        R_acc = np.where(R_acc < 0, 0, R_acc) # Replace the negatives with 0
        nsw_score = np.power(np.product(R_acc), 1/len(R_acc))
        nsw_data.append(nsw_score)
        total_data.append(np.sum(R_acc))
        print('Episode {}\nAccumulated Discounted Reward: {}\nNSW: {}\n'.format(i, R_acc, nsw_score))
    if save == True:
        np.save(file='/Users/oliviafan/Desktop/RG_data3/mixture{}_no_gamma_nsw'.format(run),
                arr=nsw_data)
        np.save(file='/Users/oliviafan/Desktop/RG_data3/mixture{}_no_gamma_total'.format(run),
                arr=total_data)
    return np.mean(nsw_data)
    
def nsw(vec, nsw_lambda): 
    vec = vec + nsw_lambda
    vec = np.where(vec <= 0, nsw_lambda, vec)  # replace any negative values or zeroes with lambda
    return np.sum(np.log(vec))    # numpy uses natural log

def grid_search(episodes, timesteps, alpha, epsilon, gamma, init_val, dimension, weights_arr, intervals): 
    # search for the optimal interval of switching policies
    arr = []
    for val in intervals:
        start_time = time.time()
        avg_nsw = mixture_policy(episodes, timesteps, alpha, epsilon, gamma, 
                                 init_val, dimension, weights_arr, val, save=False)
        print('Average NSW{}\n Time Taken: {}\n'.format(avg_nsw, start_time-time.time()))
        arr.append(avg_nsw)
    np.save('mixutre_best_interval', intervals[np.argmax(arr)])
    np.save('mixture_all_interval', arr)

def generate_weights(dim, interval):
    if interval > 1 or interval < 0: raise ValueError('Interval out of range')
    if dim == 2:
        return [[round(i, 2), round(1.0-i, 2)] for i in np.arange(0, 1+interval, interval)]
    elif dim == 3:
        weights = []
        for i in np.arange(0, 1+interval, interval):
            rest = round(1 - i, 2)
            for j in np.arange(0, rest+interval, interval):
                tmp = [round(i, 2), round(j, 2), round(rest-j, 2)]
                weights.append(tmp)
        return weights
    elif dim == 4:
        weights = []
        for i in np.arange(0, 1+interval, interval):
            rest1 = round(1 - i, 2)
            for j in np.arange(0, rest1+interval, interval):
                rest2 = round(rest1-j, 2)
                for k in np.arange(0, rest2+interval, interval):
                    tmp = [round(i, 2), round(j, 2), round(k, 2), round(rest2-k, 2)]
                    weights.append(tmp)
        return weights

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Mixture Policy on Taxi""")
    prs.add_argument("-ep", dest="episodes", type=int, default=10000, required=False, help="episodes.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.7, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.1, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.999, required=False, help="Discount rate\n")
    prs.add_argument("-i", dest="init_val", type=int, default=0.1, required=False, help="Initial values\n")
    prs.add_argument("-gs", dest="size", type=int, default=10, required=False, help="Grid size\n")
    prs.add_argument("-d", dest="dimension", type=int, default=3, required=False, help="Dimension of reward\n")
    prs.add_argument("-it", dest="interval", type=float, default=0.01, required=False, help="Interval of weights\n")
    args = prs.parse_args()
    
    env = gym.make('resource-gathering-v0')
    obs = env.reset()  
    env.seed(1122)

    for i in range(32, 51):
        mixture_policy(args.episodes, 1000, args.alpha, args.epsilon, 
                        args.gamma, args.init_val, args.dimension, weights_arr =[[0.33,0.34,0.33],[0.33,0.33,0.34],[0.34,0.33,0.33]], run=i, interval = 2)
    # grid_search(500, args.alpha, args.epsilon, 
    #             args.gamma, args.init_val, args.dimension, [[0.21, 0.79],[1.0, 0.0]], np.arange(2,102,2))