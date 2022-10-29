'''For visualization of results from taxi environments'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import scipy.stats as ss

def plot_lines(data, type, ID, title, x_range=None, linewidth=0.3, alpha=0.9, colors=['r'], labels=None, subplot=False):
    if len(data)==1:
        if x_range == None:
            interval = np.arange(1, len(data[0])+1)
        else:
            interval = np.arange(1, x_range+1)
            data = data[0][:x_range]
        plt.plot(interval, data, colors[0], linewidth=linewidth, alpha=alpha)
        plt.xlabel('Intervals (each interval is 10000 steps)')
        plt.ylabel(type)
        # plt.title('Q value loss over time for job {}'.format(ID))
        plt.title(title, fontweight='bold')
        plt.show()
    elif subplot == True:
        if x_range == None:
            interval = np.arange(1, len(data)+1)
        else:
            interval = np.arange(1, x_range+1)
            for i in range(len(data)): data[i] = data[i][:x_range]
        fig, axs = plt.subplots(len(data))
        for i in range(len(data)):
            axs[i].plot(interval, data[i], colors[i], linewidth=linewidth, alpha=alpha)
            axs[i].set_title(labels[i])
        #axs.set(xlabel='Intervals (each interval is 10000 steps)', ylabel='Q value loss')
        # plt.title('Q value loss across different experiments')
        fig.suptitle('Q value loss over time for 3 pickup locations during learning', fontweight='bold')
        fig.tight_layout()
        plt.show()
    else:
        if x_range == None:
            interval = np.arange(1, len(data)+1)
        else:
            interval = np.arange(1, x_range+1)
            for i in range(len(data)): data[i] = data[i][:x_range]
        for i in range(len(data)):
            plt.plot(interval, data[i], colors[i], linewidth=linewidth, alpha=alpha, label=labels[i])
            alpha -= 0.1
        plt.xlabel('Intervals (each interval is 10000 steps)')
        plt.ylabel('{}'.format(type))
        # plt.title('Q value loss across different experiments')
        plt.title(title, fontweight='bold')
        plt.legend()
        plt.show()

def plot_total(IDs, x_range, linewidth, alpha, colors, labels, num_locs=3):
    data = []
    for id in IDs: 
        arr = find_data(id, num_locs)
        data.append(arr)
    if x_range == None:
        interval = np.arange(1, len(data[0])+1)
    else:
        interval = np.arange(1, x_range+1)
        for i in range(len(data)): data[i] = data[i][:x_range]
    fig, axs = plt.subplots(len(data))
    for i in range(len(data)):
        axs[i].plot(interval, data[i], colors[i], linewidth=linewidth, alpha=alpha)
        axs[i].set_title(labels[i])
    fig.suptitle('Total Accumulated Reward Over Time', fontweight='bold')
    fig.tight_layout()
    plt.show()

def find_data(ID, num_locs):
    with open('Experiments/log/{}.out'.format(ID)) as f:
        text = f.readlines()
    text = ''.join(text)
    matches = re.findall('\[.*\]', text)
    for i in range(len(matches)): 
        matches[i] = matches[i].strip('][')
        matches[i] = np.fromstring(matches[i], sep=' ')
        matches[i] = np.sum(matches[i])
    return matches

def plot_r_acc(data, labels, width, ):
    locs = []
    for i in range(len(data)): data[i] = np.mean(data[i], axis=0)
    for i in range(len(data[0])):
        arr = []
        for j in range(len(data)):
            arr.append(data[j][i])
        locs.append(arr)
    
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    
    for i in range(len(data[0])):
        ax.bar(x - width*(1-i), locs[i], width, label='Location {}'.format(i))
    
    ax.set_ylabel('Average Accumulated Reward')
    ax.set_title('Average Accumulated Reward Between Algorithms', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    return plt.show()

def nsw(vec): 
    return np.sum(np.log(vec))    # numpy uses natural log

def plot_nsw(data, labels, width, nsw_lambda=1e-4):
    nsws = []
    for i in range(len(data)):
        nsw = []
        for j in range(50):
            num = np.product(data[i][j])
            nsw.append(np.power(num, 1/len(data[i][j])))
        nsws.append(np.mean(nsw))
    
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    
    ax.bar(x, nsws, width)
    ax.set_ylabel('Average NSW Score')
    ax.set_title('Average NSW Score for Different Algorithms', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    fig.tight_layout()
    return plt.show()

def plot_pareto(data):
    arr1, arr2 = [],[]
    if type(data) == dict:
        for i in data.values():
            arr1.append(i[0])
            arr2.append(i[1])
    else:
        for i in data:
            arr1.append(i[0])
            arr2.append(i[1])
    plt.title('Pareto Front approxiamted with linear weights for 2 dimensions', fontweight='bold')
    plt.ylabel('Average Accumulated Reward')
    plt.xlabel('Average Accumulated Reward')
    plt.scatter(arr1, arr2)
    plt.show()

def average_50(criterion, runs, confidence_level, criteria='nsw'):
    '''Plot average over 50 runs result (for figures in the paper)'''
    for i in range(1, runs+1):
        data = np.load('Experiments/taxi_q_tables_V2/NSW_Penalty_V2_size{}_locs{}_run{}_1_{}.npy'.format(criterion[0], criterion[1], i, criteria))
        data2 = np.load('Experiments/taxi_q_tables_V2/scalarized_ql_nsw_size{}_locs{}_{}_{}.npy'.format(criterion[0], criterion[1], i, criteria))
        data3 = np.load('Experiments/taxi_q_tables_stationary/NSW_Penalty_V2_size{}_locs{}_run{}_1_no_gamma_{}.npy'.format(criterion[0], criterion[1], i, criteria))
        data4 = np.load('Experiments/taxi_q_tables_V2/mixture_size{}_locs{}_run{}_{}.npy'.format(criterion[0], criterion[1], i, criteria))
        if i == 1: 
            total = np.array(data.tolist())
            total2 = np.array(data2.tolist())
            total3 = np.array(data3.tolist())
            total4 = np.array(data4.tolist())
        else:
            total = np.column_stack((total, data.tolist()))
            total2 = np.column_stack((total2, data2.tolist()))
            total3 = np.column_stack((total3, data3.tolist()))
            total4 = np.column_stack((total4, data4.tolist()))
        
    mean, upper_ci, lower_ci = [], [], []
    mean2, upper_ci2, lower_ci2 = [], [], []
    mean3, upper_ci3, lower_ci3 = [], [], []
    mean4, upper_ci4, lower_ci4 = [], [], []
    for i in range(len(total)): 
        avg = np.mean(total[i])
        mean.append(avg)
        lower, upper = ss.t.interval(alpha=confidence_level, df=len(total[i])-1, loc=avg, scale=ss.sem(total[i]))
        upper_ci.append(upper)
        lower_ci.append(lower)
        
        avg2 = np.mean(total2[i])
        mean2.append(avg2)
        lower2, upper2 = ss.t.interval(alpha=confidence_level, df=len(total2[i])-1, loc=avg2, scale=ss.sem(total2[i]))
        upper_ci2.append(upper2)
        lower_ci2.append(lower2)
        
        avg3 = np.mean(total3[i])
        mean3.append(avg3)
        lower3, upper3 = ss.t.interval(alpha=confidence_level, df=len(total3[i])-1, loc=avg3, scale=ss.sem(total3[i]))
        upper_ci3.append(upper3)
        lower_ci3.append(lower3)
        
        avg4 = np.mean(total4[i])
        mean4.append(avg4)
        lower4, upper4 = ss.t.interval(alpha=confidence_level, df=len(total4[i])-1, loc=avg4, scale=ss.sem(total4[i]))
        upper_ci4.append(upper4)
        lower_ci4.append(lower4)
        
    fig, ax = plt.subplots()
    timesteps = np.arange(1,5001)
    ax.plot(timesteps, mean, color='dodgerblue', linewidth=1, label='Non-stationary Policy')
    ax.fill_between(timesteps, lower_ci, upper_ci, alpha=0.3, color='dodgerblue', linewidth=0.01)
    
    ax.plot(timesteps, mean2, color='orange', linewidth=1, label='Optimized Linear Scalarization')
    ax.fill_between(timesteps, lower_ci2, upper_ci2, alpha=0.3, color='purple', linewidth=0.01)
    
    ax.plot(timesteps, mean3, color='red', linewidth=1, label='Stationary Policy')
    ax.fill_between(timesteps, lower_ci3, upper_ci3, alpha=0.3, color='red', linewidth=0.01)
    
    ax.plot(timesteps, mean4, color='purple', linewidth=1, label='Optimal Mixture Policy')
    ax.fill_between(timesteps, lower_ci4, upper_ci4, alpha=0.3, color='purple', linewidth=0.01)
    
    plt.xlabel('Episodes (10000 timesteps per episode)', fontweight='bold')
    if criteria == 'nsw': plt.ylabel('NSW Score', fontweight='bold')
    else: plt.ylabel('Utilitarian Reward', fontweight='bold')
    plt.legend()
    plt.show()

def average_50_loss(criterion, runs, confidence_level):
    '''Plot average over 50 runs result (for figures in the paper)'''
    for i in range(1, runs+1):
        data = np.load('taxi_q_tables_V2/NSW_Penalty_V2_size{}_locs{}_run{}1_loss.npy'.format(criterion[0], criterion[1], i))
        if i == 1: 
            total = np.array(data.tolist())
        else:
            total = np.column_stack((total, data.tolist()))
        
    mean, upper_ci, lower_ci = [], [], []
    for i in range(len(total)): 
        avg = np.mean(total[i])
        mean.append(avg)
        lower, upper = ss.t.interval(alpha=confidence_level, df=len(total[i])-1, loc=avg, scale=ss.sem(total[i]))
        upper_ci.append(upper)
        lower_ci.append(lower)
        
    fig, ax = plt.subplots()
    timesteps = np.arange(1,5001)
    ax.plot(timesteps, mean, color='red', linewidth=1, label='Non-stationary Policy')
    ax.fill_between(timesteps, lower_ci, upper_ci, alpha=0.3, color='red', linewidth=0.01)
    
    plt.xlabel('Episodes (10000 timesteps per episode)', fontweight='bold')
    plt.ylabel('L1 Loss', fontweight='bold')
    plt.title('Result from {} runs'.format(runs))
    plt.legend()
    plt.show()

def individual_loss(criterion, runs):
    for i in range(1, runs+1):
        data = np.load('taxi_q_tables_V2/NSW_Penalty_V2_size{}_locs{}_run{}1_loss.npy'.format(criterion[0], criterion[1], i))
        plt.plot(data)
        plt.xlabel('Episodes (10000 timesteps per episode)', fontweight='bold')
        plt.ylabel('L1 Loss', fontweight='bold')
        plt.title('Run {}'.format(i), fontweight='bold')
        plt.show()

if __name__ == '__main__':
    '''Plot Q value loss over time (learning)'''
    # data1 = np.load('Experiments/taxi_q_tables_V2/QL_Penalty_size10_locs4_nsw.npy')
    # data2 = np.load('Experiments/taxi_q_tables_V2/NSW_Penalty_V2_size10_locs4_1_nsw.npy')
    # data3 = np.load('Experiments/taxi_q_tables_V2/NSW_Penalty_V2_size10_locs4_stat_1_nsw.npy')
    # data = [data1, data2, data3]
    # labels = ['Standard Q Learning', 'Nash Q Learning, nonstatioanry', 'Nash Q Learning, stationary']
    # plot_lines(data=data, type='NSW Score', title='Performance in 10X10 grid, 4 dimensions', 
    #            ID=1932259, colors=['r',
    # 'b','g','purple','y'], x_range=10000, 
    #            linewidth=0.8, labels=labels, alpha=0.8, subplot=False)
    
    '''Plot Total Reward over Time (learning)'''
    # id = [1940069, 1940073,1940077]
    # plot_total(IDs=id, x_range=10000, linewidth=0.8, alpha=0.8,
    #            colors=['r','b','g'], labels=['8X8 Grid','10X10 Grid','12X12 Grid'])
    
    '''Plot accumulated reward'''
    # data1 = np.load('Experiments/ql_Racc_6.npy')
    # data2 = np.load('Experiments/nsw_6.npy')
    # data3 = np.load('Experiments/stationary_nsw_6.npy')
    # data = [data1, data2, data3]
    # labels = ['Standard Q learning', 'NSW non-stationary policy', 'NSW stationary policy']
    #plot_r_acc(data, labels=labels, width=0.2)
    
    '''Plot NSW'''
    # data1 = np.load('Experiments/ql_Racc_6.npy')
    # data2 = np.load('Experiments/nsw_6.npy')
    # data3 = np.load('Experiments/stationary_nsw_6.npy')
    # data = [data1, data2, data3]
    # labels = ['Standard Q learning', 'NSW non-stationary policy', 'NSW stationary policy']
    # plot_nsw(data=data, labels=labels, width=0.2)
    
    '''Plot Pareto Front'''
    # data1 = np.load('Experiments/all_points_size10_dim2.npy')
    # data2 = np.load('Experiments/pareto_front_size10_dim2.npy', allow_pickle=True).tolist()
    # print(data2)
    # plot_pareto(data2)
    
    '''Generate figures for 50 runs'''
    average_50(criterion=[10,2], runs=50, confidence_level=0.95, criteria='nsw')
    # average_50_loss(criterion=[10,2], runs=50, confidence_level=0.95)
    # individual_loss(criterion=[10,2], runs=50)