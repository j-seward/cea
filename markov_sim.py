#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:12:15 2019

@author: Jonathan Seward

Program: This program runs a simple cost effectiveness analysis comparing two 
    strategies with three health states. It can be augmented to compare more 
    than two strategies and any number of health states. It should also be 
    noted that this program is a work in progress and was my first foray into 
    Python. So please let me know if there are any bugs or weird output.
    
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Header
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A function that implements the Markov model to forecast the state
def cohort_sim(trans_mat, cost_mat, util_mat, 
               time=10, start_props=([1,0,0]), discount_rate = 0, 
               strategy = 'Strategy 1'):
    # Check to see if the transition probabilities make sense
    if int(np.sum(trans_mat)) != float(num_states):
        print("Your transition probabilities do not sum to 1.")
    else: print("Your transition matrix passes the sniff test. :)")
    
    costs = np.dot(cost_mat,start_props)
    utils = np.dot(util_mat,start_props)
    disc_cost = costs
    disc_util = utils
    i = 0
    proportion = start_props
    results = np.append(proportion,[i,costs,disc_cost,utils,disc_util])
    results_array = results
    while i != time:
        # calculates the proportion of people in each state
        proportion = np.dot(proportion,trans_mat)
        
        # calculates the cost, util, and C/E in time i (incremental)
        current_cost = np.dot(proportion,cost_mat)
        current_util = np.dot(proportion,util_mat)
       
        # calculates cumulative cost
        costs += current_cost
        utils += current_util
       
        # increments time counter
        i += 1
        
        # calculates discounted costs and utils
        if discount_rate != 0:
            disc_cost = ((1-discount_rate)**i)*current_cost
            disc_util = ((1-discount_rate)**i)*current_util
            
        # creates list of all incremental results used to create dataframe
        results = np.append(proportion,[i,current_cost,disc_cost,current_util,disc_util])
        results_array = np.vstack((results_array,results))
    
    # total cost effectiveness
    ce = costs/utils
    
    # Dataframe of all incremental results
    df = pd.DataFrame(results_array,columns = states+["Time","Costs","Discounted Costs","Utils","Discounted Utils"])
    df.name = strategy
    return costs, utils, ce, df

# ICER calcuation function
def calc_icer(base_cost, compare_cost, base_effect, compare_effect):
    icer = (compare_cost-base_cost)/(compare_effect-base_effect)
    return icer

# Net Monetary Benefits calculation function
def nmb(effectiveness, costs, wtp):
    nmb = effectiveness*wtp - costs
    return nmb

# Cycle corrections function: within cycle or half cycle
def cycle_correct(col, cycle_type = 'hcc'):
    if cycle_type == 'hcc':
        col[0] = col[0]*.5
        col[-1] = col[-1]*.5
    elif cycle_type == 'wcc':
        col2 = col[1:len(col)].append(pd.Series([0])).reset_index(drop=True)
        col3 = col*.5+col2*.5
    return col3

# Graph probs vs effectiveness
def state_probs_graph(df, time_col_name, states_list):
    lines = ['-','--','-.', ':']
    for i in range(len(states_list)):
        plt.plot(df[time_col_name],df[states_list[i]], linestyle = lines[i%4], label = states_list[i])
    plt.title("Markov Probability Analysis: "+df.name)
    plt.xlabel("Stage")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
# Survival Curve
def survival_graph(df_list, time_col_name, death_state_name):
    lines = ['-','--','-.', ':']
    i = 0
    for df in df_list:
        plt.plot(df[time_col_name], 1-df[death_state_name], linestyle = lines[i%4], label = df.name)
        i+=1
    plt.title("Survival Curve")
    plt.xlabel("Stage")
    plt.ylabel("% Surviving")
    plt.legend()
    plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The statespace
states = ["Well","Disease","Dead"]
 
# Probabilities matrix (transition matrix)
# Ensure that the order matches the current to future state transition matrix appropriately
#  Current == rows, future == columns
transMat1 = np.array([[0.20,0.80,0.00],
                      [0.17,0.80,0.03],
                      [0.00,0.00,1.00]])
transMat2 = np.array([[0.23,0.77,0.00],
                      [0.17,0.80,0.03],
                      [0.00,0.00,1.00]])
 
# Cost per state (will be multiplied by proportions so can enter annual cost across all states that apply)
costMat1 = np.array([100,5000,0])
costMat2 = np.array([100,8000,0])

# Utility per state
utilMat1 = np.array([0,.3,.1])
utilMat2 = np.array([0,.74,1])
 
# Proportion in starting in each state
initial_proportions = np.array([.5, .5, 0])

# Number of states
num_states = len(states)

# Strategies list
strategies_list = ['Strategy 1','Strategy 2']

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function that calculates cost effectiveness analysis
import collections
trans_mat_dict=collections.OrderedDict(
                {'Strategy 1': 
                    {'Transition Matrix': transMat1, 
                    'Cost Matrix': costMat1, 
                    'Utility Matrix': utilMat1}, 
                'Strategy 2': 
                    {'Transition Matrix': transMat2, 
                    'Cost Matrix': costMat2, 
                    'Utility Matrix': utilMat2}})
# Function that creates the costs utils for all time periods for every strategy
for strat in trans_mat_dict.keys():
   costs, utils, cer, df = cohort_sim(
                        trans_mat = trans_mat_dict[strat]['Transition Matrix'], 
                        cost_mat = trans_mat_dict[strat]['Cost Matrix'],
                        util_mat = trans_mat_dict[strat]['Utility Matrix'],
                        time = 1000, 
                        start_props = initial_proportions,
                        discount_rate = 0.03,
                        strategy = strat)
   trans_mat_dict[strat]['Total Cost']=costs
   trans_mat_dict[strat]['Total Utility'] = utils
   trans_mat_dict[strat]['Cost Eff. Ratio'] = cer
   trans_mat_dict[strat]['Data'] = df

# Within cycle correction
for strat in trans_mat_dict.keys():
    trans_mat_dict[strat]['Data'].Costs = cycle_correct(trans_mat_dict[strat]['Data'].Costs, 'wcc')
#dfA.Costs = cycle_correct(dfA.Costs, 'wcc')

for i in range(1,len(trans_mat_dict.keys())):
    cost0 = trans_mat_dict[next(iter(trans_mat_dict))]['Total Cost']
    util0 = trans_mat_dict[next(iter(trans_mat_dict))]['Total Utility']
    cost_comp = trans_mat_dict[list(trans_mat_dict.keys())[i]]['Total Cost']
    util_comp = trans_mat_dict[list(trans_mat_dict.keys())[i]]['Total Utility']
    icer = calc_icer(base_cost = cost0, compare_cost = cost_comp,
                  base_effect = util0, compare_effect = util_comp)
    trans_mat_dict[list(trans_mat_dict.keys())[i]]['ICER'] = (icer)


# This needs work: need to find the appropriate frontier by taking maxes and mins
#plt.plot([utilsA,utilsB],[costsA,costsB],'--')
#plt.plot(utilsA,costsA, marker = 'o')
#plt.plot(utilsB,costsB, marker = 'D')
#plt.show()

# State probabilities graphs
for strat in trans_mat_dict.keys():
    state_probs_graph(trans_mat_dict[strat]['Data'], 'Time', states)

# Survival Curve 
survival_graph([trans_mat_dict['Strategy 1']['Data'],trans_mat_dict['Strategy 2']['Data']], 
               'Time', 'Dead')

# ICER
trans_mat_dict['Strategy 2']['ICER']



