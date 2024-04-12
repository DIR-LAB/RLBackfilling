import time
import os
import os.path as osp
import torch

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding

import random
import math
import numpy as np
import sys

import csv
from rlbackfill import *

import matplotlib.pyplot as plt
plt.rcdefaults()


def load_pytorch_policy(fpath):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    model = torch.load(fname)
    return model


def get_action(model, x, y):
    """ Function for producing an action given a single state."""

    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        action = model.act(x, y)
    return action
def roundListAndString(thelist):
    #rounds elements, returns a comma seperated string
    round_list = [str(round(x, 2)) for x in thelist]
    resultString = ', '.join(round_list)
    return resultString


# @profile
def run_policy(env, model, test_len, iters, textout):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."
    # keys: bsld, wait, utils, turnaround, sld,
    # no: avg-delay, max-delay
    rl_r = {}
    rl_r['bsld'] = []
 

    heur_fcfs = {}
    heur_fcfs['bsld'] = []


    heur_wfp = {}
    heur_wfp['bsld'] = []

    heur_sjf = {}
    heur_sjf['bsld'] = []


    heur_f1 = {}
    heur_f1['bsld'] = []


    for iter_num in range(0, iters):
        env.reset_for_test(test_len)

        bsld1, wait1, utils1, turnaround1, sld1 = env.test_schedule_curr_sequence(
            env.fcfs_score)

        bsld2, wait2, utils2, turnaround2, sld2 = env.test_schedule_curr_sequence(
            env.wfp_score)

        bsld4, wait4, utils4, turnaround4, sld4 = env.test_schedule_curr_sequence(
            env.sjf_score)

        bsld5, wait5, utils5, turnaround5, sld5 = env.test_schedule_curr_sequence(
            env.f1_score)
        

        # ready, o = env.reset_to_next_backfill_or_end()

        # while not ready:
        #     continue

        heur_fcfs['bsld'].append(bsld1)

        heur_wfp['bsld'].append(bsld2)

        heur_sjf['bsld'].append(bsld4)


        heur_f1['bsld'].append(bsld5)


        ready, o = env.reset_to_next_backfill_or_end()
        if ready:
            #if we reach a point for backfilling
            total_decisions = 0
            rl_decisions = 0
            while True:
                lst = []
                for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                    if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                        lst.append(0)
                    elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                        lst.append(0)
                    else:
                        lst.append(1)

                pi = get_action(model, o, np.array(lst))
                a = pi[0]
                total_decisions += 1.0
                rl_decisions += 1.0

                o, r, d, r2, sjf_t, f1_t = env.step_for_test(a)

                if d:
                    break
        
        bsld4, wait4, utils4, turnaround4, sld4 = env.test_post_process_score()
        rl_r['bsld'].append(bsld4)

    bsld_all = []
    
    bsld_all.append(heur_fcfs['bsld'])
    bsld_all.append(heur_wfp['bsld'])
    bsld_all.append(heur_sjf['bsld'])
    bsld_all.append(heur_f1['bsld'])
    bsld_all.append(rl_r['bsld'])
    bsld_means = []
    for p in bsld_all:
        bsld_means.append(np.mean(p))


    if textout:
            #provides text output
            print(f"      FCFS WFP3 SJF F1 RL \n-------")
            print(f"bsld: {roundListAndString(bsld_means)}")
    plot_compare(bsld_all)


def plot_compare(bsld_all):
    import matplotlib.pyplot as plt

    plt.rc("font", size=16)
    fig, ax = plt.subplots(figsize=(12, 8)) # Create a single subplot

    # Plot on the single subplot
    xticks = [y + 1 for y in range(len(bsld_all))]
    ax.plot(xticks[0:1], bsld_all[0:1], 'o', color='darkorange')
    ax.plot(xticks[1:2], bsld_all[1:2], 'o', color='darkorange')
    ax.plot(xticks[2:3], bsld_all[2:3], 'o', color='darkorange')
    ax.plot(xticks[3:4], bsld_all[3:4], 'o', color='darkorange')
    ax.plot(xticks[4:5], bsld_all[4:5], 'o', color='darkorange')
    ax.boxplot(bsld_all, showfliers=False, meanline=True, showmeans=True, medianprops={
        "linewidth": 0}, meanprops={"color": "darkorange", "linewidth": 4, "linestyle": "solid"})
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(bsld_all))])
    xticklabels = ['FCFS', 'WFP3', 'SJF', 'F1', 'RL']
    ax.set(xticks=[y + 1 for y in range(len(bsld_all))],
           xticklabels=xticklabels)
    ax.set_ylabel("BSLD")
    ax.set_xlabel("Schedulers")
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--rlmodel', type=str,
                        default="./logs/ppo/ppo_s0/")
    parser.add_argument('--workload', type=str,
                        default='./data/SDSC-SP2-1998-4.2-cln.swf')
    parser.add_argument('--len', '-l', type=int, default=1024)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--iter', '-i', type=int, default=10)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--heuristic', type=str, default='fcfs')
    parser.add_argument('--enable_preworkloads', type=bool, default=False)
    parser.add_argument('--guarded', type=int, default=0)
    parser.add_argument('--relax', type=float, default=0)
    parser.add_argument('--act_size', type=int, default=1)
    parser.add_argument('--dryrun', type=bool, default=False)
    parser.add_argument('--textout', type=bool, default=True)

    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    model_file = os.path.join(current_dir, args.rlmodel)

    model = load_pytorch_policy(model_file)

    np.random.seed(args.seed)
    # initialize the environment from scratch
    env = HPCEnv(shuffle=args.shuffle, backfil=False, job_score_type=0, build_sjf=False, heuristic=args.heuristic,
                 enable_preworkloads=args.enable_preworkloads, guarded=args.guarded, relax=args.relax, act_size=args.act_size, dryrun=args.dryrun)

    
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    env.my_init(workload_file=workload_file)

    start = time.time()
    run_policy(env, model, args.len, args.iter, args.textout)
    print("elapse: {}".format(time.time() - start))
