import time
import joblib
import os
import os.path as osp
#import tensorflow as tf
#from spinup import EpochLogger
#from spinup.utils.logx import restore_tf_graph
import torch

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding

import random
import math
import numpy as np
import sys

#from HPCSimPickJobs import *
from bfTorch import *

import matplotlib.pyplot as plt
plt.rcdefaults()


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x, y):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.float32)
            action = model.act(x, y)
        return action

    return get_action


# def action_from_obs(o):
#     """return first job with lowest normalized_wait_time, effectively SJF"""
#     #observation = (job,normalized_wait_time, normalized_run_time, normalized_request_nodes, , normalized_user_id, normalized_group_id, normalized_executable_id, can_schedule_now)
#     #observation doesnt include job number, starts at normalized_wait_timenormalized_request_memory
#     lst = []
#     for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
#         """The feature vector we recieve (documented above) is a list of features for each job = JOB_FEATURES, so we step through the list to check each job"""
#         lst.append((o[i + 1], math.floor(i / JOB_FEATURES)))
#         #append normalized wait time, job number
#     min_time = min([i[0] for i in lst])
#     #get lowest job wait time
#     result = [i[1] for i in lst if i[0] == min_time]
#     #get list of jobs with the lowest wait time
#     return result[0]
#     #return first job with lowest wait time


# @profile
def run_policy(env, get_action, nums, iters, score_type):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."
    rl_r = []
    f1_r = []
    f2_r = []
    sjf_r = []
    # small_r = []
    wfp_r = []
    uni_r = []

    fcfs_r = []

    # time_total = 0
    # num_total = 0
    for iter_num in range(0, iters):
        start = iter_num * args.len #for gen_preworkloads +100000
        env.reset_for_test(nums, start)
        f1_r.append(sum(env.schedule_curr_sequence_reset(env.f1_score).values()))
        # f2_r.append(sum(env.schedule_curr_sequence_reset(env.f2_score).values()))
        uni_r.append(sum(env.schedule_curr_sequence_reset(env.uni_score).values()))
        wfp_r.append(sum(env.schedule_curr_sequence_reset(env.wfp_score).values()))

        sjf_r.append(sum(env.schedule_curr_sequence_reset(env.sjf_score).values()))
        # small_r.append(sum(env.schedule_curr_sequence_reset(env.smallest_score).values()))
        fcfs_r.append(sum(env.schedule_sequence_con(env.fcfs_score).values()))

        o = env.build_observation()
        rl = 0
        total_decisions = 0
        rl_decisions = 0
        while True:
            count = 0
            skip_ = []
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                    lst.append(0)
                elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                    lst.append(0)
                else:
                    count += 1
                    if all(o[i:i + JOB_FEATURES] == [1] * (JOB_FEATURES - 1) + [0]):
                        skip_.append(math.floor(i / JOB_FEATURES))
                    lst.append(1)

            pi = get_action(o, np.array(lst))
            # time_total += time.time() - start_time
            # num_total += 1
            # print(start_time, time_total, num_total)
            a = pi[0]
            total_decisions += 1.0
            rl_decisions += 1.0
            # else:
            #     # print('SJF')
            #     a = action_from_obs(o)
            # # print(out)
            # # v_t = get_value(o)
            if a in skip_:
                print("SKIP" + "(" + str(count) + ")", end="|")
            else:
                print (str(a)+"("+str(count)+")", end="|")
            o, r, d, _ = env.step_for_test(a)
            rl += r
            if d:
                # print("RL decision ratio:",rl_decisions/total_decisions)
                print("Sequence Length:",rl_decisions)
                break
        rl_r.append(rl)
        print ("")

    # plot

    all_data = []
    all_data.append(fcfs_r)
    all_data.append(wfp_r)
    all_data.append(uni_r)
    all_data.append(sjf_r)
    all_data.append(f1_r)
    all_data.append(rl_r)
    # all_data.append(fcfs_r)

    all_medians = []
    for p in all_data:
        all_medians.append(np.median(p))
    all_means = []
    for p in all_data:
        all_means.append(np.mean(p))
    print(*all_means, sep=', ')



if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--rlmodel', type=str, default="./data/logs/ppo_50/ppo_50_s0")
    #parser.add_argument('--rlmodel', type=str, default="./data/logs/rlbf/rlbf_s0")
    #parser.add_argument('--rlmodel', type=str, default="./data/logs/rlbf-epoch100/rlbf-epoch100_s0")
    #parser.add_argument('--rlmodel', type=str, default="./data/logs/full-train/full-train_s0")
    #parser.add_argument('--rlmodel', type=str, default="./trained_models/bsld/sdsc_sp2/sdsc_sp2_s4")
    parser.add_argument('--workload', type=str, default='./data/SDSC-SP2-1998-4.2-cln.swf')
    parser.add_argument('--len', '-l', type=int, default=1024)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--iter', '-i', type=int, default=10)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=10000)
    parser.add_argument('--heuristic', type=str, default='fcfs')
    parser.add_argument('--enable_preworkloads', type=bool, default=False)

    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    model_file = os.path.join(current_dir, args.rlmodel)

    get_action = load_pytorch_policy(model_file, "") 

    # initialize the environment from scratch
    env = HPCEnv(shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, job_score_type=args.score_type,
                 batch_job_slice=args.batch_job_slice, build_sjf=False, heuristic=args.heuristic, enable_preworkloads=args.enable_preworkloads)
    env.my_init(workload_file=workload_file)
    env.seed(args.seed)

    start = time.time()
    run_policy(env, get_action, args.len, args.iter, args.score_type)
    print("elapse: {}".format(time.time() - start))