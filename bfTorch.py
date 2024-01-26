import json
import joblib
import shutil
import numpy as np
import argparse
import math
import random

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.nn.functional import relu, softmax
import torch.nn.functional as F

import scipy.signal

import os.path as osp, time, atexit, os
import warnings
from copy import deepcopy
import os, subprocess, sys
import string
from subprocess import CalledProcessError
from textwrap import dedent
import time

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding

from mpi4py import MPI

# Workload and Job 
import re

#debug! remove me later!!!!!!
#import sys

class ProfileNode:
    """
    Used for Conservative Backfilling
    """
    def __init__(self, start_time, end_time, processors, id, running):
        self.start_time = start_time
        self.end_time = end_time
        self.processors = processors
        self.next = None
        self.id = id
        self.running = False

class Job:
    """
    1. Job Number -- a counter field, starting from 1.
    2. Submit Time -- in seconds. The earliest time the log refers to is zero, and is usually the submittal time of the first job. The lines in the log are sorted by ascending submittal times. It makes sense for jobs to also be numbered in this order.
    3. Wait Time -- in seconds. The difference between the job's submit time and the time at which it actually began to run. Naturally, this is only relevant to real logs, not to models.
    4. Run Time -- in seconds. The wall clock time the job was running (end time minus start time).
    We decided to use ``wait time'' and ``run time'' instead of the equivalent ``start time'' and ``end time'' because they are directly attributable to the Scheduler and application, and are more suitable for models where only the run time is relevant.
    Note that when values are rounded to an integral number of seconds (as often happens in logs) a run time of 0 is possible and means the job ran for less than 0.5 seconds. On the other hand it is permissable to use floating point values for time fields.
    5. Number of Allocated Processors -- an integer. In most cases this is also the number of processors the job uses; if the job does not use all of them, we typically don't know about it.
    6. Average CPU Time Used -- both user and system, in seconds. This is the average over all processors of the CPU time used, and may therefore be smaller than the wall clock runtime. If a log contains the total CPU time used by all the processors, it is divided by the number of allocated processors to derive the average.
    7. Used Memory -- in kilobytes. This is again the average per processor.
    8. Requested Number of Processors.
    9. Requested Time. This can be either runtime (measured in wallclock seconds), or average CPU time per processor (also in seconds) -- the exact meaning is determined by a header comment. In many logs this field is used for the user runtime estimate (or upper bound) used in backfilling. If a log contains a request for total CPU time, it is divided by the number of requested processors.
    10. Requested Memory (again kilobytes per processor).
    11. Status 1 if the job was completed, 0 if it failed, and 5 if cancelled. If information about chekcpointing or swapping is included, other values are also possible. See usage note below. This field is meaningless for models, so would be -1.
    12. User ID -- a natural number, between one and the number of different users.
    13. Group ID -- a natural number, between one and the number of different groups. Some systems control resource usage by groups rather than by individual users.
    14. Executable (Application) Number -- a natural number, between one and the number of different applications appearing in the workload. in some logs, this might represent a script file used to run jobs rather than the executable directly; this should be noted in a header comment.
    15. Queue Number -- a natural number, between one and the number of different queues in the system. The nature of the system's queues should be explained in a header comment. This field is where batch and interactive jobs should be differentiated: we suggest the convention of denoting interactive jobs by 0.
    16. Partition Number -- a natural number, between one and the number of different partitions in the systems. The nature of the system's partitions should be explained in a header comment. For example, it is possible to use partition numbers to identify which machine in a cluster was used.
    17. Preceding Job Number -- this is the number of a previous job in the workload, such that the current job can only start after the termination of this preceding job. Together with the next field, this allows the workload to include feedback as described below.
    18. Think Time from Preceding Job -- this is the number of seconds that should elapse between the termination of the preceding job and the submittal of this one.
    """
    def __init__(self, line = "0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0"):
        line = line.strip()
        s_array = re.split("\\s+", line)
        self.job_id = int(s_array[0])
        self.submit_time = int(s_array[1])
        self.wait_time = int(s_array[2])
        self.run_time = int(s_array[3])
        self.number_of_allocated_processors = int(s_array[4])
        self.average_cpu_time_used = float(s_array[5])
        self.used_memory = int(s_array[6])

        # "requested number of processors" and "number of allocated processors" are typically mixed.
        # I do not know their difference clearly. But it seems to me using a larger one will be sufficient.
        self.request_number_of_processors = int(s_array[7])
        self.number_of_allocated_processors = max(self.number_of_allocated_processors, self.request_number_of_processors)
        self.request_number_of_processors = self.number_of_allocated_processors
        
        self.request_number_of_nodes = -1
        
        # if we use the job's request time field
        # for model, request_time might be empty. In this case, we set request_time to the run_time
        self.request_time = int(s_array[8])
        if self.request_time == -1:
            self.request_time = self.run_time

        # if we use the run time as the most accurate request time
        # self.request_time = self.run_time + 60
        # if we gradually increase the accuracy of job's request time
        # with a percentage wrong estimation and round to a fixed time: 1,2,3,... hours.
        # this.requestTime = (int) (this.runTime + this.runTime * 0.4);
        # int roundsTo = 60 * 60; //round up to hours
        # this.requestTime = (this.requestTime / roundsTo + 1) * roundsTo;

        self.request_memory = int(s_array[9])
        self.status = int(s_array[10])
        self.user_id = int(s_array[11])
        self.group_id = int(s_array[12])
        self.executable_number = int(s_array[13])
        self.queue_number = int(s_array[14])

        try:
            self.partition_number = int(s_array[15])
        except ValueError:
            self.partition_number = 0

        self.proceeding_job_number = int(s_array[16])
        self.think_time_from_proceeding_job = int(s_array[17])

        self.random_id = self.submit_time

        self.scheduled_time = -1

        self.allocated_machines = None

        self.slurm_in_queue_time = 0
        self.slurm_age = 0
        self.slurm_job_size = 0.0
        self.slurm_fair = 0.0
        self.slurm_partition = 0
        self.slurm_qos = 0
        self.slurm_tres_cpu = 0.0
        
    def __eq__(self, other):
        return self.job_id == other.job_id

    def __lt__(self, other):
        return self.job_id < other.job_id

    def __hash__(self):
        return hash(self.job_id)

    def __str__(self):
        return "J["+str(self.job_id)+"]-["+str(self.request_number_of_processors)+"]-["+str(self.submit_time)+"]-["+str(self.request_time)+"]"
    def __feature__(self):
        return [self.submit_time, self.request_number_of_processors, self.request_time,
                self.user_id, self.group_id, self.executable_number, self.queue_number]


class Workloads:

    def __init__(self, path):
        self.all_jobs = []
        self.max = 0
        self.max_exec_time = 0
        self.min_exec_time = sys.maxsize
        self.max_job_id = 0

        self.max_requested_memory = 0
        self.max_user_id = 0
        self.max_group_id = 0
        self.max_executable_number = 0
        self.max_job_id = 0
        self.max_nodes = 0
        self.max_procs = 0

        with open(path) as fp:
            for line in fp:
                if line.startswith(";"):
                    if line.startswith("; MaxNodes:"):
                        self.max_nodes = int(line.split(":")[1].strip())
                    if line.startswith("; MaxProcs:"):
                        self.max_procs = int(line.split(":")[1].strip())
                    continue

                j = Job(line)
                if j.run_time > self.max_exec_time:
                    self.max_exec_time = j.run_time
                if j.run_time < self.min_exec_time:
                    self.min_exec_time = j.run_time
                if j.request_memory > self.max_requested_memory:
                    self.max_requested_memory = j.request_memory
                if j.user_id > self.max_user_id:
                    self.max_user_id = j.user_id
                if j.group_id > self.max_group_id:
                    self.max_group_id = j.group_id
                if j.executable_number > self.max_executable_number:
                    self.max_executable_number = j.executable_number

                # filter those illegal data whose runtime < 0
                if j.run_time < 0:
                    j.run_time = 10
                if j.run_time > 0:
                    self.all_jobs.append(j)
                
                    if j.request_number_of_processors > self.max:
                        self.max = j.request_number_of_processors

        # if max_procs = 0, it means node/proc are the same.
        if self.max_procs == 0:
            self.max_procs = self.max_nodes

        print ("Max Allocated Processors:", str(self.max), "; max node:", self.max_nodes,
               "; max procs:", self.max_procs,
               "; max execution time:", self.max_exec_time)

        self.all_jobs.sort(key=lambda job: job.job_id)

    def size(self):
        return len(self.all_jobs)

    def reset(self):
        for job in self.all_jobs:
            job.scheduled_time = -1

    def __getitem__(self, item):
        return self.all_jobs[item]


def test_job_workload():
    print ("Loading the workloads...")
    load = Workloads("../../../data/lublin_256.swf")
    print ("Finish loading the workloads...", type(load[0]))
    print (load.max_nodes, load.max_procs)
    print (load[0].__feature__())
    print (load[1].__feature__())
    
    # empty_job_str = "0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0"
    # empty_job = Job(empty_job_str)
    # print (empty_job.job_id, empty_job.feature())

class Machine:
    def __init__(self, id):
        self.id = id
        self.running_job_id = -1
        self.is_free = True
        self.job_history = []

    def taken_by_job(self, job_id):
        if self.is_free:
            self.running_job_id = job_id
            self.is_free = False
            self.job_history.append(job_id)
            return True
        else:
            return False

    def release(self):
        if self.is_free:
            return -1
        else:
            self.is_free = True
            self.running_job_id = -1
            return 1

    def reset(self):
        self.is_free = True
        self.running_job_id = -1
        self.job_history = []

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return "M["+str(self.id)+"] "


class Cluster:
    def __init__(self, cluster_name, node_num, num_procs_per_node):
        self.name = cluster_name
        self.total_node = node_num
        self.free_node = node_num
        self.used_node = 0
        self.num_procs_per_node = num_procs_per_node
        self.all_nodes = []

        for i in range(self.total_node):
            self.all_nodes.append(Machine(i))

    def feature(self):
        return [self.free_node]

    def can_allocated(self, job):
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes > self.free_node:
            return False
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes <= self.free_node:
            #print(f"debug! can allocate {job} due to enough nodes being available!")
            return True
        request_node = int(math.ceil(float(job.request_number_of_processors)/float(self.num_procs_per_node)))
        job.request_number_of_nodes = request_node
        if request_node > self.free_node:
            return False
        else:
            #print(f"debug! can allocate {job} due to enough proc nodes being available!")
            return True
        

    def allocate(self, job_id, request_num_procs):
        allocated_nodes = []
        request_node = int(math.ceil(float(request_num_procs) / float(self.num_procs_per_node)))

        if request_node > self.free_node:
            return []

        allocated = 0

        for m in self.all_nodes:
            if allocated == request_node:
                return allocated_nodes
            if m.taken_by_job(job_id):
                allocated += 1
                self.used_node += 1
                self.free_node -= 1
                allocated_nodes.append(m)

        if allocated == request_node:
            return allocated_nodes

        print ("Error in allocation, there are enough free resources but can not allocated!")
        return []

    def release(self, releases):
        self.used_node -= len(releases)
        self.free_node += len(releases)

        for m in releases:
            m.release()

    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def reset(self):
        self.used_node = 0
        self.free_node = self.total_node
        for m in self.all_nodes:
            m.reset()

# HPC Env
MAX_QUEUE_SIZE = 128
MLP_SIZE = 256

MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours

# each job has three features: wait_time, requested_node, runtime, machine states,
JOB_FEATURES = 8
DEBUG = False

JOB_SEQUENCE_SIZE = 256
SKIP_TIME = 360 # skip 60 seconds

class HPCEnv(gym.Env):
    def __init__(self,shuffle=False, backfil=False, skip=False, job_score_type=0, batch_job_slice=0, build_sjf=False, heuristic="fcfs", enable_preworkloads=False):  # do nothing and return. A workaround for passing parameters to the environment
        super(HPCEnv, self).__init__()
        print("Initialize Simple HPC Env")

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(JOB_FEATURES * MAX_QUEUE_SIZE,),
                                            dtype=np.float32)
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.start_idx_last_reset = 0

        self.loads = None
        self.cluster = None

        self.bsld_algo_dict = {}
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []
        self.enable_preworkloads = enable_preworkloads
        if self.enable_preworkloads:
            self.start = 100000
        self.pre_workloads = []

        self.shuffle = shuffle
        self.backfil = backfil
        self.skip = skip
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization
        self.job_score_type = job_score_type
        self.batch_job_slice = batch_job_slice

        self.build_sjf = build_sjf
        self.sjf_scores = []

        #section for backfilling variables
        self.backfilling = False #are we currently backfilling?
        self.rjob = 0 #rjob = relative job = job we are backfilling relative to
        

        if heuristic == "f1":
            self.heuristic = self.f1_score
        elif heuristic == "f2":
            self.heuristic = self.f2_score
        elif heuristic == "f3":
            self.heuristic = self.f3_score
        elif heuristic == "f4":
            self.heuristic = self.f4_score
        elif heuristic == "sjf":
            self.heuristic = self.sjf_score
        elif heuristic == "smallest":
            self.heuristic = self.smallest_score
        elif heuristic == "wfp":
            self.heuristic = self.wfp_score
        elif heuristic == "uni":
            self.heuristic = self.uni_score
        elif heuristic == "fcfs":
            self.heuristic = self.fcfs_score


    #@profile
    def my_init(self, workload_file = '', sched_file = ''):
        print ("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Cluster", self.loads.max_nodes, self.loads.max_procs/self.loads.max_nodes)
        self.penalty_job_score = JOB_SEQUENCE_SIZE * self.loads.max_exec_time / 10

        if self.enable_preworkloads:
            job_sequence_size = JOB_SEQUENCE_SIZE
            print("pre workloads enabled!")
            self.gen_preworkloads(job_sequence_size + np.random.randint(job_sequence_size)+10000) #+10000
        

        if self.build_sjf: #this is for trajectory filtering.
            #calculate SJF scores for all sample sequence and save them here
            index = 0
            if self.batch_job_slice == 0:
                max_index = self.loads.size() - JOB_SEQUENCE_SIZE - 1
            else:
                max_index = min(self.batch_job_slice, self.loads.size()) - JOB_SEQUENCE_SIZE - 1
            print("max index... initializing SJF Score Array", max_index)

            while index <= max_index:
                index += 1
                if index % 100 == 0:
                    print("index", index)

                self.cluster.reset()
                self.loads.reset()

                self.job_queue = []
                self.running_jobs = []
                self.visible_jobs = []
                self.pairs = []

                self.current_timestamp = 0
                self.start = 0
                self.next_arriving_job_idx = 0
                self.last_job_in_batch = 0
                self.num_job_in_batch = 0
                self.scheduled_rl = {}
                self.penalty = 0
                self.pivot_job = False
                self.scheduled_scores = []

                job_sequence_size = JOB_SEQUENCE_SIZE
                self.pre_workloads = []

                self.start = index;
                self.start_idx_last_reset = self.start
                self.num_job_in_batch = job_sequence_size
                self.last_job_in_batch = self.start + self.num_job_in_batch
                self.current_timestamp = self.loads[self.start].submit_time
                self.job_queue.append(self.loads[self.start])
                self.next_arriving_job_idx = self.start + 1

        

                self.sjf_scores.append(sum(self.schedule_curr_sequence_reset(self.sjf_score).values()))

            #print(self.sjf_scores)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def f1_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        return (np.log10(request_time if request_time>0 else 0.1) * request_processors + 870 * np.log10(submit_time if submit_time>0 else 0.1))

    def f2_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f2: r^(1/2)*n + 25600 * log10(s)
        return (np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time))

    def f3_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f3: r * n + 6860000 * log10(s)
        return (request_time * request_processors + 6860000 * np.log10(submit_time))

    def f4_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f4: r * sqrt(n) + 530000 * log10(s)
        return (request_time * np.sqrt(request_processors) + 530000 * np.log10(submit_time))

    def sjf_score(self, job):
        # run_time = job.run_time
        request_time = job.request_time
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier 
        return (request_time, submit_time)
    
    def smallest_score(self, job):
        request_processors = job.request_number_of_processors
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier 
        return (request_processors, submit_time)

    def wfp_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time-job.submit_time
        return -np.power(float(waiting_time)/request_time, 3)*request_processors

    def uni_score(self,job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time-job.submit_time

        return -(waiting_time+1e-15)/(np.log2(request_processors+1e-15)*request_time)

    def fcfs_score(self, job):
        submit_time = job.submit_time
        return submit_time

    def gen_preworkloads(self, size):
        # Generate some running jobs to randomly fill the cluster.
        # size = self.np_random.randint(2 * job_sequence_size)
        #print("ent")
        running_job_size = size
        for i in range(running_job_size):
            _job = self.loads[self.start - i - 1]
            req_num_of_processors = _job.request_number_of_processors
            runtime_of_job = _job.request_time
            job_tmp = Job()
            job_tmp.job_id = (-1 - i)  # to be different from the normal jobs; normal jobs have a job_id >= 0
            job_tmp.request_number_of_processors = req_num_of_processors
            job_tmp.run_time = runtime_of_job
            if self.cluster.can_allocated(job_tmp):
                self.running_jobs.append(job_tmp)
                job_tmp.scheduled_time = max(0, (self.current_timestamp - random.randint(0, max(runtime_of_job, 1))))
                # job_tmp.scheduled_time = max(0, (self.current_timestamp - runtime_of_job/2))
                job_tmp.allocated_machines = self.cluster.allocate(job_tmp.job_id, job_tmp.request_number_of_processors)
                self.pre_workloads.append(job_tmp)
            else:
                break

    def refill_preworkloads(self):
        for _job in self.pre_workloads:
            self.running_jobs.append(_job)
            _job.allocated_machines = self.cluster.allocate(_job.job_id, _job.request_number_of_processors)    

    #@profile
    def reset(self):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        if self.enable_preworkloads:
            self.start = 100000
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []

        job_sequence_size = JOB_SEQUENCE_SIZE

        #section for backfilling variables
        self.backfilling = False #are we currently backfilling?
        self.rjob = 0 #rjob = relative job = job we are backfilling relative to 



        self.pre_workloads = []
        
        assert self.batch_job_slice == 0 or self.batch_job_slice>=job_sequence_size

        if self.build_sjf:
            done = False
            while not done:
                # randomly sample a sequence of jobs from workload (self.start_idx_last_reset + 1) % (self.loads.size() - 2 * job_sequence_size
                if self.batch_job_slice == 0:
                    self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
                else:
                    self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))

                if self.sjf_scores[self.start] > 10 and self.sjf_scores[self.start] < 150:
                    done = True
        else:
            if self.batch_job_slice == 0:
                self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
            else:
                self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        self.profile_head = None
        #for conservative backfilling

        if self.enable_preworkloads:
            job_sequence_size = JOB_SEQUENCE_SIZE
            print("pre workloads enabled!")
            self.gen_preworkloads(job_sequence_size + np.random.randint(job_sequence_size)+10000) #+10000

        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.sjf_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f1_score).values()))
        # self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.smallest_score).values()))
        # self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.fcfs_score).values()))
        #self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f2_score).values()))
        #self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f3_score).values()))
        #self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f4_score).values()))        

        return self.build_observation(), self.build_critic_observation()
        
        #print(np.mean(self.scheduled_scores))
        '''
        if (np.mean(self.scheduled_scores) > 5):
            return self.build_observation()
        else:
            return self.reset()
        '''

    def reset_for_test(self, num,start):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []

        #section for backfilling variables
        self.backfilling = False #are we currently backfilling?
        self.rjob = 0 #rjob = relative job = job we are backfilling relative to 

        job_sequence_size = num
        assert self.batch_job_slice == 0 or self.batch_job_slice>=job_sequence_size
        if self.batch_job_slice == 0:
            self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
        else:
            self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))
        #self.start = start
        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        self.profile_head = None
        if self.enable_preworkloads:
            job_sequence_size = JOB_SEQUENCE_SIZE
            print("pre workloads enabled!")
            self.gen_preworkloads(job_sequence_size + np.random.randint(job_sequence_size) + 10000)
        #for conservative backfilling
    
    def skip_for_resources_greedy(self, job, scheduled_logs):
        #note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    #@profile
    def moveforward_for_resources_backfill_greedy(self, job, scheduled_logs):
        #note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= job.request_number_of_processors:
                break

        while not self.cluster.can_allocated(job):
            # try to backfill as many jobs as possible. Use FCFS
            self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)
            job_queue_iter_copy.remove(job) #so we dont try and backfill the relative job


            for _j in job_queue_iter_copy:
                if (self.current_timestamp + _j.request_time) < earliest_start_time:
                    if self.cluster.can_allocated(_j):
                        # we should be OK to schedule the job now
                        assert _j.scheduled_time == -1  # this job should never be scheduled before.
                        _j.scheduled_time = self.current_timestamp
                        _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                        self.running_jobs.append(_j)
                        score = self.job_score(_j)   # calculated reward
                        scheduled_logs[_j.job_id] = score
                        self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
                
            if self.next_arriving_job_idx < self.last_job_in_batch \
            and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job

    def post_process_score(self, scheduled_logs):
        if self.job_score_type == 0:
            # bsld
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        elif self.job_score_type == 1:
            # wait time
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        elif self.job_score_type == 2:
            # turnaround time
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        elif self.job_score_type == 3:
            total_cpu_hour = (self.current_timestamp - self.loads[self.start].submit_time)*self.loads.max_procs
            for i in scheduled_logs:
                scheduled_logs[i] /= total_cpu_hour
        elif self.job_score_type == 4:
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        else:
            raise NotImplementedError
    #@profile
    def schedule_curr_sequence_reset(self, score_fn):
        # schedule the sequence of jobs using heuristic algorithm. 
        scheduled_logs = {}
        # f = False
        # if score_fn.__name__ == "sjf_score":
        #     f = True
        #     num_total = 0
        # start_time = time.time()
        while True:
            self.job_queue.sort(key=lambda j: score_fn(j))
            job_for_scheduling = self.job_queue[0]
            #print(f"debug! total job count ={len(self.job_queue)}")
            # if f:
            #     num_total += 1
            # if selected job needs more resources, skip scheduling and try again after adding new jobs or releasing some resources
            if not self.cluster.can_allocated(job_for_scheduling):
                if self.backfil:
                    self.moveforward_for_resources_backfill_greedy(job_for_scheduling, scheduled_logs)
                else:
                    self.skip_for_resources_greedy(job_for_scheduling, scheduled_logs)
            
            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                        job_for_scheduling.request_number_of_processors)
            self.running_jobs.append(job_for_scheduling)
            score = self.job_score(job_for_scheduling)  # calculated reward
            scheduled_logs[job_for_scheduling.job_id] = score
            self.job_queue.remove(job_for_scheduling)

            not_empty = self.moveforward_for_job()
            if not not_empty:
                break
        self.post_process_score(scheduled_logs)
        # if f:
        #     print((time.time()-start_time)/num_total, num_total)
        # reset again
        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        if self.enable_preworkloads:
            self.refill_preworkloads()

        return scheduled_logs
    
    def schedule_curr_sequence(self, score_fn):
        # schedule the sequence of jobs using heuristic algorithm. 
        scheduled_logs = {}
        # f = False
        # if score_fn.__name__ == "sjf_score":
        #     f = True
        #     num_total = 0
        # start_time = time.time()
        while True:
            self.job_queue.sort(key=lambda j: score_fn(j))
            job_for_scheduling = self.job_queue[0]
            # if f:
            #     num_total += 1
            # if selected job needs more resources, skip scheduling and try again after adding new jobs or releasing some resources
            if not self.cluster.can_allocated(job_for_scheduling):
                #print(f'debug! cannot allocate {job_for_scheduling}, backfilling it now!')
                if self.backfil:
                    #print("debug! backfill opportunity found!")
                    self.rjob = job_for_scheduling
                    self.backfilling = True
                    return False
                else:
                    #print("debug! skipping backfill opportunity!")
                    self.skip_for_resources_greedy(job_for_scheduling, scheduled_logs)
            
            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                        job_for_scheduling.request_number_of_processors)
            self.running_jobs.append(job_for_scheduling)
            score = self.job_score(job_for_scheduling)  # calculated reward
            self.scheduled_rl[job_for_scheduling.job_id] = score
            self.job_queue.remove(job_for_scheduling)

            not_empty = self.moveforward_for_job()
            if not not_empty:
                return True

    def build_critic_observation(self):
        vector = np.zeros(JOB_SEQUENCE_SIZE * 3,dtype=float)
        earlist_job = self.loads[self.start_idx_last_reset]
        earlist_submit_time = earlist_job.submit_time
        pairs = []
        for i in range(self.start_idx_last_reset, self.last_job_in_batch+1):
            job = self.loads[i]
            submit_time = job.submit_time - earlist_submit_time
            request_processors = job.request_number_of_processors
            request_time = job.request_time

            normalized_submit_time = min(float(submit_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
            normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
            normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs), 1.0 - 1e-5)

            pairs.append([normalized_submit_time, normalized_run_time, normalized_request_nodes])

        for i in range(JOB_SEQUENCE_SIZE):
            vector[i*3:(i+1)*3] = pairs[i]

        return vector

    def build_observation(self):
        vector = np.zeros((MAX_QUEUE_SIZE) * JOB_FEATURES, dtype=float)
        self.job_queue.sort(key=lambda job: self.fcfs_score(job))
        self.visible_jobs = []
        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.job_queue):
                self.visible_jobs.append(self.job_queue[i])
            else:
                break
        self.visible_jobs.sort(key=lambda j: self.fcfs_score(j))
        if self.shuffle:
            random.shuffle(self.visible_jobs)


        #@ddai: optimize the observable jobs
        self.visible_jobs = []
        if len(self.job_queue) <= MAX_QUEUE_SIZE:
            for i in range(0, len(self.job_queue)):
                self.visible_jobs.append(self.job_queue[i])
        else:
            visible_f1 = []
            f1_index = 0
            self.job_queue.sort(key=lambda job: self.f1_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_f1.append(self.job_queue[i])
            
            visible_f2 = []
            f2_index = 0
            self.job_queue.sort(key=lambda job: self.f2_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_f2.append(self.job_queue[i])
            
            visible_sjf = []
            sjf_index = 0
            self.job_queue.sort(key=lambda job: self.sjf_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_sjf.append(self.job_queue[i])

            visible_small = []
            small_index = 0
            self.job_queue.sort(key=lambda job: self.smallest_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_small.append(self.job_queue[i])

            visible_random = []
            random_index = 0
            shuffled = list(self.job_queue)
            random.shuffle(shuffled)
            for i in range(0, MAX_QUEUE_SIZE):
                visible_random.append(shuffled[i])

            index = 0

            while index < MAX_QUEUE_SIZE:
                f1_job = visible_f1[f1_index]
                f1_index += 1
                f2_job = visible_f2[f2_index]
                f2_index += 1
                sjf_job = visible_sjf[sjf_index]
                sjf_index += 1
                small_job = visible_small[small_index]
                small_index += 1
                random_job = visible_sjf[random_index]
                random_index += 1
                #if (not f1_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                #    self.visible_jobs.append(f1_job)
                #    index += 1
                #if (not f2_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                #    self.visible_jobs.append(f2_job)
                #    index += 1
                if (not sjf_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                    self.visible_jobs.append(sjf_job)
                    index += 1
                if (not small_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                    self.visible_jobs.append(small_job)
                    index += 1
                if (not random_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                    self.visible_jobs.append(random_job)
                    index += 1


        '''
        @ddai: OPTIMIZE_OBSV. This time, we calculate the earliest start time of each job and expose that to the RL agent.
        if it is 0, then the job can start now, if it is near 1, that means it will have to wait for a really long time to start.
        The earliest start time is calculated based on current resources and the running jobs. It assumes no more jobs will be scheduled.

        # calculate the free resources at each outstanding ts
        free_processors_pair = []
        free_processors = (self.cluster.free_node * self.cluster.num_procs_per_node)
        free_processors_pair.append((free_processors, 0))

        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
        for rj in self.running_jobs:
            free_processors += rj.request_number_of_processors
            free_processors_pair.append((free_processors, (rj.scheduled_time + rj.run_time - self.current_timestamp)))
        '''

        self.pairs = []
        add_skip = False
        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.visible_jobs) and i < (MAX_QUEUE_SIZE ):
                job = self.visible_jobs[i]
                submit_time = job.submit_time
                request_processors = job.request_number_of_processors
                request_time = job.request_time
                # run_time = job.run_time
                wait_time = self.current_timestamp - submit_time

                # make sure that larger value is better.
                normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
                normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
                normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs),  1.0 - 1e-5)

                '''
                @ddai: part 2 of OPTIMIZE_OBSV
                earliest_start_time = 1
                for fp, ts in free_processors_pair:
                    if request_processors < fp:
                        earliest_start_time = ts
                        break
                normalized_earliest_start_time = min(float(earliest_start_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
                '''

                # add extra parameters, include "Requested Memory", "User Id", "Groupd Id", "Exectuable Id", if its value does not exist in the trace (-1), we set it to 1 by default.
                if job.request_memory == -1:
                    normalized_request_memory = 1
                else:
                    normalized_request_memory = min(float(job.request_memory)/float(self.loads.max_requested_memory), 1.0 - 1e-5)

                if job.user_id == -1:
                    normalized_user_id = 1
                else:
                    normalized_user_id = min(float(job.user_id)/float(self.loads.max_user_id), 1.0-1e-5)

                if job.group_id == -1:
                    normalized_group_id = 1
                else:
                    normalized_group_id = min(float(job.group_id)/float(self.loads.max_group_id), 1.0-1e-5)

                if job.executable_number == -1:
                    normalized_executable_id = 1
                else:
                    normalized_executable_id = min(float(job.executable_number)/float(self.loads.max_executable_number), 1.0-1e-5)

                if self.cluster.can_allocated(job):
                    can_schedule_now = 1.0 - 1e-5
                else:
                    can_schedule_now = 1e-5
                self.pairs.append([job,normalized_wait_time, normalized_run_time, normalized_request_nodes, normalized_request_memory, normalized_user_id, normalized_group_id, normalized_executable_id, can_schedule_now])

            elif self.skip and not add_skip:  # the next job is skip
                add_skip = True
                if self.pivot_job:
                    self.pairs.append([None, 1, 1, 1, 1, 1, 1, 1, 1])
                else:
                    self.pairs.append([None, 1, 1, 1, 1, 1, 1, 1, 0])
            else:
                self.pairs.append([None,0,1,1,1,1,1,1,0])

        for i in range(0, MAX_QUEUE_SIZE):
            vector[i*JOB_FEATURES:(i+1)*JOB_FEATURES] = self.pairs[i][1:]

        return vector

    def moveforward_for_resources_backfill_modified(self, rjob, job_for_scheduling):
        #note that this function is only called when current job can not be scheduled.
        #got rid of assert here because it seems like it works fine without and control wise it checks below in the while not so it should be fine

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= rjob.request_number_of_processors:
                break

        while not self.cluster.can_allocated(rjob):
            _j = job_for_scheduling
            if self.cluster.can_allocated(_j):
                # we should be OK to schedule the job now
                assert _j.scheduled_time == -1  # this job should never be scheduled before.
                _j.scheduled_time = self.current_timestamp
                _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                self.running_jobs.append(_j)
                score = self.job_score(_j)   # calculated reward
                self.scheduled_rl[_j.job_id] = score
                self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
                
            if self.next_arriving_job_idx < self.last_job_in_batch \
            and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job
            return False #backfilling isnt done, ask for more jobs
        #print(f'debug! Backfilling completed succesfully! Returning true')
        return True #backfilling is done, return to heuristic scheduling
    
    #@profile
    def moveforward_for_resources_backfill(self, job):
        #note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= job.request_number_of_processors:
                break

        while not self.cluster.can_allocated(job):
            # try to backfill as many jobs as possible. Use FCFS
            self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)
            for _j in job_queue_iter_copy:
                if self.cluster.can_allocated(_j) and (self.current_timestamp + _j.request_time) < earliest_start_time:
                    # we should be OK to schedule the job now
                    assert _j.scheduled_time == -1  # this job should never be scheduled before.
                    _j.scheduled_time = self.current_timestamp
                    _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                    self.running_jobs.append(_j)
                    score = self.job_score(_j)   # calculated reward
                    self.scheduled_rl[_j.job_id] = score
                    self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
                
            if self.next_arriving_job_idx < self.last_job_in_batch \
            and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job
    
    def skip_for_resources(self, job):
        #note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    #@profile
    def moveforward_for_job(self):
        if self.job_queue:
            return True

        # if we need to add job, but can not add any more, return False indicating the job_queue is for sure empty now.
        if self.next_arriving_job_idx >= self.last_job_in_batch:
            assert not self.job_queue
            return False

        # move forward to add jobs into job queue.
        while not self.job_queue:
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                return True     # job added
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def job_score(self, job_for_scheduling):

        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization 4: Average slowdown
        if self.job_score_type == 0:
            # bsld
            _tmp = max(1.0, (float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                            /
                            max(job_for_scheduling.run_time, 10)))
        elif self.job_score_type == 1:
            #wait time
            _tmp = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time)
        elif self.job_score_type == 2:
            # turnaround time
            _tmp = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
        elif self.job_score_type == 3:
            # utilization
            _tmp = -float(job_for_scheduling.run_time*job_for_scheduling.request_number_of_processors)
        elif self.job_score_type == 4:
            # sld
            _tmp = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)\
                /job_for_scheduling.run_time
        else:
            raise NotImplementedError

            # Weight larger jobs.
        #_tmp = _tmp * (job_for_scheduling.run_time * job_for_scheduling.request_number_of_processors)
        return _tmp

    def has_only_one_job(self):
        if len(self.job_queue) == 1:
            return True
        else:
            return False

    def skip_schedule(self):
        # schedule nothing, just move forward to next timestamp. It should 1) add a new job; 2) finish a running job; 3) reach skip time
        next_time_after_skip = self.current_timestamp + SKIP_TIME

        next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
        next_resource_release_machines = []
        if self.running_jobs:  # there are running jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

        if self.next_arriving_job_idx >= self.last_job_in_batch and not self.running_jobs:
            if not self.pivot_job:
                self.pivot_job = True
                return False, 0
            else:
                return False, 0

        if next_time_after_skip < min(self.loads[self.next_arriving_job_idx].submit_time, next_resource_release_time):
            self.current_timestamp = next_time_after_skip
            return False, 0
        
        if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
            self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
            self.job_queue.append(self.loads[self.next_arriving_job_idx])
            self.next_arriving_job_idx += 1
        else:
            self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
            self.cluster.release(next_resource_release_machines)
            self.running_jobs.pop(0)  # remove the first running job.
        return False, 0

    def schedule(self, job_for_scheduling):
        # make sure we move forward and release needed resources
        if not self.cluster.can_allocated(job_for_scheduling):
            if self.backfil:
                self.moveforward_for_resources_backfill(job_for_scheduling)
            else:
                self.skip_for_resources(job_for_scheduling)
        
        # we should be OK to schedule the job now
        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        job_for_scheduling.scheduled_time = self.current_timestamp
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id, job_for_scheduling.request_number_of_processors)
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)   # calculated reward
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)  # remove the job from job queue

        # after scheduling, check if job queue is empty, try to add jobs. 
        not_empty = self.moveforward_for_job()

        if not_empty:
            # job_queue is not empty
            return False
        else:
            # job_queue is empty and can not add new jobs as we reach the end of the sequence
            return True

    def valid(self, a):
        action = a[0]
        return self.pairs[action][0]

    #@profile
    def step(self, a):
        job_for_scheduling = self.pairs[a][0]
        if self.backfilling:
            done = self.moveforward_for_resources_backfill_modified(
                self.rjob, job_for_scheduling)
            # generate done signal
            if not done:
                #print(f'debug! backfilling not done for rjob = {self.rjob}, asking for new job to backfill')
                obs = self.build_observation()
                return [obs, 0, False, 0, 0, 0]
            # return to ask for another job to backfill
            if done:
                #print(f'debug! backfilling is done! return to heur scheduling')
                self.backfilling = False
                self.rjob = 0
                # exit backfilling loop
        """
        basically two done signals being generated (1 for the backfilling status, 1 for the entire trajectory)
        but they shouldnt collide bc no matter what either skip_schedule or schedule runs after the backfilling and generates the traj done signal
        """
        if not job_for_scheduling:
            done, _ = self.skip_schedule()
        else:
            done = self.schedule_curr_sequence(self.heuristic)
            #schedule using modified heuristic scheduling, maybe include argument in the program to change this for testing purposes???            

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, 0, 0, 0]
        else:
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())
            best_total = min(self.scheduled_scores)
            sjf = self.scheduled_scores[0]
            f1 = self.scheduled_scores[1]
            rwd2 = (best_total - rl_total)
            rwd = -rl_total
            '''
            if (best_total) < rl_total:
                rwd = -1
            elif best_total == rl_total:
                rwd = 0
            else:
                rwd = 1    
            '''
            return [None, rwd, True, rwd2, sjf, f1]
    
    def step_for_test(self, a):
        job_for_scheduling = self.pairs[a][0]
        if self.backfilling:
            #print("debug! backfilling job!")
            done = self.moveforward_for_resources_backfill_modified(
                self.rjob, job_for_scheduling)
            # generate done signal
            if not done:
                #print(f'debug! backfilling not done for rjob = {self.rjob}, asking for new job to backfill')
                obs = self.build_observation()
                return [obs, 0, False, 0]
            # return to ask for another job to backfill
            if done:
                #print("debug! done backfilling!")
                #print(f'debug! backfilling is done! return to heur scheduling')
                self.backfilling = False
                self.rjob = 0
                # exit backfilling loop
        """
        basically two done signals being generated (1 for the backfilling status, 1 for the entire trajectory)
        but they shouldnt collide bc no matter what either skip_schedule or schedule runs after the backfilling and generates the traj done signal
        """
        if not job_for_scheduling:
            #print("debug! skipping current job!")
            done, _ = self.skip_schedule()
        else:
            #print("debug! scheduling via heuristic!")
            done = self.schedule_curr_sequence(self.sjf_score)
            #schedule using modified heuristic scheduling, maybe include argument in the program to change this for testing purposes???            

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, 0]
        else:
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())
            return [None, rl_total, True, None]
      
    def find_anchor_point(self, job):
        """using proc profile, find earliest time when job can start."""
        #print(f"debug! finding anchor point for {job.job_id}")
        earliest_start_time = self.current_timestamp
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        #key=lambda running_job: (running_job.scheduled_time + running_job.request_time)
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        #feel like there might be something here idk what yet
        #print(f"current amount of free proc before running jobs = {free_processors}")
        if free_processors < job.request_number_of_processors:
            for running_job in self.running_jobs:
                free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
                earliest_start_time = running_job.scheduled_time + running_job.request_time
                #earliest_start_time = running_job.scheduled_time + running_job.request_time
                # min(earliest_start_time, running_job.scheduled_time + running_job.request_time - job.request_time)
                if free_processors >= job.request_number_of_processors:
                    break
        #print(f"current amount of free proc after running jobs = {free_processors}")
        #print(f"debug! earliest starting time = {earliest_start_time}, free proc = {free_processors}, job requesting {job.request_number_of_processors} procs")
        #sorts through running jobs to find when the first running job will release enough procs for it to run
        current = self.profile_head
        while current is not None:
            enough_proc = (free_processors - current.processors) >= job.request_number_of_processors if not current.running else True
            #if a job is already running then we have already done the deduction at the current timestep
            if current.start_time >= earliest_start_time and enough_proc:
                #if we are after running job release time and if we have enough available processors and the job hasn't already started in the past
                anchor = current.start_time
                end_time = anchor + job.request_time
                #print(f"debug! potential anchor found at {current.id}")
                #anchor = potential time where we could start job at the same time, end time= when the job we are trying to schedule will end
                if self.check_availability(current, anchor, end_time, job.request_number_of_processors):
                    return current.id, current.start_time
                    #job id of anchor point
            current = current.next
        #print(f"debug! could not find anchor point, returning None and None :(")
        return None, None
        #if we cant find a place to anchor the job, place it at the end of the queue

    def check_availability(self, current, start_range, end_range, requested_proc):
        """determines if resources used by job remain available throughout its duration"""
        if start_range < self.current_timestamp:
            return False
            #prevents issue where we anchor to a job that has already started in the past
        #print(f"amount of nodes request by job = {requested_proc}")
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        #key=lambda running_job: (running_job.scheduled_time + running_job.request_time)
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        #print(f"current amount of free proc before running jobs = {free_processors}")
        if start_range != self.current_timestamp:
        #if we are starting a jump at the current time, we dont need to calculate for running jobs because the free procs have already been subtracted
            for running_job in self.running_jobs:
                if (running_job.scheduled_time <= start_range < (running_job.scheduled_time + running_job.request_time)) or (start_range <= running_job.scheduled_time < end_range):
                    #if running job is in range of the job we are trying to schedule
                    #print(f"running job found in range = {running_job.job_id}")
                    running_proc = len(running_job.allocated_machines) * self.cluster.num_procs_per_node
                    free_processors -= running_proc
                    #print(f"new free proc after = {free_processors} ")
                    if free_processors < requested_proc:
                        #print("debug! not enough free processors in range due to running jobs!")
                        return False
        #we have to check the running jobs and subtract the available node count accordingly bc some jobs might be running in the time range
        #TODO: possibly check if it is None at the beginning to see if there are free processors
        while current is not None and current.start_time < end_range:
            #check proc map if we have enough processors available
            #if current.start_time != self.current_timestamp:
                #if the job 
            if ((current.start_time <= start_range < current.end_time) or (start_range <= current.start_time < end_range)) and not current.running:
                #print(f"scheduled job found in range = {current.id}")
                free_processors -= current.processors
                #print(f"new proc count after subtraction = {free_processors}")
                if free_processors < requested_proc:
                    #print(f"debug! not enough free processors in range due to scheduled job id={current.id}!")
                    return False
            current = current.next
        return True

    
    def update_proc(self, job, anchor_id):
        """Used for Conservative Backfilling.
        Update proc map of jobs when a new job is scheduled"""
        current = self.profile_head
        new_node = ProfileNode(self.current_timestamp, (self.current_timestamp + job.request_time), job.request_number_of_processors, job.job_id, False)
        if anchor_id is None:
            # if we cant find a place to anchor it add the node to the end of the list
            if self.profile_head is None:
                #print("debug! no valid head found, setting new one!")
                self.profile_head = new_node
            else:
                #add job at the end of the scheduling list
                current = self.profile_head
                while current is not None:
                    if current.next is None:
                        new_node.start_time = current.end_time
                        new_node.end_time = new_node.start_time + job.request_time
                        current.next = new_node
                        break 
                        #this way the job will start when the last one ends
                    current = current.next
                if new_node.start_time <= self.current_timestamp:
                    #print(f"debug! new job start time={new_node.start_time} is before {self.current_timestamp}, moving it forward")
                    new_node.start_time = self.current_timestamp
                    #if the last job in the current scheduling ends before the current timestamp, we can start this job at the current moment
                #print(f"debug! job {job.job_id} scheduled to start at={new_node.start_time}, based on job {current.id} end={current.end_time}")
        else:
            # Find the anchor node and attach the new node after it
            current = self.profile_head
            while current is not None:
                if current.id == anchor_id:
                    new_node.start_time = current.start_time
                    new_node.end_time = new_node.start_time + job.request_time
                    new_node.next = current.next
                    current.next = new_node
                    break
                current = current.next
            #print(f"debug! job {job.job_id} scheduled to start={new_node.start_time} based on job={current.id}")


    def schedule_job_conservative(self, job):
        """schedules a single job using conservative backfilling, but doesn't actually start it"""
        #print(f"debug! in schedule_job_conservative scheduling, job id = {job.job_id}")
        anchor_id, _ = self.find_anchor_point(job)
        #print(f"debug! in schedule_job_conservative scheduling, anchor_id = {anchor_id}")
        self.update_proc(job, anchor_id)
        #print(f"debug! in schedule_job_conservative scheduling, proc map updated!")


    def start_job_conservative(self, job_id, scheduled_logs):
        """starts a single job using conservative backfilling."""
        job = None
        for i in self.job_queue:
            if i.job_id == job_id:
                job = i
        #this loop gets us our job from the queue based on its id
        if job is None:
            #print(f"debug! ERROR could not find job with id={job_id}")
            #print("debug! Current job queue =")
            #print(*self.job_queue, sep=",")
            #print("skipping this job!")
            return scheduled_logs
        #if we cant find the job for whatever reason, then skip it
        assert job.scheduled_time == -1  # this job should never be scheduled before.
        job.scheduled_time = self.current_timestamp
        #print(f"job {job.job_id} started at {self.current_timestamp}, ends at {self.current_timestamp + job.run_time}")
        job.allocated_machines = self.cluster.allocate(job.job_id, job.request_number_of_processors)
        self.running_jobs.append(job)
        score = self.job_score(job)  # calculated reward
        scheduled_logs[job] = score
        self.job_queue.remove(job)
        current = self.profile_head
        while current is not None:
            if current.id == job.job_id:
                #print(f"set proc {current.id} to Running!")
                current.running = True
            current = current.next
        #sets jobs status to running in proc map
        #print("debug! in start_job_conservative scheduling, job started succesfully!")
        # if job.run_time != job.request_time:
        #     print(f"debug! discrepancy found! job run time={job.run_time} vs request={job.request_time}")
        return scheduled_logs

    def get_next_start(self):
        """used by conservative backfilling.
        goes through the proc map and returns the first jobs start time that is not already running"""
        current = self.profile_head
        min_start = sys.maxsize
        min_id = None
        while current is not None:
            if not current.running and current.start_time < min_start:
                min_start = current.start_time 
                min_id = current.id
            current = current.next
        return min_start, min_id

    def moveforward_conservative(self, score_fn):
        #print(f"debug! time is now moving forward!")
        """Used for Conservative Backfilling.
        Moves time forward based on if we need more jobs, or if we have to move to next scheduled jobs timestamp"""
        out_of_jobs = False
        if self.next_arriving_job_idx >= self.last_job_in_batch:
                #print(f"debug! run out of new jobs!")
                #assert not self.job_queue
                if not self.job_queue:
                    #print(f"debug! no jobs in queue! ending simulation!")
                    return False
                else:
                    #print("debug! still have jobs in queue, continuing forward in time until they are all gone!")
                    out_of_jobs = True
        #if there is no next job to add to the queue
        if not self.job_queue and not out_of_jobs:
            #print(f"debug! no jobs in queue, comparing next submit vs next release!")
            while not self.job_queue:
                next_resource_release_time, next_resource_release_machines= self.get_next_release()
                #print(f"debug! next submit={self.loads[self.next_arriving_job_idx].submit_time} vs next release={next_resource_release_time}")
                if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                    #if the next job arrives before or same time as release
                    self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                    #print(f"debug! moving time forward to add jobs to queue, new timestamp = {self.current_timestamp}")
                    self.job_queue.append(self.loads[self.next_arriving_job_idx])
                    self.next_arriving_job_idx += 1
                    return True     # job added
                else:
                    self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                    #print(f"debug! moving time forward to next release time, new timestamp = {self.current_timestamp}")
                    self.cluster.release(next_resource_release_machines)
                    do_compression = None
                    if len(self.running_jobs) >= 1:
                        do_compression = self.running_jobs[0].request_time > self.running_jobs[0].run_time
                    #if there is at least one running job in the queue, check if we can compress
                    running_id = self.running_jobs[0].job_id
                    self.running_jobs.pop(0)  # remove the first running job.
                    self.remove_running(running_id) #remove running job from proc map
                    if do_compression:
                        self.compress_schedule(score_fn)

        #section above handles moving time forward when queue is empty
        else:
            #print(f"debug! jobs found in queue, doing normal time progression")
            #if there are jobs in queue, focus on moving time forward based on submission vs scheduled time
            next_resource_release_time, next_resource_release_machines= self.get_next_release()
            proc_map_next_start, job_id = self.get_next_start()
            #print(f"debug! next submit={self.loads[self.next_arriving_job_idx].submit_time} vs next start={proc_map_next_start} vs next release {next_resource_release_time}")
            next_time = min(proc_map_next_start, next_resource_release_time, self.loads[self.next_arriving_job_idx].submit_time) if not out_of_jobs else min(proc_map_next_start, next_resource_release_time)
            #print(f"debug! current next_time = {next_time}")
            #find the smallest value to determine the next course of action
            #print(f"next arriving index = {self.next_arriving_job_idx}, out of jobs = {out_of_jobs}")
            if not out_of_jobs and next_time ==  self.loads[self.next_arriving_job_idx].submit_time:
                #move time forward and add job to queue
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                #print(f"debug! moving time forward to add jobs to queue, new timestamp = {self.current_timestamp}")
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            if next_time == proc_map_next_start:
                self.current_timestamp = max(self.current_timestamp, proc_map_next_start)
                #print(f"debug! moving time forward to next jobs scheduled time, new timestamp={self.current_timestamp} based on job={job_id}")
                #move to next scheduled job time so it can run
            if next_time == next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                #print(f"debug! moving time forward to next release time, new timestamp = {self.current_timestamp}")
                self.cluster.release(next_resource_release_machines)
                do_compression = self.running_jobs[0].request_time > self.running_jobs[0].run_time
                running_id = self.running_jobs[0].job_id
                self.running_jobs.pop(0)  # remove the first running job.
                self.remove_running(running_id) #remove running job from proc map
                if do_compression:
                    self.compress_schedule(score_fn)
            return True


    def get_next_release(self):
        if not self.running_jobs:  # there are no running jobs
                    next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                    next_resource_release_machines = []
                    #print(f"debug! running jobs not found, next release={next_resource_release_time}!")
                    return next_resource_release_time, next_resource_release_machines
        else:
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines
            #print(f"debug! running jobs found, next release={next_resource_release_time}")
            return next_resource_release_time, next_resource_release_machines
        
    def remove_running(self, running_id):
        """used by conservative backfilling, removes a specific job that is currently running from the proc map"""
        running_id = id
        current = self.profile_head
        if current is None:
            return
            #might need to fix at some point
        else:
            if current.id == running_id:
                #print(f"debug! running id is the profile head, substituting!")
                self.profile_head = self.profile_head.next
                return True
            prev = None

            # Traverse the linked list until the node with matching id is found
            while current and current.id != running_id:
                prev = current
                current = current.next
                return True

            # If the node is found, remove it by adjusting the pointers
            if current:
                #print(f"debug! running node found! prev={prev}, current={current}")
                prev.next = current.next
                return True
            else:
                print("Node with running_id not found!")

    def schedule_sequence_con(self, score_fn):
        """Using a score function, schedules an entire sequence using Conservative Backfilling"""
        scheduled_logs = {}
        scheduled_jobs = []
        #jobs that have already been scheduled to prevent repeats
        while True:
                #print(f"debug! current timestamp = {self.current_timestamp}")
                job_for_scheduling = None
                self.job_queue.sort(key=lambda j: score_fn(j))
                for j in self.job_queue:
                    #print(f"current amount of jobs in queue = {len(self.job_queue)}")
                    if j.job_id not in scheduled_jobs:
                        #print(f"debug! {j.job_id} not in {scheduled_jobs}")
                        #job hasn't been scheduled, so we can schedule it now
                        scheduled_jobs.append(j.job_id)
                        #TODO: error read me later!
                        job_for_scheduling = j
                        if job_for_scheduling is not None:
                            #this way we don't reschedule jobs if we don't get a new one from the previous loop
                            self.schedule_job_conservative(job_for_scheduling)
                #this tracks jobs that have already been scheduled and stops us from reschedulign the same job over and over
                current = self.profile_head
                while current is not None:
                    #if not current.running:
                        #print(f"debug! job id={current.id}, start={current.start_time}, end={current.end_time}, timestamp={self.current_timestamp}")
                    if current.start_time == self.current_timestamp and not current.running:
                        #print(f"debug! starting job {current.id}")
                        scheduled_logs = self.start_job_conservative(current.id, scheduled_logs)
                    current = current.next
                not_done = self.moveforward_conservative(score_fn)
                if not not_done:
                    break
        self.post_process_score(scheduled_logs)

        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        if self.enable_preworkloads:
            self.refill_preworkloads()

        return scheduled_logs
    
    def remove_not_running(self):
         # Create a new profile head and a current pointer
        new_profile_head = None
        current = self.profile_head
        # Traverse the linked list
        while current is not None:
            if current.running:
                # If current.running is True, keep the node in the new linked list
                if new_profile_head is None:
                    # If new_profile_head is not initialized, set it to the first node
                    new_profile_head = current
                else:
                    # Otherwise, link the current node to the previous node
                    new_profile_head.next = current
                    new_profile_head = current
            current = current.next
        
        # Set the next pointer of the last node in the new linked list to None
        if new_profile_head is not None:
            new_profile_head.next = None
        
        # Update the profile_head to the new linked list
        self.profile_head = new_profile_head

    def compress_schedule(self, score_fn):
        """used for conservative backfilling.
        called when job ends early, reschedules jobs in queue"""
        #print("debug! compressing jobs!")
        scheduled_jobs = []
        self.remove_not_running()
        #print("debug! removed all jobs not already running")
        self.job_queue.sort(key=lambda j: score_fn(j))
        for j in self.job_queue:
            job_for_scheduling = None
            if j.job_id not in scheduled_jobs:
                #print(f"debug! {j.job_id} not in {scheduled_jobs}")
                #job hasn't been scheduled, so we can schedule it now
                scheduled_jobs.append(j.job_id)
                job_for_scheduling = j
            #this tracks jobs that have already been scheduled and stops us from reschedulign the same job over and over
            if job_for_scheduling is not None:
                #this way we don't reschedule jobs if we don't get a new one from the previous loop
                #print(f"debug! compress scheduling job {job_for_scheduling.job_id}")
                self.schedule_job_conservative(job_for_scheduling)
        #print("debug! finished compressing jobs!")
"""
    CONSERVATIVE BACKFILLING
    terms
    ---
    proc map = linked list containing: job id, processors used, start time, end time, and running(bool)
    job queue = list of jobs queued to be run 
    main scheduling routine
        1. find anchor point for job
            a. scan for first position in proc map with enough available processors, if found the node becomes the anchor point
            b. scan from anchor point to end to check if enough processors remain available throughout the jobs duration
            c. return anchor point and if duration is good
        2. update proc map 
            a. if anchor point and duration are good, attach to anchor in proc map
            b. if anchor point and duration are not good, attach to the end of the proc map
        3. if we are at a jobs scheduled time, run it and set it to running, remove from queue
        4. when job ends, remove it from proc map
    compression routine (AKA if a job ends early)
        1. remove all non-running jobs from proc map
        2. re-add scheduled jobs updating their scheduling time using the new information
"""

def test_hpc_env():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = HPCEnv(batch_job_slice=100, build_sjf=True)
    env.seed(0)
    env.my_init(workload_file=workload_file, sched_file=workload_file)


DIV_LINE_WIDTH = 80

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching 
# experiments.
WAIT_BEFORE_LAUNCH = 5

color2num = dict(gray=30,
                 red=31,
                 green=32,
                 yellow=33,
                 blue=34,
                 magenta=35,
                 cyan=36,
                 white=37,
                 crimson=38)


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v)
                for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def statistics_scalar(x, with_min_and_max=False):
    x = np.array(x, dtype=np.float32)
    global_sum = np.sum(x)
    global_n = len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean)**2)

    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, global_min, global_max
    return mean, std

def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=''):
    print(('Message from %d: %s \t '%(MPI.COMM_WORLD.Get_rank(), string))+str(m))

def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()

def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()

def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)

def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff

def mpi_sum(x):
    return mpi_op(x, MPI.SUM)

def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()
    
def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    #print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads()==1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    #print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)

def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]

def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)

"""
Logging Code
"""
class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """
    def __init__(self,
                 output_dir=None,
                 output_fname='progress.txt',
                 exp_name=None):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        if proc_id()==0:
            self.output_dir = output_dir or "/tmp/experiments/%i" % int(
                time.time())
            if osp.exists(self.output_dir):
                print(
                    "Warning: Log dir %s already exists! Storing info there anyway."
                    % self.output_dir)
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            print(
                colorize("Logging data to %s" % self.output_file.name,
                        'green',
                        bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        if proc_id()==0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if proc_id()==0:
            output = json.dumps(config_json,
                                separators=(',', ':\t'),
                                indent=4,
                                sort_keys=True)
            print(colorize('Saving config:\n', color='cyan', bold=True))
            print(output)
            with open(osp.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)

    def save_state(self, state_dict, itr=None):
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_tf_saver``. 

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """
        if proc_id()==0:
            fname = 'vars.pkl' if itr is None else 'vars%d.pkl' % itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except:
                self.log('Warning: could not pickle state_dict.', color='red')
            if hasattr(self, 'tf_saver_elements'):
                self._tf_simple_save(itr)
            if hasattr(self, 'pytorch_saver_elements'):
                self._pytorch_simple_save(itr)

    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to 
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        if proc_id()==0:
            assert hasattr(self, 'pytorch_saver_elements'), \
                "First have to setup saving with self.setup_pytorch_saver"
            fpath = 'pyt_save'
            fpath = osp.join(self.output_dir, fpath)
            fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
            fname = osp.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # We are using a non-recommended way of saving PyTorch models,
                # by pickling whole objects (which are dependent on the exact
                # directory structure at the time of saving) as opposed to
                # just saving network weights. This works sufficiently well
                # for the purposes of Spinning Up, but you may want to do
                # something different for your personal PyTorch project.
                # We use a catch_warnings() context to avoid the warnings about
                # not being able to save the source code.
                torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if proc_id()==0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%' + '%d' % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-" * n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val
                print(fmt % (key, valstr))
                vals.append(val)
            print("-" * n_slashes, flush=True)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers) + "\n")
                self.output_file.write("\t".join(map(str, vals)) + "\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use 

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use 

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self,
                    key,
                    val=None,
                    with_min_and_max=False,
                    average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(
                v[0], np.ndarray) and len(v[0].shape) > 0 else v
            stats = statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key,
                                stats[0])
            if not (average_only):
                super().log_tabular('Std' + key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max' + key, stats[3])
                super().log_tabular('Min' + key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(
            v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return statistics_scalar(vals)


def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.

    If no seed is given and datestamp is false, 

    ::

        output_dir = data_dir/exp_name

    If a seed is given and datestamp is false,

    ::

        output_dir = data_dir/exp_name/exp_name_s[seed]

    If datestamp is true, amend to

    ::

        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in 
    ``spinup/user_config.py``. 

    Args:

        exp_name (string): Name for experiment.

        seed (int): Seed for random number generators used by experiment.

        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.

        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.

    Returns:

        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])

    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath),
                         exp_name=exp_name)
    return logger_kwargs


def all_bools(vals):
    return all([isinstance(v,bool) for v in vals])

def valid_str(v):
    """ 
    Convert a value or values to a string which could go in a filepath.

    Partly based on `this gist`_.

    .. _`this gist`: https://gist.github.com/seanh/93666

    """
    if hasattr(v, '__name__'):
        return valid_str(v.__name__)

    if isinstance(v, tuple) or isinstance(v, list):
        return '-'.join([valid_str(x) for x in v])

    # Valid characters are '-', '_', and alphanumeric. Replace invalid chars
    # with '-'. 
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = ''.join(c if c in valid_chars else '-' for c in str_v)
    return str_v


"""
PPO Code
"""
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        size = size * 100 # assume the traj can be really long
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
       # self.cobs_buf = np.zeros(combined_shape(size, JOB_SEQUENCE_SIZE*3), dtype=np.float32)
        self.cobs_buf = None
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.mask_buf = np.zeros(combined_shape(size, MAX_QUEUE_SIZE), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, cobs, act, mask, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
       # self.cobs_buf[self.ptr] = cobs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr < self.max_size
        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0

        actual_adv_buf = np.array(self.adv_buf, dtype = np.float32)
        actual_adv_buf = actual_adv_buf[:actual_size]

        adv_mean, adv_std = mpi_statistics_scalar(actual_adv_buf)
        '''
        # This code is doing the advantage normalization trick; should be 
        # print ("-----------------------> actual_adv_buf: ", actual_adv_buf)
        adv_sum = np.sum(actual_adv_buf)
        adv_n = len(actual_adv_buf)
        adv_mean = adv_sum / adv_n
        adv_sum_sq = np.sum((actual_adv_buf - adv_mean) ** 2)
        adv_std = np.sqrt(adv_sum_sq / adv_n)
        # print ("-----------------------> adv_std:", adv_std)
        '''
        actual_adv_buf = (actual_adv_buf - adv_mean) / adv_std
        
        data = dict(obs=self.obs_buf[:actual_size], act=self.act_buf[:actual_size], mask=self.mask_buf[:actual_size], ret=self.ret_buf[:actual_size], adv=actual_adv_buf, logp=self.logp_buf[:actual_size])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

        # return [self.obs_buf[:actual_size], self.act_buf[:actual_size], self.mask_buf[:actual_size], actual_adv_buf, self.ret_buf[:actual_size], self.logp_buf[:actual_size]]

"""
Network configurations
"""
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class RLActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # hidden_sizes = (32, 16)
        # self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.dense1 = nn.Linear(JOB_FEATURES, 32)
        self.dense2 = nn.Linear(32, 16)
        self.dense3 = nn.Linear(16, 8)
        self.dense4 = nn.Linear(8, 1)

    def _distribution(self, obs, mask):
        x = obs.view(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        logits = torch.squeeze(self.dense4(x), -1) 
        #logits = self.logits_net(obs)
        logits = logits + (mask-1)*1000000
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
    
    def forward(self, obs, mask, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        x = obs.view(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        logits = torch.squeeze(self.dense4(x), -1)
        logits = logits + (mask-1)*1000000
        pi = Categorical(logits=logits)
        #pi = self._distribution(obs, mask)

        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class RLCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        hidden_sizes = (32, 16, 8)
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, mask):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class RLActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        # build actor function
        self.pi = RLActor(obs_dim, action_space.n, hidden_sizes, activation)
        # build value function
        self.v  = RLCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, mask):
        with torch.no_grad():
            pi = self.pi._distribution(obs, mask)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs, mask)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs, mask):
        return self.step(obs, mask)[0]


"""
Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""
def ppo(workload_file, model_path, ac_kwargs=dict(), seed=0, 
        traj_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, pre_trained=0, trained_model=None, attn=False,shuffle=False, backfil=False, skip=False, score_type=0, batch_job_slice=0, heuristic="fcfs", enable_preworkloads=False):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = HPCEnv(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type, batch_job_slice=batch_job_slice, build_sjf=False, heuristic=heuristic, enable_preworkloads=False)
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)
    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = RLActorCritic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['attn'] = attn

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)


    # Inputs to computation graph

    local_traj_per_epoch = int(traj_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_traj_per_epoch * JOB_SEQUENCE_SIZE, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old, mask = data['obs'], data['act'], data['adv'], data['logp'], data['mask']

        # Policy loss
        pi, logp = ac.pi(obs, mask, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    
    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret, mask = data['obs'], data['ret'], data['mask']
        return ((ac.v(obs, mask) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        

    [o, co], r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0, 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    start_time = MPI.Wtime()
    num_total = 0
    for epoch in range(epochs):
        t = 0
        while True:
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if all(o[i:i+JOB_FEATURES] == [0]+[1]*(JOB_FEATURES-2)+[0]):
                    lst.append(0)
                elif all(o[i:i+JOB_FEATURES] == [1]*JOB_FEATURES):
                    lst.append(0)
                else:
                    lst.append(1)

            a, v_t, logp_t = ac.step(torch.as_tensor(o, dtype=torch.float32), np.array(lst).reshape(1,-1))
            #a, v_t, logp_t = ac.step(torch.as_tensor(o, dtype=torch.float32), np.array(lst).reshape(1,-1))

            num_total += 1
            '''
            action = np.random.choice(np.arange(MAX_QUEUE_SIZE), p=action_probs)
            log_action_prob = np.log(action_probs[action])
            '''

            # save and log
            buf.store(o, None, a, np.array(lst), r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, r2, sjf_t, f1_t = env.step(a[0])
            ep_ret += r
            ep_len += 1
            show_ret += r2
            sjf += sjf_t
            f1 += f1_t

            if d:
                t += 1
                buf.finish_path(r)
                logger.store(EpRet=ep_ret, EpLen=ep_len, ShowRet=show_ret, SJF=sjf, F1=f1)
                [o, co], r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0, 0, 0, 0
                if t >= local_traj_per_epoch:
                    # print ("state:", state, "\nlast action in a traj: action_probs:\n", action_probs, "\naction:", action)
                    break
        # print("Sample time:", (time.time()-start_time)/num_total, num_total)
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # start_time = time.time()
        update()
        # print("Train time:", time.time()-start_time)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)* traj_per_epoch * JOB_SEQUENCE_SIZE)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('ShowRet', average_only=True)
        logger.log_tabular('SJF', average_only=True)
        logger.log_tabular('F1', average_only=True)
        logger.log_tabular('Time', MPI.Wtime()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    #test_job_workload();
    #test_hpc_env()

    '''
    actual training code
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2 lublin_256.swf SDSC-SP2-1998-4.2-cln.swf
    parser.add_argument('--model', type=str, default='./data/lublin_256.schd')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trajs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default='./logs/ppo_temp/ppo_temp_s0')
    parser.add_argument('--attn', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)
    parser.add_argument('--heuristic', type=str, default='fcfs')
    parser.add_argument('--enable_preworkloads', type=bool, default='False')
    args = parser.parse_args()
    
    mpi_fork(args.cpu)  # run parallel code with mpi

    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, './logs/')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)

    ppo(workload_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pre_trained=0, attn=args.attn,shuffle=args.shuffle, backfil=args.backfil,
            skip=args.skip, score_type=args.score_type, batch_job_slice=args.batch_job_slice, heuristic=args.heuristic, enable_preworkloads=args.enable_preworkloads)