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

import os.path as osp
import time
import atexit
import os
import warnings
from copy import deepcopy
import os
import subprocess
import sys
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

# debug! remove me later!!!!!!
# import sys


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

    def __init__(self, line="0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0"):
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
        self.number_of_allocated_processors = max(
            self.number_of_allocated_processors, self.request_number_of_processors)
        self.request_number_of_processors = self.number_of_allocated_processors

        self.request_number_of_nodes = -1

        # if we use the job's request time field
        # for model, request_time might be empty. In this case, we set request_time to the run_time
        self.request_time = int(s_array[8])
        if self.request_time == -1 or self.request_time < self.run_time:
            self.request_time = self.run_time
        # self.request_time *= 2
        #self.request_time += self.run_time * np.random.uniform(0, 1)
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
        return "id: J["+str(self.job_id)+"]- cores: ["+str(self.request_number_of_processors)+"]- submit time: ["+str(self.submit_time)+"]- request time: ["+str(self.request_time)+"]- run time: [" + str(self.run_time) + "]- schedule time: ["+str(self.scheduled_time)+"]"

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

                if j.request_number_of_processors > self.max_procs and self.max_procs != 0:
                    j.request_number_of_processors = self.max_procs
                    #makes sure no job has more than file set max procs
                if j.request_number_of_processors > self.max:
                    self.max = j.request_number_of_processors
                    #used to display the max amount of processors we actually use


        # if max_procs = 0, it means node/proc are the same.
        if self.max_procs == 0:
            self.max_procs = self.max_nodes

        print("Max Allocated Processors:", str(self.max), "; max node:", self.max_nodes,
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
    print("Loading the workloads...")
    load = Workloads("../../../data/lublin_256.swf")
    print("Finish loading the workloads...", type(load[0]))
    print(load.max_nodes, load.max_procs)
    print(load[0].__feature__())
    print(load[1].__feature__())

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
            # print(f"debug! can allocate {job} due to enough nodes being available!")
            return True
        request_node = int(math.ceil(
            float(job.request_number_of_processors)/float(self.num_procs_per_node)))
        job.request_number_of_nodes = request_node
        if request_node > self.free_node:
            return False
        else:
            # print(f"debug! can allocate {job} due to enough proc nodes being available!")
            return True

    def allocate(self, job_id, request_num_procs):
        allocated_nodes = []
        request_node = int(
            math.ceil(float(request_num_procs) / float(self.num_procs_per_node)))

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

        print("Error in allocation, there are enough free resources but can not allocated!")
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
MAX_QUEUE_SIZE = 128 #256
MLP_SIZE = 256
MAX_WAIT_TIME = 12 * 60 * 60  # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60  # assume maximal runtime is 12 hours
JOB_FEATURES = 6 #is feature count + 1

JOB_SEQUENCE_SIZE = 256#512


class HPCEnv(gym.Env):
    # do nothing and return. A workaround for passing parameters to the environment
    def __init__(self, shuffle=False, backfil=False, job_score_type=0, build_sjf=False, heuristic="fcfs", enable_preworkloads=False, guarded=True, relax=0, act_size=1, dryrun=False):
        super(HPCEnv, self).__init__()
        print("Initialize Simple HPC Env")

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(JOB_FEATURES *
                                                   MAX_QUEUE_SIZE,),
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
        self.pre_workloads = []

        self.shuffle = shuffle
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization
        self.job_score_type = job_score_type

        self.build_sjf = build_sjf

        # section for backfilling variables
        self.rjob = 0  # rjob = relative job = job we are backfilling relative to
        self.delay = 0
        self.action_count = 0
        self.heur_easy = 0
        self.heur_easysjbf = 0
        self.heur_easyrelax = 0
        self.guarded = guarded
        self.relax = relax
        self.act_size = act_size
        self.dryrun = dryrun

        if heuristic == 'fcfs':
            self.heuristic = self.fcfs_score
        elif heuristic == 'sjf':
            self.heuristic = self.sjf_score

    def my_init(self, workload_file='', sched_file=''):
        print("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster(
            "Cluster", self.loads.max_nodes, self.loads.max_procs/self.loads.max_nodes)
        self.penalty_job_score = JOB_SEQUENCE_SIZE * self.loads.max_exec_time / 10

    def dryrun_print(self, string):
        if self.dryrun:
            print(string)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def f1_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        return (np.log10(request_time if request_time > 0 else 0.1) * request_processors + 870 * np.log10(submit_time if submit_time > 0 else 0.1))

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

    def largest_score(self, job):
        neg_request_processors = 0 - job.request_number_of_processors
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier
        return (neg_request_processors, submit_time)

    def wfp_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time-job.submit_time
        return -np.power(float(waiting_time)/request_time, 3)*request_processors

    def uni_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time-job.submit_time

        return -(waiting_time+1e-15)/(np.log2(request_processors+1e-15)*request_time)

    def fcfs_score(self, job):
        submit_time = job.submit_time
        return submit_time
    
    def saf_score(self, job):
        request_time = job.request_time
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        saf = request_time * request_processors
        return (saf, submit_time)

    def srf_score(self, job):
        request_time = job.request_time
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        srf = request_time / request_processors
        return (srf, submit_time)

    def node_score(self, job):
        nodes = job.request_number_of_processors
        return nodes

    def gen_preworkloads(self):
        # Generate some running jobs to randomly fill the cluster.
        size = np.random.randint(2 * JOB_SEQUENCE_SIZE)
        # print("ent")
        running_job_size = size
        for i in range(running_job_size):
            rand_job_index = np.random.randint(self.loads.size())
            if rand_job_index >= self.start and rand_job_index < self.last_job_in_batch:
                continue

            _job = self.loads[rand_job_index]
            req_num_of_processors = _job.request_number_of_processors
            runtime_of_job = _job.request_time
            job_tmp = Job()
            # to be different from the normal jobs; normal jobs have a job_id >= 0
            job_tmp.job_id = (-1 - rand_job_index)
            job_tmp.request_number_of_processors = req_num_of_processors
            job_tmp.run_time = runtime_of_job
            if self.cluster.can_allocated(job_tmp):
                self.running_jobs.append(job_tmp)
                job_tmp.scheduled_time = max(
                    0, (self.current_timestamp - random.randint(0, max(runtime_of_job, 1))))
                # job_tmp.scheduled_time = max(0, (self.current_timestamp - runtime_of_job/2))
                job_tmp.allocated_machines = self.cluster.allocate(
                    job_tmp.job_id, job_tmp.request_number_of_processors)
                self.pre_workloads.append(job_tmp)
            else:
                break

    def reset_for_test(self, len):
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

        # section for backfilling variables
        self.rjob = None  # rjob = relative job = job we are backfilling relative to
        self.delay = 0
        self.action_count = 0
        self.sjf_backfills = 0

        self.pre_workloads = []

        if not self.dryrun:
            self.start = np.random.randint(
                len, (self.loads.size() - len - 1))
            self.num_job_in_batch = len
        else:
            # dry run
            JOB_SEQUENCE_SIZE = self.loads.size()
            self.start = 0
            self.num_job_in_batch = JOB_SEQUENCE_SIZE

        self.start_idx_last_reset = self.start
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        if self.enable_preworkloads:
            self.gen_preworkloads()

    def reset(self):
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

        # section for backfilling variables
        self.rjob = None  # rjob = relative job = job we are backfilling relative to
        self.delay = 0
        self.action_count = 0
        self.sjf_backfills = 0
        self.backfilled_jobs = []

        self.pre_workloads = []

        global JOB_SEQUENCE_SIZE

        if not self.dryrun:
            self.start = np.random.randint(
                JOB_SEQUENCE_SIZE, (self.loads.size() - JOB_SEQUENCE_SIZE - 1))
            self.num_job_in_batch = JOB_SEQUENCE_SIZE
        else:
            # dry run
            JOB_SEQUENCE_SIZE = self.loads.size()
            self.start = 0
            self.num_job_in_batch = JOB_SEQUENCE_SIZE

        self.start_idx_last_reset = self.start
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        if self.enable_preworkloads:
            self.gen_preworkloads()

        self.heur_easy = self.easyback_schedule_curr_sequence(self.fcfs_score)
        # print(self.heur_easy)
        self.heur_easysjbf = self.easyback_schedule_curr_sequence(
            self.sjf_score)
        # print(self.heur_easysjbf)
        self.heur_bestfit = self.easyback_schedule_curr_sequence(
            self.largest_score)
        # print(self.heur_bestfit)

        # print("RL schedule")
        return self.reset_to_next_backfill_or_end()

    def reset_to_next_backfill_or_end(self):
        while True:
            while not self.job_queue:
                schedule_continue = self.jump_to_next_scheduling_point()
                if not schedule_continue:
                    return False, None

            self.job_queue.sort(key=lambda j: self.heuristic(j)) #self.fcfs_score(j)
            job_for_scheduling = self.job_queue[0]
            self.rjob = job_for_scheduling

            if self.cluster.can_allocated(job_for_scheduling):
                assert job_for_scheduling.scheduled_time == -1
                job_for_scheduling.scheduled_time = self.current_timestamp
                job_for_scheduling.allocated_machines = self.cluster.allocate(
                    job_for_scheduling.job_id, job_for_scheduling.request_number_of_processors)
                self.running_jobs.append(job_for_scheduling)
                self.job_queue.remove(job_for_scheduling)

                # after scheduling, move forward to the next scheduling point
                if self.job_queue:
                    # if job queue is not empty, just go back to schedule agin
                    continue
                else:  # if self.job_queue is empty now
                    schedule_continue = self.jump_to_next_scheduling_point()
                    if not schedule_continue:
                        return False, None
            else:
                # now, we need to backfill.
                n = self.get_number_backfillable_jobs(job_for_scheduling)

                if n > self.act_size:
                    # call RL to make decisions.
                    return True, self.build_observation(job_for_scheduling)
                elif n == 0:
                    # nothing to backfill, skip to next scheduling point
                    schedule_continue = self.jump_to_next_scheduling_point()
                    assert schedule_continue
                else:
                    # sjf backfill all and jump to next scheduling point
                    earliest_start_time = self.current_timestamp

                    self.running_jobs.sort(key=lambda running_job: (
                        running_job.scheduled_time + running_job.request_time))
                    # calculate when will be the earliest start time of "job_for_scheduling"
                    avail_procs = self.cluster.free_node * self.cluster.num_procs_per_node
                    for running_job in self.running_jobs:
                        avail_procs += len(running_job.allocated_machines) * \
                            self.cluster.num_procs_per_node
                        earliest_start_time = running_job.scheduled_time + running_job.request_time
                        if avail_procs >= job_for_scheduling.request_number_of_processors:
                            break

                    expected_waitingtime = earliest_start_time - job_for_scheduling.submit_time

                    # try to backfill all possible jobs.
                    # self.job_queue.sort(key=lambda _j: self.sjf_score(_j))
                    self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
                    for _j in self.job_queue:
                        if _j != job_for_scheduling:
                            if (self.current_timestamp + _j.request_time) < (earliest_start_time + expected_waitingtime * self.relax) and self.cluster.can_allocated(_j):
                                assert _j.scheduled_time == -1
                                _j.scheduled_time = self.current_timestamp
                                _j.allocated_machines = self.cluster.allocate(
                                    _j.job_id, _j.request_number_of_processors)
                                self.running_jobs.append(_j)
                                self.job_queue.remove(_j)

                    # Move to the next timestamp
                    schedule_continue = self.jump_to_next_scheduling_point()
                    if not schedule_continue:
                        return False, None

    def finish_rl_traj(self):
        # done, calculate rewards
        rl_total = self.post_process_score()
        # print("RLBackfill:", rl_total)
        # fcfs = self.scheduled_score[0]
        rwd2 = -rl_total
        if self.action_count == 0:
            self.action_count += 1
            # added to fix divide by 0 bug
        rwd = (self.heur_easysjbf - rl_total) / self.heur_easysjbf
        # rwd = (self.heur_easy - rl_total) #/ self.heur_easy

        # if rwd < 0:

        #     rwd = -1

        # elif rwd == 0:

        #     rwd = 0

        # else:

        #     rwd = 1

        bf_total, sched_total = self.post_process_score_bfheur()
        # - self.delay/self.action_count  # added to fix divide by 0 bug
        return [None, rwd, True, rwd2, self.heur_easysjbf, self.heur_easy, bf_total, sched_total]
    
    def finish_rl_traj_test(self):
        # done, calculate rewards
        rl_total = self.post_process_score()
        # print("RLBackfill:", rl_total)
        # fcfs = self.scheduled_score[0]
        rwd2 = self.heur_easy 
        if self.action_count == 0:
            self.action_count += 1
            # added to fix divide by 0 bug
        rwd = -rl_total

        # - self.delay/self.action_count  # added to fix divide by 0 bug
        return [None, rwd, True, rwd2, self.heur_easysjbf, self.heur_easy]

    def post_process_score(self):
        scheduled_logs = []
        bsld, wait, utils, turnaround, sld = [0, 0, 0, 0, 0]

        for i in range(self.start, self.last_job_in_batch):
            scheduled_logs.append(self.loads[i])

        for job in scheduled_logs:
            bsld += (self.job_score(job, 0)/self.num_job_in_batch)
            wait += (self.job_score(job, 1)/self.num_job_in_batch)
            turnaround += (self.job_score(job, 2) / self.num_job_in_batch)
            sld += (self.job_score(job, 4) / self.num_job_in_batch)

        # utiilization
        # why do this? +JOB_SEQUENCE_SIZE//100
        start = self.loads[self.start].submit_time
        end = self.loads[self.last_job_in_batch-1].submit_time
        # start = 0.15*(self.current_timestamp-self.loads[self.start].submit_time)+self.loads[self.start].submit_time
        # end = 0.85*(self.current_timestamp-self.loads[self.start].submit_time)+self.loads[self.start].submit_time
        total_cpu_hour = (end - start)*self.loads.max_procs
        s = 0
        for job in scheduled_logs:
            calc_start = max(job.scheduled_time, start)
            calc_end = min(job.scheduled_time+job.run_time, end)
            dur = max(0, (calc_end - calc_start))
            s += float(dur * job.request_number_of_processors)
        s /= float(total_cpu_hour)
        utils = s

        match self.job_score_type:
            case 0:
                # print("return bsld")
                return bsld
            case 1:
                # print("return wait")
                return wait
            case 2:
                # print("return turn")
                return turnaround
            case 3:
                # print("return util")
                return utils
            case 4:
                # print("return sld")
                return sld

    def test_post_process_score(self):
        scheduled_logs = []
        bsld, wait, utils, turnaround, sld = [0, 0, 0, 0, 0]

        for i in range(self.start, self.last_job_in_batch):
            scheduled_logs.append(self.loads[i])

        for job in scheduled_logs:
            bsld += (self.job_score(job, 0)/self.num_job_in_batch)
            wait += (self.job_score(job, 1)/self.num_job_in_batch)
            turnaround += (self.job_score(job, 2) / self.num_job_in_batch)
            sld += (self.job_score(job, 4) / self.num_job_in_batch)

        # utiilization
        # why do this? +JOB_SEQUENCE_SIZE//100
        start = self.loads[self.start].submit_time
        end = self.loads[self.last_job_in_batch-1].submit_time
        # start = 0.15*(self.current_timestamp-self.loads[self.start].submit_time)+self.loads[self.start].submit_time
        # end = 0.85*(self.current_timestamp-self.loads[self.start].submit_time)+self.loads[self.start].submit_time
        total_cpu_hour = (end - start)*self.loads.max_procs
        s = 0
        for job in scheduled_logs:
            calc_start = max(job.scheduled_time, start)
            calc_end = min(job.scheduled_time+job.run_time, end)
            dur = max(0, (calc_end - calc_start))
            s += float(dur * job.request_number_of_processors)
        s /= float(total_cpu_hour)
        utils = s

        return bsld, wait, utils, turnaround, sld

    def post_process_score_bfheur(self):
        #returns the bsld of backfilled and heuristic scheduled jobs
        scheduled_logs = []
        backfilled_logs = []
        bf_bsld, sched_bsld = 0, 0
        #we are going to score the bsld of the jobs backfilled and the jobs scheduled heuristically seperately

        for i in range(self.start, self.last_job_in_batch):
            if self.loads[i].job_id in self.backfilled_jobs:
                backfilled_logs.append(self.loads[i])
            else:
                scheduled_logs.append(self.loads[i])
        #create 2 logs storing info about jobs scheduled and jobs backfilled

        for job in scheduled_logs:
            sched_bsld += (self.job_score(job, 0)/self.num_job_in_batch)
        for job in backfilled_logs:
            bf_bsld += (self.job_score(job, 0)/self.num_job_in_batch)


        return -bf_bsld, -sched_bsld
    
    
    
    # run the sequence and return all the needed metrics values.
    def test_schedule_curr_sequence(self, sort_fn):
        assert self.job_queue
        # main scheduling loop
        while True:
            job_for_scheduling = None
            early_stop = False

            while not self.job_queue:
                schedule_continue = self.jump_to_next_scheduling_point()
                if not schedule_continue:
                    early_stop = True
                    break

            if early_stop:
                break

            # greedy scheduler
            self.job_queue.sort(key=lambda j: sort_fn(j))
            job_for_scheduling = self.job_queue[0]

            # can not schedule "job_for_scheduling" job
            if not self.cluster.can_allocated(job_for_scheduling):
                earliest_start_time = self.current_timestamp

                self.running_jobs.sort(key=lambda running_job: (
                    running_job.scheduled_time + running_job.request_time))
                # calculate when will be the earliest start time of "job_for_scheduling"
                avail_procs = self.cluster.free_node * self.cluster.num_procs_per_node
                for running_job in self.running_jobs:
                    avail_procs += len(running_job.allocated_machines) * \
                        self.cluster.num_procs_per_node
                    earliest_start_time = running_job.scheduled_time + running_job.request_time
                    if avail_procs >= job_for_scheduling.request_number_of_processors:
                        break

                expected_waitingtime = earliest_start_time - job_for_scheduling.submit_time

                while not self.cluster.can_allocated(job_for_scheduling):
                    self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
                    for _j in self.job_queue:
                        if _j != job_for_scheduling:
                            # self.dryrun_print("Backfill: try" + str(_j))
                            if (self.current_timestamp + _j.request_time) < (earliest_start_time + expected_waitingtime * self.relax) and self.cluster.can_allocated(_j):
                                # this job can fit based on time and resource needs
                                # self.dryrun_print("Backfill: yes" + str(_j))
                                assert _j.scheduled_time == -1
                                _j.scheduled_time = self.current_timestamp
                                _j.allocated_machines = self.cluster.allocate(
                                    _j.job_id, _j.request_number_of_processors)
                                self.running_jobs.append(_j)
                                self.job_queue.remove(_j)

                    # self.dryrun_print("Backfill: No more jobs backfillable, move to next ts")
                    # Move to the next timestamp
                    self.jump_to_next_scheduling_point()
                    # self.dryrun_print("Backfill: Move to " + str(self.current_timestamp))

            # now we can schedule job_for_scheduling
            # self.dryrun_print("schedule " + str(job_for_scheduling))
            assert job_for_scheduling.scheduled_time == -1
            job_for_scheduling.scheduled_time = self.current_timestamp
            job_for_scheduling.allocated_machines = self.cluster.allocate(
                job_for_scheduling.job_id, job_for_scheduling.request_number_of_processors)
            self.running_jobs.append(job_for_scheduling)
            self.job_queue.remove(job_for_scheduling)

            # after scheduling, move forward the timeline
            if self.job_queue:  # if job queue is not empty, just go back to schedule agin
                continue
            else:  # if can not move to the next scheduling point, break
                if not self.jump_to_next_scheduling_point():
                    break

        # after finish all scheduling.
        bsld, wait, utils, turnaround, sld = self.test_post_process_score()

        # reset again but do not change sampled index
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

        return bsld, wait, utils, turnaround, sld

    # build_observation should not include the selected job.

    def easyback_schedule_curr_sequence(self, sort_fn):
        assert self.job_queue
        # self.dryrun_print("\n-------easyback schedule--------" + sort_fn.__name__)
        # main scheduling loop
        while True:
            job_for_scheduling = None
            early_stop = False

            while not self.job_queue:
                schedule_continue = self.jump_to_next_scheduling_point()
                if not schedule_continue:
                    early_stop = True
                    break

            if early_stop:
                break

            # greedy scheduler
            self.job_queue.sort(key=lambda j: self.fcfs_score(j))
            job_for_scheduling = self.job_queue[0]
            # self.dryrun_print("schedule job: " + str(job_for_scheduling))

            # can not schedule "job_for_scheduling" job
            if not self.cluster.can_allocated(job_for_scheduling):
                # self.dryrun_print("can not schedule job: " + str(job_for_scheduling) + " try to backfill")
                earliest_start_time = self.current_timestamp

                self.running_jobs.sort(key=lambda running_job: (
                    running_job.scheduled_time + running_job.request_time))
                # calculate when will be the earliest start time of "job_for_scheduling"
                avail_procs = self.cluster.free_node * self.cluster.num_procs_per_node
                for running_job in self.running_jobs:
                    avail_procs += len(running_job.allocated_machines) * \
                        self.cluster.num_procs_per_node
                    earliest_start_time = running_job.scheduled_time + running_job.request_time
                    if avail_procs >= job_for_scheduling.request_number_of_processors:
                        break

                expected_waitingtime = earliest_start_time - job_for_scheduling.submit_time

                while not self.cluster.can_allocated(job_for_scheduling):
                    self.job_queue.sort(key=lambda _j: sort_fn(_j))
                    for _j in self.job_queue:
                        if _j != job_for_scheduling:
                            # self.dryrun_print("Backfill: try" + str(_j))
                            if (self.current_timestamp + _j.request_time) < (earliest_start_time + expected_waitingtime * self.relax) and self.cluster.can_allocated(_j):
                                # this job can fit based on time and resource needs
                                # self.dryrun_print("Backfill: yes" + str(_j))
                                assert _j.scheduled_time == -1
                                _j.scheduled_time = self.current_timestamp
                                _j.allocated_machines = self.cluster.allocate(
                                    _j.job_id, _j.request_number_of_processors)
                                self.running_jobs.append(_j)
                                self.job_queue.remove(_j)

                    # self.dryrun_print("Backfill: No more jobs backfillable, move to next ts")
                    # Move to the next timestamp
                    self.jump_to_next_scheduling_point()
                    # self.dryrun_print("Backfill: Move to " + str(self.current_timestamp))

            # now we can schedule job_for_scheduling
            # self.dryrun_print("schedule " + str(job_for_scheduling))
            assert job_for_scheduling.scheduled_time == -1
            job_for_scheduling.scheduled_time = self.current_timestamp
            job_for_scheduling.allocated_machines = self.cluster.allocate(
                job_for_scheduling.job_id, job_for_scheduling.request_number_of_processors)
            self.running_jobs.append(job_for_scheduling)
            self.job_queue.remove(job_for_scheduling)

            # after scheduling, move forward the timeline
            if self.job_queue:  # if job queue is not empty, just go back to schedule agin
                continue
            else:  # if can not move to the next scheduling point, break
                if not self.jump_to_next_scheduling_point():
                    break

        # after finish all scheduling.
        w = self.post_process_score()

        # reset again but do not change sampled index
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

        return w

    # build_observation should not include the selected job.

    def optimize_jobs(self, jobs):
        """
        given a list of jobs, chooses the highest scoring from a given list of heuristics. 
        currently only picks top ranked job for each heuristic; might lead to issues later?idk
        """
        optimized_jobs = []
        heuristics = [self.fcfs_score, self.sjf_score, self.saf_score, self.f1_score, self.node_score, self.srf_score] #heuristics to sort with
        heur_rank = [0 for i in heuristics] #tracks heuristic rank, decreases after each time selected, 
        added_jobs = set() #tracks jobs already added
        optim_len = min(len(heuristics), len(jobs)) 
        #if we have more jobs than heuristics, only pick the highest scoring of each heur
        # print(f"debug! optim_len = {optim_len}")
        for i in range(0, optim_len):
            current_heur = i % len(heuristics) #this way we can iterate through each heuristic
            heur_rank[current_heur] += 1 #increase ranking since we using this heuristic
            # print(f"debug! current_heur={current_heur}")
            #we dont increment in the selection statement to force it to move onto next ranked job
            jobs.sort(key=lambda job: heuristics[current_heur](job))
            #sort using heuristic
            new_job = jobs[heur_rank[current_heur]]
            if new_job.job_id not in added_jobs:
                #if job hasn't been added, add it to the optimized jobs
                added_jobs.add(new_job.job_id)
                optimized_jobs.append(new_job)
        return optimized_jobs

    
    def build_observation(self, selected_job):
        # we include selected_job features into each vector
        vector = np.zeros((MAX_QUEUE_SIZE) * JOB_FEATURES, dtype=float)

        # calculate the earliest start time
        earliest_start_time = self.current_timestamp
        self.running_jobs.sort(key=lambda running_job: (
            running_job.scheduled_time + running_job.request_time))
        # calculate when will be the earliest start time of "selected_job"
        avail_procs = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            avail_procs += len(running_job.allocated_machines) * \
                self.cluster.num_procs_per_node
            earliest_start_time = running_job.scheduled_time + running_job.request_time
            if avail_procs >= selected_job.request_number_of_processors:
                break
        expected_waitingtime = earliest_start_time - selected_job.submit_time

        self.can_backfilled_jobs = []

        for _j in self.job_queue:
            if _j == selected_job:
                continue
            if self.guarded:
                if (self.current_timestamp + _j.request_time) < (earliest_start_time + self.relax * expected_waitingtime) and self.cluster.can_allocated(_j):
                    self.can_backfilled_jobs.append(_j)
            else:
                if self.cluster.can_allocated(_j):
                    self.can_backfilled_jobs.append(_j)

        #self.can_backfilled_jobs = self.optimize_jobs(self.can_backfilled_jobs)            

    

        self.visible_jobs = []
        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.can_backfilled_jobs):
                self.visible_jobs.append(self.can_backfilled_jobs[i])
            else:
                break
        # self.visible_jobs.sort(key=lambda j: self.fcfs_score(j))
        if self.shuffle:
            random.shuffle(self.visible_jobs)

        self.pairs = []

        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.visible_jobs) and i < (MAX_QUEUE_SIZE) and not len(self.can_backfilled_jobs) == 0:
                job = self.visible_jobs[i]
                v = self.job_feature_vector(
                    job, earliest_start_time, expected_waitingtime, selected_job)
                v = [job] + v
                self.pairs.append(v)
            else:
                v = [0, 1, 1, 1, 1, 0]
                #v = [0, 1, 1, 0]
                v = [None] + v
                self.pairs.append(v)

        for i in range(0, MAX_QUEUE_SIZE):
            vector[i*JOB_FEATURES:(i+1)*JOB_FEATURES] = self.pairs[i][1:]

        return vector

    def job_feature_vector(self, job, earliest_start_time, expected_waitingtime, selected_job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        wait_time = self.current_timestamp - (submit_time)

        wait_max = max(self.current_timestamp - j.submit_time for j in self.can_backfilled_jobs) + 1
        real_earliest_st = (earliest_start_time +
                            self.relax * expected_waitingtime) + 1 #added +1 to avoid scenario where earliest_start == current_time
        gap_time = real_earliest_st - self.current_timestamp 
        assert self.cluster.free_node >= job.request_number_of_nodes
        # gap_resources = (self.cluster.free_node + 1) #- job.request_number_of_nodes
        gap_resources = (self.cluster.free_node + 1) 
        # #maybe this should be different?look this over again

        if gap_resources == 0:
            print(f"debug! self.cluster.free_node={self.cluster.free_node}, job.request_number_of_nodes={job.request_number_of_nodes}")
        if gap_time == 0 or gap_time + 1 == 0:
            print(f"debug! real_earliest_st={real_earliest_st}, self.current_timestamp + job.request_time={self.current_timestamp}")

        # make sure that larger value is better.
        normalized_wait_time = min(
            float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5) #float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5) 
        normalized_run_time = min(
            float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5) #self.loads.max_exec_time
        normalized_request_nodes = min(
            float(request_processors) / float(self.loads.max_procs),  1.0 - 1e-5) #self.loads.max_procs
        #print(f"debug! {request_processors / gap_resources }, actual normalized nodes={normalized_request_nodes}")
        

        # develop a set of useful features.
        # real_earliest_st = (earliest_start_time +
        #                     self.relax * expected_waitingtime)
        # gap_time = real_earliest_st - \
        #     (self.current_timestamp)
        normalized_gap_time = min(float(request_time) / float(gap_time + 1), 1.0 - 1e-5) #this is the one that doesnt break things use this one
        # normalized_gap_time = min(
        #     float(job.request_time) / float(real_earliest_st - self.current_timestamp), 1.0 - 1e-5)

        # gap_resources = (self.cluster.free_node - job.request_number_of_nodes)
        # normalized_gap_resources = min(float(gap_resources) / float(self.loads.max_procs),  1.0 - 1e-5)
        normalized_gap_resources = min(
            float(job.request_number_of_nodes) / float(self.cluster.free_node),  1.0 - 1e-5)

        normalized_earlist_st = min(
            float(real_earliest_st - self.current_timestamp) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
        # selected_job_wait_time = self.current_timestamp - selected_job.submit_time
        # selected_job_norm_wait_time = min(float(selected_job_wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)

        selected_job_norm_run_time = min(
            float(selected_job.request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)

        selected_job_norm_request_nodes = min(float(selected_job.request_number_of_processors) / float(self.loads.max_procs),  1.0 - 1e-5)

        return [normalized_wait_time, normalized_run_time, normalized_request_nodes,
                normalized_gap_time, normalized_gap_resources, 0]
        # return [normalized_wait_time, normalized_run_time, normalized_request_nodes, 0]
        # return [normalized_wait_time, normalized_run_time, normalized_request_nodes, actual_normalized_request_nodes, 0]

    def job_score(self, job_for_scheduling, score_type):
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization 4: Average slowdown
        if score_type == 0:
            # bsld
            _tmp = max(1.0, (float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                             /
                             max(job_for_scheduling.run_time, 10)))
        elif score_type == 1:
            # wait time
            _tmp = float(job_for_scheduling.scheduled_time -
                         job_for_scheduling.submit_time)
        elif score_type == 2:
            # turnaround time
            _tmp = float(job_for_scheduling.scheduled_time -
                         job_for_scheduling.submit_time + job_for_scheduling.run_time)
        elif score_type == 3:
            # utilization
            # -float(job_for_scheduling.run_time*job_for_scheduling.request_number_of_processors)
            _tmp = job_for_scheduling
        elif score_type == 4:
            # sld
            _tmp = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)\
                / job_for_scheduling.run_time
        else:
            raise NotImplementedError

        return _tmp

    def jump_to_next_scheduling_point(self):
        # schedule nothing, just move forward to next timestamp. It should 1) add a new job; 2) finish a running job; 3) reach skip time
        if self.running_jobs:  # there are running jobs
            next_resource_release_time = sys.maxsize
            next_resource_release_machines = []
            self.running_jobs.sort(key=lambda running_job: (
                running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (
                self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            # no new job inserted, just move to release time
            if self.next_arriving_job_idx >= self.last_job_in_batch:
                self.current_timestamp = max(
                    self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.
            else:  # there are still new jobs
                # if submit time comes first, move to submission point
                if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                    self.current_timestamp = max(
                        self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                    self.job_queue.append(
                        self.loads[self.next_arriving_job_idx])
                    self.next_arriving_job_idx += 1
                else:  # if release time comes first, move to release time
                    self.current_timestamp = max(
                        self.current_timestamp, next_resource_release_time)
                    self.cluster.release(next_resource_release_machines)
                    self.running_jobs.pop(0)  # remove the first running
            return True

        else:  # if there are no running jobs
            # also no job inserted
            if self.next_arriving_job_idx >= self.last_job_in_batch:
                return False
            else:  # move to the next new job
                self.current_timestamp = max(
                    self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                return True

    def queue_has_backfillable_job(self, selected_job):
        for job in self.job_queue:
            # print (job)
            if job == selected_job:
                continue
            if self.guarded:
                # calculate the earliest start time
                earliest_start_time = self.current_timestamp
                self.running_jobs.sort(key=lambda running_job: (
                    running_job.scheduled_time + running_job.request_time))
                # calculate when will be the earliest start time of "selected_job"
                avail_procs = self.cluster.free_node * self.cluster.num_procs_per_node
                for running_job in self.running_jobs:
                    avail_procs += len(running_job.allocated_machines) * \
                        self.cluster.num_procs_per_node
                    earliest_start_time = running_job.scheduled_time + running_job.request_time
                    if avail_procs >= selected_job.request_number_of_processors:
                        break
                expected_waitingtime = earliest_start_time - selected_job.submit_time

                if (self.current_timestamp + job.request_time) < (earliest_start_time + self.relax * expected_waitingtime) and self.cluster.can_allocated(job):
                    return True
            else:
                if self.cluster.can_allocated(job):
                    return True
        return False

    def get_number_backfillable_jobs(self, selected_job):
        num_backfillable_jobs = 0
        for job in self.job_queue:
            if job == selected_job:
                continue
            if self.guarded:
                # calculate the earliest start time
                earliest_start_time = self.current_timestamp
                self.running_jobs.sort(key=lambda running_job: (
                    running_job.scheduled_time + running_job.request_time))
                # calculate when will be the earliest start time of "selected_job"
                avail_procs = self.cluster.free_node * self.cluster.num_procs_per_node
                for running_job in self.running_jobs:
                    avail_procs += len(running_job.allocated_machines) * \
                        self.cluster.num_procs_per_node
                    earliest_start_time = running_job.scheduled_time + running_job.request_time
                    if avail_procs >= selected_job.request_number_of_processors:
                        break
                expected_waitingtime = earliest_start_time - selected_job.submit_time

                if (self.current_timestamp + job.request_time) < (earliest_start_time + self.relax * expected_waitingtime) and self.cluster.can_allocated(job):
                    num_backfillable_jobs += 1
            else:
                if self.cluster.can_allocated(job):
                    num_backfillable_jobs += 1
        return num_backfillable_jobs

    def step(self, a):
        # print(self.pairs, a)
        job_for_backfilling = self.pairs[a][0]

        # job is the selected job
        job_for_scheduling = self.rjob
        assert not self.cluster.can_allocated(job_for_scheduling)

        # print("selected_job:", job)
        # print("job for backscheduling:", job_for_backfilling)

        if job_for_backfilling:
            # we do not care about earliest start time if it is RL decision
            # take the backfilling action
            self.action_count += 1
            self.backfilled_jobs.append(job_for_backfilling.job_id)
            job_for_backfilling.scheduled_time = self.current_timestamp
            job_for_backfilling.allocated_machines = self.cluster.allocate(
                job_for_backfilling.job_id, job_for_backfilling.request_number_of_processors)
            self.running_jobs.append(job_for_backfilling)
            self.job_queue.remove(job_for_backfilling)
        # else:
        #    print("no job selected by RL for backfilling")

        # try to move forward till the next backfilling time or the end of the whole scheduling and return done to the agent.
        while True:
            if self.cluster.can_allocated(job_for_scheduling):
                assert job_for_scheduling.scheduled_time == -1
                job_for_scheduling.scheduled_time = self.current_timestamp
                job_for_scheduling.allocated_machines = self.cluster.allocate(
                    job_for_scheduling.job_id, job_for_scheduling.request_number_of_processors)
                self.running_jobs.append(job_for_scheduling)
                self.job_queue.remove(job_for_scheduling)

                # after scheduling, move forward to the next scheduling point
                if self.job_queue:
                    # if job queue is not empty, just go back to schedule agin
                    job_for_scheduling = self.job_queue[0]
                    self.rjob = job_for_scheduling
                    continue
                else:  # if self.job_queue is empty now
                    schedule_continue = self.jump_to_next_scheduling_point()
                    if not schedule_continue:
                        return self.finish_rl_traj_test()
            else:
                # now, we need to backfill.
                n = self.get_number_backfillable_jobs(job_for_scheduling)

                if n > self.act_size:
                    # call RL to make decisions.
                    return [self.build_observation(job_for_scheduling), 0, False, 0, 0, 0, 0, 0] 
                elif n == 0:
                    # nothing to backfill, skip to next scheduling point
                    schedule_continue = self.jump_to_next_scheduling_point()
                    assert schedule_continue
                else:
                    # sjf backfill all and jump to next scheduling point
                    earliest_start_time = self.current_timestamp

                    self.running_jobs.sort(key=lambda running_job: (
                        running_job.scheduled_time + running_job.request_time))
                    # calculate when will be the earliest start time of "job_for_scheduling"
                    avail_procs = self.cluster.free_node * self.cluster.num_procs_per_node
                    for running_job in self.running_jobs:
                        avail_procs += len(running_job.allocated_machines) * \
                            self.cluster.num_procs_per_node
                        earliest_start_time = running_job.scheduled_time + running_job.request_time
                        if avail_procs >= job_for_scheduling.request_number_of_processors:
                            break

                    expected_waitingtime = earliest_start_time - job_for_scheduling.submit_time

                    # try to backfill all possible jobs.
                    # self.job_queue.sort(key=lambda _j: self.sjf_score(_j))
                    self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
                    for _j in self.job_queue:
                        if _j != job_for_scheduling:
                            if (self.current_timestamp + _j.request_time) < (earliest_start_time + expected_waitingtime * self.relax) and self.cluster.can_allocated(_j):
                                assert _j.scheduled_time == -1
                                self.action_count += 1
                                self.backfilled_jobs.append(job_for_backfilling.job_id)
                                _j.scheduled_time = self.current_timestamp
                                _j.allocated_machines = self.cluster.allocate(
                                    _j.job_id, _j.request_number_of_processors)
                                self.running_jobs.append(_j)
                                self.job_queue.remove(_j)

                    # Move to the next timestamp. it may end the game.
                    schedule_continue = self.jump_to_next_scheduling_point()
                    if not schedule_continue:
                        return self.finish_rl_traj()
                '''
                if (self.queue_has_backfillable_job(job)):
                    return [self.build_observation(job), 0, False, 0, 0, 0]
                else:  # nothing to backfill for current ready job
                    schedule_continue = self.jump_to_next_scheduling_point()
                    assert schedule_continue
                '''
            while not self.job_queue:
                schedule_continue = self.jump_to_next_scheduling_point()
                if not schedule_continue:
                    return self.finish_rl_traj()

            self.job_queue.sort(key=lambda j: self.fcfs_score(j))
            job_for_scheduling = self.job_queue[0]
            self.rjob = job_for_scheduling

    def step_for_test(self, a):
        job_for_backfilling = self.pairs[a][0]

        # job is the selected job
        job_for_scheduling = self.rjob
        assert not self.cluster.can_allocated(job_for_scheduling)

        if job_for_backfilling:
            # we do not care about earliest start time if it is RL decision
            # take the backfilling action
            job_for_backfilling.scheduled_time = self.current_timestamp
            job_for_backfilling.allocated_machines = self.cluster.allocate(
                job_for_backfilling.job_id, job_for_backfilling.request_number_of_processors)
            self.running_jobs.append(job_for_backfilling)
            self.job_queue.remove(job_for_backfilling)
        # else:
        #    print("no job selected by RL for backfilling")

        # try to move forward till the next backfilling time or the end of the whole scheduling and return done to the agent.
        while True:
            if self.cluster.can_allocated(job_for_scheduling):
                assert job_for_scheduling.scheduled_time == -1
                job_for_scheduling.scheduled_time = self.current_timestamp
                job_for_scheduling.allocated_machines = self.cluster.allocate(
                    job_for_scheduling.job_id, job_for_scheduling.request_number_of_processors)
                self.running_jobs.append(job_for_scheduling)
                self.job_queue.remove(job_for_scheduling)

                # after scheduling, move forward to the next scheduling point
                if self.job_queue:
                    # if job queue is not empty, just go back to schedule agin
                    job_for_scheduling = self.job_queue[0]
                    self.rjob = job_for_scheduling
                    continue
                else:  # if self.job_queue is empty now
                    schedule_continue = self.jump_to_next_scheduling_point()
                    if not schedule_continue:
                        return self.finish_rl_traj_test()
            else:
                # now, we need to backfill.
                n = self.get_number_backfillable_jobs(job_for_scheduling)

                if n > self.act_size:
                    # call RL to make decisions.
                    return [self.build_observation(job_for_scheduling), 0, False, 0, 0, 0]
                elif n == 0:
                    # nothing to backfill, skip to next scheduling point
                    schedule_continue = self.jump_to_next_scheduling_point()
                    assert schedule_continue
                else:
                    # sjf backfill all and jump to next scheduling point
                    earliest_start_time = self.current_timestamp

                    self.running_jobs.sort(key=lambda running_job: (
                        running_job.scheduled_time + running_job.request_time))
                    # calculate when will be the earliest start time of "job_for_scheduling"
                    avail_procs = self.cluster.free_node * self.cluster.num_procs_per_node
                    for running_job in self.running_jobs:
                        avail_procs += len(running_job.allocated_machines) * \
                            self.cluster.num_procs_per_node
                        earliest_start_time = running_job.scheduled_time + running_job.request_time
                        if avail_procs >= job_for_scheduling.request_number_of_processors:
                            break

                    expected_waitingtime = earliest_start_time - job_for_scheduling.submit_time

                    # try to backfill all possible jobs.
                    # self.job_queue.sort(key=lambda _j: self.sjf_score(_j))
                    self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
                    for _j in self.job_queue:
                        if _j != job_for_scheduling:
                            if (self.current_timestamp + _j.request_time) < (earliest_start_time + expected_waitingtime * self.relax) and self.cluster.can_allocated(_j):
                                assert _j.scheduled_time == -1
                                _j.scheduled_time = self.current_timestamp
                                _j.allocated_machines = self.cluster.allocate(
                                    _j.job_id, _j.request_number_of_processors)
                                self.running_jobs.append(_j)
                                self.job_queue.remove(_j)

                    # Move to the next timestamp. it may end the game.
                    schedule_continue = self.jump_to_next_scheduling_point()
                    if not schedule_continue:
                        return self.finish_rl_traj_test()

            while not self.job_queue:
                schedule_continue = self.jump_to_next_scheduling_point()
                if not schedule_continue:
                    return self.finish_rl_traj_test()

            self.job_queue.sort(key=lambda j: self.heuristic(j)) #self.fcfs_score(j)
            job_for_scheduling = self.job_queue[0]
            self.rjob = job_for_scheduling


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


DIV_LINE_WIDTH = 80

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(
    osp.dirname(osp.dirname(__file__))), 'data')

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
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
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
    if n <= 1:
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
    print(('Message from %d: %s \t ' %
          (MPI.COMM_WORLD.Get_rank(), string))+str(m))


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
    # print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    # print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)


def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs() == 1:
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
        if proc_id() == 0:
            self.output_dir = output_dir or "/tmp/experiments/%i" % int(
                time.time())
            if osp.exists(self.output_dir):
                print(
                    "Warning: Log dir %s already exists! Storing info there anyway."
                    % self.output_dir)
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(
                osp.join(self.output_dir, output_fname), 'w')
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
        if proc_id() == 0:
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
        if proc_id() == 0:
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
        if proc_id() == 0:
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
        if proc_id() == 0:
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
        if proc_id() == 0:
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
    return all([isinstance(v, bool) for v in vals])


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
        size = size * 50  # assume the traj can be really long
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
       # self.cobs_buf = np.zeros(combined_shape(size, JOB_SEQUENCE_SIZE*3), dtype=np.float32)
        self.cobs_buf = None
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)
        self.mask_buf = np.zeros(combined_shape(
            size, MAX_QUEUE_SIZE), dtype=np.float32)
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
        self.adv_buf[path_slice] = discount_cumsum(
            deltas, self.gamma * self.lam)

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

        actual_adv_buf = np.array(self.adv_buf, dtype=np.float32)
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

        data = dict(obs=self.obs_buf[:actual_size], act=self.act_buf[:actual_size], mask=self.mask_buf[:actual_size],
                    ret=self.ret_buf[:actual_size], adv=actual_adv_buf, logp=self.logp_buf[:actual_size])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

        # return [self.obs_buf[:actual_size], self.act_buf[:actual_size], self.mask_buf[:actual_size], actual_adv_buf, self.ret_buf[:actual_size], self.logp_buf[:actual_size]]


"""
Network configurations
"""


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layer = [nn.Linear(sizes[j], sizes[j+1]), act()]
        layers += layer
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
        # logits = self.logits_net(obs)
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
        # pi = self._distribution(obs, mask)

        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class RLCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        hidden_sizes = (32, 16, 8)
        #initial layer is 1024 x 32, 1024 should be obs_dim dimensions but thats wrong?
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, mask):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


class RLActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        # build actor function
        self.pi = RLActor(obs_dim, action_space.n, hidden_sizes, activation)
        # build value function
        self.v = RLCritic(obs_dim, hidden_sizes, activation)

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
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, pre_trained=0, trained_model=None, attn=False, shuffle=False, backfil=False, score_type=0, heuristic="fcfs", enable_preworkloads=False, guarded=False, relax=0, act_size=1, dryrun=False):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = HPCEnv(shuffle=shuffle, backfil=backfil, job_score_type=score_type, build_sjf=False,
                 heuristic=heuristic, enable_preworkloads=enable_preworkloads, guarded=guarded, relax=relax, act_size=act_size, dryrun=dryrun)
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
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Inputs to computation graph

    local_traj_per_epoch = int(traj_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_traj_per_epoch *
                    JOB_SEQUENCE_SIZE, gamma, lam)

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
                logger.log(
                    'Early stopping at step %d due to reaching max kl.' % i)
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

    r, d, ep_ret, ep_len, show_ret, sjf, f1, bf_r, sched_r = 0, False, 0, 0, 0, 0, 0, 0, 0

    ready, o = env.reset()
    while not ready:
        ready, o = env.reset()

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

            a, v_t, logp_t = ac.step(torch.as_tensor(
                o, dtype=torch.float32), np.array(lst).reshape(1, -1))

            num_total += 1
            '''
            action = np.random.choice(np.arange(MAX_QUEUE_SIZE), p=action_probs)
            log_action_prob = np.log(action_probs[action])
            '''

            # save and log
            buf.store(o, None, a, np.array(lst), r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, r2, sjf_t, f1_t, bf_t, sched_t = env.step(a[0])
            ep_ret += r
            ep_len += 1
            show_ret += r2
            sjf += sjf_t
            f1 += f1_t
            bf_r += bf_t
            sched_r += sched_t

            if d:
                t += 1
                buf.finish_path(r)
                logger.store(EpRet=ep_ret, EpLen=ep_len, ShowRet=show_ret, Delay=env.delay /
                             env.action_count, SJF=sjf, F1=f1, Backfills=env.action_count, BackfillScore=bf_r, HeuristicScore=sched_r)
                r, d, ep_ret, ep_len, show_ret, sjf, f1, bf_r, sched_r = 0, False, 0, 0, 0, 0, 0, 0, 0
                ready, o = env.reset()
                while not ready:
                    ready, o = env.reset()

                if t >= local_traj_per_epoch:
                    # print ("state:", state, "\nlast action in a traj: action_probs:\n", action_probs, "\naction:", action)
                    break
        # print("Sample time:", (time.time()-start_time)/num_total, num_total)
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        if dryrun:
            return

        # start_time = time.time()
        update()
        # print("Train time:", time.time()-start_time)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1) *
                           traj_per_epoch * JOB_SEQUENCE_SIZE)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('ShowRet', average_only=True)
        logger.log_tabular('Delay', average_only=True)
        logger.log_tabular('SJF', average_only=True)
        logger.log_tabular('F1', average_only=True)
        logger.log_tabular('Backfills', average_only=True)
        logger.log_tabular('Time', MPI.Wtime()-start_time)
        logger.log_tabular('BackfillScore', average_only=True)
        logger.log_tabular('HeuristicScore', average_only=True)
        logger.dump_tabular()


if __name__ == '__main__':
    # test_job_workload();
    # test_hpc_env()

    '''
    actual training code
    '''
    parser = argparse.ArgumentParser()
    # RICC-2010-2 lublin_256.swf SDSC-SP2-1998-4.2-cln.swf
    parser.add_argument('--workload', type=str,
                        default='./data/lublin_256.swf')
    parser.add_argument('--model', type=str, default='./data/lublin_256.schd')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trajs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str,
                        default='./logs/ppo_temp/ppo_temp_s0')
    parser.add_argument('--attn', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=1)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--heuristic', type=str, default='fcfs')
    parser.add_argument('--enable_preworkloads', type=bool, default=False)
    parser.add_argument('--guarded', type=bool, default=False)
    parser.add_argument('--relax', type=float, default=0)
    parser.add_argument('--act_size', type=int, default=1)
    parser.add_argument('--dryrun', type=bool, default=False)

    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, './logs/')
    logger_kwargs = setup_logger_kwargs(
        args.exp_name, seed=args.seed, data_dir=log_data_dir)

    ppo(workload_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pre_trained=0, attn=args.attn, shuffle=args.shuffle, backfil=args.backfil, score_type=args.score_type, heuristic=args.heuristic, enable_preworkloads=args.enable_preworkloads, guarded=args.guarded, relax=args.relax, act_size=args.act_size, dryrun=args.dryrun)
