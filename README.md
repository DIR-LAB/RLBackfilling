# RLBackfilling Using PyTorch

> [!IMPORTANT]
> We noticed that the way we calculated the bsld is not accurate after the paper was published. Hence, we corrected it in this update (bad9bc3) and also fixed the corresponding tables/plots in the paper on [Arxiv](https://arxiv.org/abs/2404.09264)


This repo includes the bfTorch source code and necessary datasets to run the experiments/tests. 

The code has been tested on Ubuntu 18.04 with PyTorch 1.13 and Gym 0.21. 

## Installation
### Required Software
* Python 3.9 and PyTorch
Use VirtuanEnv or Conda to build a Python3.9 environment and PyTorch at least 1.13.0
Note that, we do not leverage GPUs, so no need to configure the GPU version of PyTorch.

* OpenMPI and mpi4py
```bash
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
conda install mpi4py
```

### Clone RLBackfilling
```bash
git clone https://github.com/DIR-LAB/RLBackfilling.git
```

### Install Dependencies
```shell script
cd RLBackfilling
pip install -r requirements.txt
```

### File Structure

```
data/: Contains a series of workload and real-world traces.
plot.py: Plot the trained results.
rlbackfill.py: Used to train and run RLBackfilling models, as well as tests for actual/noisy runtime performance
rlbackfill-test.py: Generates raw avgbsld scores and box+whisker plot.
```
To change the hyper-parameters, such as `MAX_OBSV_SIZE` or the trajectory length during training, you can change them in rlbackfill.py.

### Training
To train a RL model based on a job trace, run this command:
```bash
python rlbackfill.py --workload ./data/lublin_256.swf --exp_name your-exp-name --trajs 400 --epochs 300 --heuristic fcfs
```

There are many other parameters in the source file.
* `--model`, specify a saved trained model (for two-step training and re-training)
* `--pre_trained`, specify whether this trainig will be a twp-step training or re-training

### Monitor Training 

After running Default Training, a folder named `logs/your-exp-name/` will be generated. 

```bash
python plot.py ./data/logs/your-exp-name/ -x Epoch -s 1
```

It will plot the training curve.

### Test and Compare

After the RLBackfiller converges, you can test the result and compare it with different policies such as FCFS, SJF, WFP3, UNICEP, and F1.

```bash
python rlbackfill-test.py --rlmodel "./logs/your-exp-name/your-exp-name_s0/" --workload "./data/lublin_256.swf" --len 2048 --iter 10
```
There are many parameters you can use:
* `--seed`, the seed for random sampling
* `--iter`, how many iterations for the testing

### Runtime Testing

To test the how differences in runtime accuracy can affect results, modify the `self.request_time` in the `Job class` located in rlbackfill.py
For example, to test how 100% accurate request time affects scheduling, perform this change in the code
```python
self.request_time = self.run_time
```
