# Generalized Policy Improvement (GPI) Algorithms

This repository is the official implementation for the forthcoming paper Generalized Policy Improvement Algorithms with Theoretically Supported Sample Reuse, which introduces the reinforcement learning algorithm class known as Generalized Policy Improvement (GPI) algorithms. This extends the work done in the paper [Generalized Proximal Policy Optimization with Sample Reuse](https://proceedings.neurips.cc/paper/2021/hash/63c4b1baf3b4460fa9936b1a20919bec-Abstract.html), whose code was first published at [this repository](https://github.com/jqueeney/geppo).

GPI algorithms combine the approximate policy improvement benefits of on-policy methods with theoretically supported sample reuse. As a result, these algorithms provide practical guarantees on performance while making more efficient use of data than their on-policy counterparts. The GPI framework is theoretically supported by a Generalized Policy Improvement lower bound that can be approximated using data from all recent policies.

In this repository, we include on-policy and generalized versions of three popular on-policy methods: 

1. PPO and its generalized version GePPO
2. TRPO and its generalized version GeTRPO
3. VMPO and its generalized version GeVMPO

Implementation of these algorithms follows the methodology described in the GPI paper, which in some cases differs from the implementation choices described in the original papers that introduced these on-policy algorithms.

## Requirements

The source code requires the following packages to be installed (we have included the latest version used to test the code in parentheses):

- python (3.8.13)
- dm-control (1.0.0)
- gurobi (9.5.1)
- gym (0.21.0)
- matplotlib (3.5.1)
- mujoco-py (1.50.1.68)
- numpy (1.22.3)
- scipy (1.8.0)
- seaborn (0.11.2)
- tensorflow (2.7.0)

See the file `environment.yml` for the latest conda environment used to run our code, which can be built with conda using the command `conda env create`.

Some OpenAI Gym environments and all DeepMind Control Suite environments require the MuJoCo physics engine. Please see the [MuJoCo website](https://mujoco.org/) for more information. 

GPI algorithms use Gurobi to determine the optimal policy weights for their theoretically supported sample reuse, which requires a Gurobi license. Please see the [Gurobi website](https://www.gurobi.com/downloads/) for more information on downloading Gurobi and obtaining a license. Alternatively, GPI algorithms can be run without Gurobi by using uniform policy weights with the `--uniform` option.

## Training

Simulations can be run by calling `run` on the command line. See below for examples of running PPO and GePPO on both OpenAI Gym and DeepMind Control Suite environments:

```
python -m gpi.run --env_type gym --env_name HalfCheetah-v3 --alg_name ppo
python -m gpi.run --env_type gym --env_name HalfCheetah-v3 --alg_name geppo

python -m gpi.run --env_type dmc --env_name cheetah --task_name run --alg_name ppo
python -m gpi.run --env_type dmc --env_name cheetah --task_name run --alg_name geppo
```

Hyperparameters can be changed to non-default values by using the relevant option on the command line. For more information on the inputs accepted by `run`, use the `--help` option.

The results of simulations are saved in the `logs/` folder upon completion.

## Evaluation

The results of simulations saved in the `logs/` folder can be visualized by calling `plot` on the command line:

```
python -m gpi.plot --on_file <filename> --gpi_file <filename>
```

By default, this command saves a plot of average performance throughout training in the `figs/` folder. Other metrics can be plotted using the `--metric` option. For more information on the inputs accepted by `plot`, use the `--help` option.