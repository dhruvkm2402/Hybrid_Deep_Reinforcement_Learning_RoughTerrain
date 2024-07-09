# About this repository
 We demonstrate the effectiveness of using Hybrid Deep Reinforcement Learning architecture for rough terrain navigation of an ackermann-steered platform called AgileX HunterSE. The algorithm combines DRL and LQR  controller and is deployed on NVIDIA's Isaac Sim simulator. 
 Following are the instructions to replicate the scenario. 

https://github.com/dhruvkm2402/Hybrid_Deep_Reinforcement_Learning_RoughTerrain/assets/99369975/2c1b83c5-27fc-4368-8b76-4fa354ec7e06

https://github.com/dhruvkm2402/Hybrid_Deep_Reinforcement_Learning_RoughTerrain/assets/99369975/998968f1-5e92-431b-b2b4-f48adca7ad45


 
# Omniverse Isaac Gym Reinforcement Learning Environments for Isaac Sim

*PLEASE NOTE: Version 4.0.0 will be the last release of OmniIsaacGymEnvs. Moving forward, OmniIsaacGymEnvs will be merging with IsaacLab (https://github.com/isaac-sim/IsaacLab). All future updates will be available as part of the IsaacLab repository.*

For tutorials on migrating to IsaacLab, please visit: https://isaac-sim.github.io/IsaacLab/source/migration/migrating_from_omniisaacgymenvs.html.



This repository contains Reinforcement Learning examples that can be run with the latest release of [Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html). RL examples are trained using PPO from [rl_games](https://github.com/Denys88/rl_games) library and examples are built on top of Isaac Sim's `omni.isaac.core` and `omni.isaac.gym` frameworks.

Please see [release notes](docs/release_notes.md) for the latest updates.

## System Requirements

It is recommended to have at least 32GB RAM and a GPU with at least 12GB VRAM. For detailed system requirements, please visit the [Isaac Sim System Requirements](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html#system-requirements) page. Please refer to the [Troubleshooting](docs/troubleshoot.md#memory-consumption) page for a detailed breakdown of memory consumption.

## Installation

Follow the Isaac Sim [documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) to install the latest Isaac Sim release. 

*Examples in this repository rely on features from the most recent Isaac Sim release. Please make sure to update any existing Isaac Sim build to the latest release version, 4.0.0, to ensure examples work as expected.*

Once installed, this repository can be used as a python module, `omniisaacgymenvs`, with the python executable provided in Isaac Sim.

To install `omniisaacgymenvs`, first clone this repository:

```bash
git clone https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git
```

Once cloned, locate the [python executable in Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html). By default, this should be `python.sh`. We will refer to this path as `PYTHON_PATH`.

To set a `PYTHON_PATH` variable in the terminal that links to the python executable, we can run a command that resembles the following. Make sure to update the paths to your local path.

```
For Linux: alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
For Windows: doskey PYTHON_PATH=C:\Users\user\AppData\Local\ov\pkg\isaac_sim-*\python.bat $*
For IsaacSim Docker: alias PYTHON_PATH=/isaac-sim/python.sh
```

Install `omniisaacgymenvs` as a python module for `PYTHON_PATH`:

```bash
PYTHON_PATH -m pip install -e .
```

The following error may appear during the initial installation. This error is harmless and can be ignored.

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
```
Additional setup and example details can be found at https://github.com/isaac-sim/OmniIsaacGymEnvs
### Running Rough Path Tracking using Hybrid Deep Reinforcement Learning Example
 - Folder structure (Pointing out some necessary files in order to successfully train/evaluate the scenario)
``` 
├── omniisaacgymenvs
│   ├── cfg
│   │   ├── config.yaml
│   │   ├── task
│   │   │   ├── HunterTaskE2E.yaml
│   │   │   ├── HunterTask.yaml
│   │   └── train
│   │       ├── HunterTaskE2EPPO.yaml
│   │       ├── HunterTaskE2ESAC.yaml
│   │       ├── HunterTaskPPO.yaml
│   ├── robots
│   │   └── articulations
│   │       ├── hunter.py
│   ├── scripts
│   │   ├── random_policy.py
│   │   ├── rlgames_demo.py
│   │   └── rlgames_train.py
│   ├── tasks
│   │   ├── HunterTask_PyTorchE2E.py
│   │   ├── HunterTask_PyTorch.py
│   ├── USD_Files
│   │   └── hunter_aim4.usd
│   ├── utils
│   │   ├── task_util.py
│   └── Waypoints
│       ├── Austin_centerline2.csv
│       ├── BrandsHatch_centerline.csv
│       └── Silverstone_centerline.csv
├── README.md
├── setup.py
├── terrainfile_link.txt
```
- Config files: Simulation and DRL algorithm related parameters
- hunter.py: Imports the USD file with a specified confiuration
- Task files: HDRL (HunterTask_PyTorch.py) and End-to-End DRL (HunterTask_PyTorchE2E.py), These are scripts where states, actions, rewards, observations, etc. are defined.
- hunter_aim4.usd: USD file of the robot AgileX HunterSE
- task_util.py: Define task keys for running training and evaluation commands
- terrainfile_link.txt: Rough terrain USD file
### Important: Make sure to replace the directory in task file for each of the waypoints and terrain as "home/your_username/..." and add the terrain USD file in USD_Files folder
### Running the HDRL training (Be in the OmniIsaacGymEnvs/omniisaacgymenvs directory)
'''
PYTHON_PATH scripts/rlgames_train.py task=HunterTask headless=True
'''
### Running the End-toEnd DRL PPO training (Be in the OmniIsaacGymEnvs/omniisaacgymenvs directory)
'''
PYTHON_PATH scripts/rlgames_train.py task=HunterTaskE2E headless=True
'''
### Running the End-toEnd DRL SAC training (Be in the OmniIsaacGymEnvs/omniisaacgymenvs directory)
- Also, running SAC gives an error, kindly change the sac_agent.py located in the following directory
  '''
   cd ~/.local/share/ov/pkg/isaac-sim-4.0.0/kit/python/lib/python3.10/site-packages/rl_games/algos_torch
  '''
and modify lines 494 - 497 to the following. The issue is reported here https://github.com/Denys88/rl_games/issues/263 
'''
if isinstance(next_obs, dict):    
                next_obs_processed = next_obs['obs']
                self.obs = next_obs_processed.clone()
            else:
                self.obs = next_obs.clone()
'''   
'''
PYTHON_PATH scripts/rlgames_train.py task=HunterTaskE2E train=HunterTaskE2ESAC headless=True
'''
The trained models are saved in the run folder
To use tensorbaord
'''
PYTHON_PATH -m tensorboard.main --logdir runs/EXPERIMENT_NAME/summaries
'''
In order to evaluate the trained agents run the following code as per the task
'''
PYTHON_PATH scripts/rlgames_train.py task=Ant checkpoint=runs/Experiment_Name/nn/Experiment_name.pth test=True num_envs=64
'''

### Acknowledgements
We'd like to acknowledge the following references:
'''
@misc{makoviychuk2021isaac,
      title={Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning}, 
      author={Viktor Makoviychuk and Lukasz Wawrzyniak and Yunrong Guo and Michelle Lu and Kier Storey and Miles Macklin and David Hoeller and Nikita Rudin and Arthur Allshire and Ankur Handa and Gavriel State},
      year={2021},
      journal={arXiv preprint arXiv:2108.10470}
}
'''
'''
@misc{rudin2021learning,
      title={Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning}, 
      author={Nikita Rudin and David Hoeller and Philipp Reist and Marco Hutter},
      year={2021},
      journal = {arXiv preprint arXiv:2109.11978}
}
'''
'''
@unknown{unknown,
author = {Sakai, Atsushi and Ingram, Daniel and Dinius, Joseph and Chawla, Karan and Raffin, Antonin and Paques, Alexis},
year = {2018},
month = {08},
pages = {},
title = {PythonRobotics: a Python code collection of robotics algorithms}
}
https://github.com/AtsushiSakai/PythonRobotics
'''

