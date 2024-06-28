## About this repository
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
 - Folder structure 


