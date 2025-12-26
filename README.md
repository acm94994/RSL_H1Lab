# Unitree H1 Locomotion and Deployment (IsaacLab -> MuJoCo)

## Overview

This project/repository is a shot at training a policy for locomotion of a Unitree H1 in IsaacLab, in a rough environment, and the deployment of the same in a flat MuJoCo environment. 

This package was partially created using the template package creation tool in IsaacLab, and multiple modifications made on top of it.

## Installation

### IsaacLab

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Creating a new conda environment is recommended for this, and is covered in the installation guide as well.
  
- Install the python module for rsl_rl using the command below.

    ```bash
    ./isaaclab.sh -i rsl_rl
    ```

### RSL_H1Lab
- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
    
    ```bash
    git clone https://github.com/acm94994/RSL_H1Lab.git
    cd RSL_H1Lab
    ```

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/RSL_H1Lab

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        python scripts/rsl_rl/train.py --task=<TASK_NAME>
        # task names: IsaacH1RoughRSL, IsaacH1RoughPlayRSL, IsaacH1FlatRSL, IsaacH1FlatPlayRSL
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### MuJoCo
- Install using the command below.

    ```bash
    pip install mujoco==3.2.7
    ```

## Running the Scripts

### Policy
- Please refer to the IsaacLab docs for the [other parameters involved in training](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html).

- Train your policy with the command below
    ```bash
    python scripts/rsl_rl/train.py --task IsaacH1RoughRSL --headless
    ```
- Running the Play environment. This also exports your policy, which will be important in the next step.

    ```bash
    python scripts/rsl_rl/play.py --task IsaacH1RoughPlayRSL --headless --video --video_length 500 
    # For running without opening simulation, and record a video of 500 timesteps.
    ```


### Deployment in MuJoCo

- A sample policy policy.pt has been provided to test. If you want to deploy your custom policy, follow the instructions below, deleting the .pt files.
- Copy your policy to the deployment folder. 
    ```bash
    cp logs/<your-log-spanning-multiple-directories>/exported/<task>_policy_<number_iterations>.pt assets
    # <task> used here is IsaacH1RoughPlayRSL. Other task names available above.
    ```

- Run deployment with keyboard control:

    ```bash
    python deployment/deploy_mujoco.py
    ```

- Run deployment with `walk.py` for interactive keyboard control or automated command sequences:

    ```bash
    # Keyboard control (default - use W/A/S/D/Q/E keys)
    python deployment/walk.py
    
    # Automated mode (1s wait, 5s forward, 1s wait, 3s rotate left, 1s wait, 3s rotate right, 1s wait, 3s forward)
    python deployment/walk.py --no-keyboard
    
    # Custom policy path
    python deployment/walk.py --policy_path assets/IsaacH1FlatPlayRSL_policy_7999_iterations.pt
    
    # Adjust control parameters
    python deployment/walk.py --action_scale 0.5 --vel_scale_x 0.5 --vel_scale_rot 1.0 --n_substeps 4
    ```

- To verify the PyTorch joint indexes, run the following commands.
    ```bash
    # if you are in the deployment directory
    cd .. 
    python tests/h1_joints.py

    ```
