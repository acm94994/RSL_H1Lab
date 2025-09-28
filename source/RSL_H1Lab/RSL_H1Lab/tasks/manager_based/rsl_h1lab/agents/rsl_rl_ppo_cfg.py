# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import torch

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

def mirror_joint_tensor(original: torch.Tensor, mirrored: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Mirror a tensor of joint values by swapping left/right pairs and inverting yaw/roll joints.
    
    Args:
        original: Input tensor of shape [..., num_joints] where num_joints is 23
        mirrored: Output tensor of same shape to store mirrored values
        offset: Optional offset to add to indices if tensor has additional dimensions
        
    Returns:
        Mirrored tensor with same shape as input
    """
    # Define pairs of indices to swap (left/right pairs)
    swap_pairs = [
        (2 + offset, 7 + offset),   # hip_pitch
        (1 + offset, 6 + offset),   # hip_roll
        (0 + offset, 5 + offset),   # hip_yaw
        (3 + offset, 8 + offset),   # knee
        (11 + offset, 15 + offset),  # shoulder_pitch
        (4 + offset, 9 + offset), # ankle
        (12 + offset, 16 + offset), # shoulder_roll
        # (15 + offset, 16 + offset), # ankle_roll
        (13 + offset, 17 + offset), # shoulder_yaw
        (14 + offset, 18 + offset), # elbow
        # (21 + offset, 22 + offset)  # wrist_roll
    ]
    
    # Define indices that need to be inverted (yaw/roll joints)
    invert_indices = [
        # 0 + offset,   # waist_yaw
        # 5 + offset, 
        1 + offset,   # left_hip_roll
        6 + offset,   # right_hip_roll
        0 + offset,   # left_hip_yaw
        5 + offset,   # right_hip_yaw
        12 + offset,  # left_shoulder_roll
        16 + offset,  # right_shoulder_roll
        4 + offset,  # left_ankle_roll
        9 + offset,  # right_ankle_roll
        13 + offset,  # left_shoulder_yaw
        17 + offset,  # right_shoulder_yaw
        14 + offset,  # left_elbow
        18 + offset,  # right_elbow
        # 21 + offset,  # left_wrist_roll
        # 22 + offset   # right_wrist_roll
    ]
    
    # First copy non-swapped, non-inverted values
    non_swap_indices = [i for i in range(original.shape[-1]) if i not in [idx for pair in swap_pairs for idx in pair]]
    mirrored[..., non_swap_indices] = original[..., non_swap_indices]
    
    # Swap left/right pairs
    for left_idx, right_idx in swap_pairs:
        mirrored[..., left_idx] = original[..., right_idx]
        mirrored[..., right_idx] = original[..., left_idx]
    
    # Invert yaw/roll joints
    mirrored[..., invert_indices] = -mirrored[..., invert_indices]
    



def mirror_observation_policy(obs):
    if obs is None:
        return obs
    
    # _obs = torch.clone(obs)
    # flipped_obs = torch.clone(obs)

    # # print(flipped_obs.shape)
    # # Mirror projected gravity (flip y)
    # flipped_obs[..., 1] = -_obs[..., 1]  # y component of projected_gravity

    
    # # Mirror velocity commands (flip y and z)
    # flipped_obs[..., 4] = -_obs[..., 4]  # y component of velocity_commands
    # flipped_obs[..., 5] = -_obs[..., 5]  # z component of velocity_commands

    # mirror_joint_tensor(_obs,flipped_obs,6) 
    # mirror_joint_tensor(_obs,flipped_obs,25) 
    # mirror_joint_tensor(_obs,flipped_obs,44) 

    # # print(torch.vstack((_obs, flipped_obs)).shape)

    # return torch.vstack((_obs, flipped_obs))
    if hasattr(obs, "keys") and "policy" in obs.keys():
        _obs = obs["policy"].clone()
        flipped_obs = _obs.clone()

        # Mirror projected gravity (flip y)
        flipped_obs[..., 1] = -_obs[..., 1]
        # Mirror velocity commands (flip y and z)
        flipped_obs[..., 4] = -_obs[..., 4]
        flipped_obs[..., 5] = -_obs[..., 5]

        mirror_joint_tensor(_obs, flipped_obs, 6)
        mirror_joint_tensor(_obs, flipped_obs, 25)
        mirror_joint_tensor(_obs, flipped_obs, 44)

        # Stack along batch dimension
        # stacked = torch.cat([_obs, flipped_obs], dim=0)
        # Return a new TensorDict with the stacked tensor
        new_obs = obs.clone(False)
        new_obs["policy"] = flipped_obs
        return new_obs

    # If obs is a plain tensor
    _obs = torch.clone(obs)
    flipped_obs = torch.clone(obs)
    flipped_obs[..., 1] = -_obs[..., 1]
    flipped_obs[..., 4] = -_obs[..., 4]
    flipped_obs[..., 5] = -_obs[..., 5]
    mirror_joint_tensor(_obs, flipped_obs, 6)
    mirror_joint_tensor(_obs, flipped_obs, 25)
    mirror_joint_tensor(_obs, flipped_obs, 44)
    # return torch.cat([_obs, flipped_obs], dim=0)
    return flipped_obs

# def mirror_observation_critic(obs):
#     if obs is None:
#         return obs
    
#     # print(obs.shape)
    
#     _obs = torch.clone(obs)
#     flipped_obs = torch.clone(obs)
#     # Mirror base linear velocity (flip y)
#     flipped_obs[..., 1] = -_obs[..., 1]  # y component of base_lin_vel
    
#     # Mirror base angular velocity (flip z)
#     flipped_obs[..., 5] = -_obs[..., 5]  # z component of base_ang_vel
    
#     # Mirror projected gravity (flip y)
#     flipped_obs[..., 7] = -_obs[..., 7]  # y component of projected_gravity
    
#     # Mirror velocity commands (flip y and z)
#     flipped_obs[..., 10] = -_obs[..., 10]  # y component of velocity_commands
#     flipped_obs[..., 11] = -_obs[..., 11]  # z component of velocity_commands

#     mirror_joint_tensor(_obs,flipped_obs,12) 
#     mirror_joint_tensor(_obs,flipped_obs,31) 
#     mirror_joint_tensor(_obs,flipped_obs,50) 

#     return torch.vstack((_obs, flipped_obs))


def mirror_actions(actions):
    # if actions is None:
    #     return None

    # _actions = torch.clone(actions)
    # flip_actions = torch.zeros_like(_actions)
    # mirror_joint_tensor(_actions, flip_actions)
    # # return torch.vstack((_actions, flip_actions))
    # return torch.cat([_actions, flip_actions], dim=0)
    if actions is None:
        return None

    # If actions is a TensorDict, operate on its tensors
    if hasattr(actions, "keys") and "actions" in actions.keys():
        _actions = actions["actions"].clone()
        flip_actions = torch.zeros_like(_actions)
        mirror_joint_tensor(_actions, flip_actions)
        new_actions = actions.clone(False)
        new_actions["actions"] = flip_actions
        return new_actions

    # If actions is a plain tensor
    _actions = torch.clone(actions)
    flip_actions = torch.zeros_like(_actions)
    mirror_joint_tensor(_actions, flip_actions)
    return flip_actions

# def data_augmentation_func_g1(obs, actions, env, is_critic):
#     obs_batch = mirror_observation(obs)

#     mean_actions_batch = mirror_actions(actions)
#     return obs_batch, mean_actions_batch


def data_augmentation_func_h1(env, obs, actions):
    # if obs_type == "policy":
    #     obs_batch = mirror_observation_policy(obs)
    # elif obs_type == "critic":
    #     obs_batch = mirror_observation_critic(obs)
    # else:
    #     raise ValueError(f"Invalid observation type: {obs_type}")
    obs_batch = mirror_observation_policy(obs)
    mean_actions_batch = mirror_actions(actions)
    return obs_batch, mean_actions_batch
    # return obs_aug, actions_aug


symmetry_cfg = RslRlSymmetryCfg(
    use_data_augmentation=True,
    use_mirror_loss=True,
    data_augmentation_func=data_augmentation_func_h1,
    mirror_loss_coeff=0.25
)

@configclass
class H1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "h1_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # symmetry_cfg=symmetry_cfg,
    )


@configclass
class H1FlatPPORunnerCfg(H1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.experiment_name = "h1_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]