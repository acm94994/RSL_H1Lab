import torch

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
        (0 + offset, 1 + offset),   # hip_yaw
        (3 + offset, 4 + offset),   # hip_roll
        (5 + offset, 6 + offset),   # shoulder_pitch
        (7 + offset, 8 + offset),   # hip_pitch
        (9 + offset, 10 + offset),  # shoulder_roll
        (11 + offset, 12 + offset), # knee
        (13 + offset, 14 + offset), # shoulder_yaw
        (15 + offset, 16 + offset), # ankle
        (17 + offset, 18 + offset), # elbow
    ]
    
    # Define indices that need to be inverted (yaw/roll joints)
    invert_indices = [
        2 + offset,   # torso
        3 + offset,   # left_hip_roll
        4 + offset,   # right_hip_roll
        0 + offset,   # left_hip_yaw
        1 + offset,   # right_hip_yaw
        9 + offset,  # left_shoulder_roll
        10 + offset,  # right_shoulder_roll
        13 + offset,  # left_shoulder_yaw
        14 + offset,  # right_shoulder_yaw
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
    
    _obs = torch.clone(obs)
    flipped_obs = torch.clone(obs)
    # Mirror projected gravity (flip y)
    flipped_obs[..., 1] = -_obs[..., 1]  # y component of projected_gravity

    # Mirror velocity commands (flip y and z)
    flipped_obs[..., 4] = -_obs[..., 4]  # y component of velocity_commands
    flipped_obs[..., 5] = -_obs[..., 5]  # z component of velocity_commands

    mirror_joint_tensor(_obs, flipped_obs, 6)
    mirror_joint_tensor(_obs, flipped_obs, 25)
    mirror_joint_tensor(_obs, flipped_obs, 44)

    # Only return the mirrored batch (do not double batch size)
    return flipped_obs

def mirror_observation(obs):
    if obs is None:
        return obs
    
    _obs = torch.clone(obs)
    flipped_obs = torch.clone(obs)
    # Mirror base linear velocity (flip y)
    flipped_obs[..., 1] = -_obs[..., 1]  # y component of base_lin_vel
    
    # Mirror base angular velocity (flip z)
    flipped_obs[..., 5] = -_obs[..., 5]  # z component of base_ang_vel
    
    # Mirror projected gravity (flip y)
    flipped_obs[..., 7] = -_obs[..., 7]  # y component of projected_gravity
    
    # Mirror velocity commands (flip y and z)
    flipped_obs[..., 10] = -_obs[..., 10]  # y component of velocity_commands
    flipped_obs[..., 11] = -_obs[..., 11]  # z component of velocity_commands

    mirror_joint_tensor(_obs, flipped_obs, 12)
    mirror_joint_tensor(_obs, flipped_obs, 31)
    mirror_joint_tensor(_obs, flipped_obs, 50)
    
    return torch.cat((_obs, flipped_obs), dim=0)


def mirror_actions(actions):
    if actions is None:
        return None

    _actions = torch.clone(actions)
    flip_actions = torch.zeros_like(_actions)
    mirror_joint_tensor(_actions, flip_actions)
    return torch.cat((_actions, flip_actions), dim=0)
    




def data_augmentation_func_h1(env, obs, actions):
    obs_batch = mirror_observation(obs)
    mean_actions_batch = mirror_actions(actions)
    return obs_batch, mean_actions_batch
    