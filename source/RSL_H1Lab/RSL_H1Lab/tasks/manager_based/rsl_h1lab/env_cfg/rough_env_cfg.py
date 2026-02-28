# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import RSL_H1Lab.tasks.manager_based.rsl_h1lab.mdp as mdp
from RSL_H1Lab.tasks.manager_based.rsl_h1lab.env_cfg.velocity_env_cfg import (
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets import H1_MINIMAL_CFG  # isort: skip


##
# Domain Randomization Configuration
# Based on: https://lilianweng.github.io/posts/2019-05-05-domain-randomization/
#
# Key randomization parameters for sim2real transfer:
# 1. Mass and dimensions of robot bodies
# 2. Damping, friction of the joints
# 3. Gains for the PD controller (stiffness/damping)
# 4. Center of mass variations
# 5. Ground friction properties
# 6. External disturbances (forces/torques)
# 7. Observation noise (sensor noise)
##


@configclass
class H1DomainRandomizationCfg(EventCfg):
    """Domain randomization configuration for sim2real transfer.
    
    This implements uniform domain randomization as described in:
    - Peng et al. 2018: "Sim-to-real transfer of robotic control with dynamics randomization"
    - OpenAI 2018: "Learning Dexterous In-Hand Manipulation"
    """

    # === STARTUP RANDOMIZATION (applied once at environment creation) ===
    
    # Randomize ground/contact friction properties
    # This helps the policy generalize to different floor surfaces
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.5),   # Vary friction ±40%
            "dynamic_friction_range": (0.4, 1.2),  # Vary friction ±50%
            "restitution_range": (0.0, 0.125),       # Slight bounce variation
            "num_buckets": 64,
        },
    )

    # Randomize base/torso mass (simulates payload variations)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "mass_distribution_params": (-3.0, 5.0),  # Add -3kg to +5kg
            "operation": "add",
        },
    )

    # Randomize limb masses (manufacturing tolerances, wear)
    add_limb_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*hip_.*", ".*knee_.*", ".*ankle_.*"]),
            "mass_distribution_params": (0.8, 1.2),  # Scale mass ±20%
            "operation": "scale",
        },
    )

    # Randomize center of mass position (load distribution variations)
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "com_range": {"x": (-0.08, 0.08), "y": (-0.05, 0.05), "z": (-0.03, 0.03)},
        },
    )

    # Randomize actuator PD gains (motor response variations)
    # Critical for sim2real: real motors have different response characteristics
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.25),  # Scale Kp ±25%
            "damping_distribution_params": (0.75, 1.25),    # Scale Kd ±25%
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Randomize joint friction (mechanical wear, lubrication)
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.4, 2.5),  # Scale friction 0.4x to 2.5x
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # === RESET RANDOMIZATION (applied at each episode reset) ===

    # External force/torque disturbances (wind, collisions, perturbations)
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "force_range": (-10.0, 10.0),   # Random forces up to 10N
            "torque_range": (-5.0, 5.0),    # Random torques up to 5Nm
        },
    )

    # Reset base pose with randomization
    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-0.3, 0.3),
    #             "y": (-0.3, 0.3),
    #             "z": (-0.1, 0.1),
    #             "roll": (-0.2, 0.2),
    #             "pitch": (-0.2, 0.2),
    #             "yaw": (-0.3, 0.3),
    #         },
    #     },
    # )

    # Reset joint positions with randomization
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.8, 1.2),  # ±20% of default position
    #         "velocity_range": (-0.12, 0.12),  # Small initial velocities
    #     },
    # )

    # === INTERVAL RANDOMIZATION (applied periodically during episode) ===

    # Random pushes to test balance recovery
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(3.0, 15.0),  # Push every 3-15 seconds
    #     params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
    # )


@configclass
class H1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle")}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")}
    )


@configclass
class H1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: H1Rewards = H1Rewards()
    events: H1DomainRandomizationCfg = H1DomainRandomizationCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # === ABLATION: Disable domain randomization ===
        # Comment/uncomment these lines to enable/disable specific randomizations
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.add_limb_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_friction = None
        self.events.base_external_force_torque = None

        # Domain Randomization is now handled by H1DomainRandomizationCfg
        # The following overrides can be used to tune specific randomization parameters:
        
        # Adjust joint reset range (1.0 = no randomization)
        # self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)  # ±25% of default position
        # self.events.reset_robot_joints.params["velocity_range"] = (-0.2, 0.2)  # Small initial velocities

        # self.events.reset_base.params["pose_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-3.14, 3.14)}
        # self.events.reset_base.params["velocity_range"] = {
        #     "x": (-0.3, 0.3),
        #     "y": (-0.3, 0.3),
        #     "z": (-0.1, 0.1),
        #     "roll": (-0.25, 0.25),
        #     "pitch": (-0.25, 0.25),
        #     "yaw": (-0.35, 0.35),
        # }
        
        # Adjust external force disturbances
        # self.events.base_external_force_torque.params["force_range"] = (-5.0, 5.0)
        
        # Adjust push robot velocity (for balance recovery training)
        # self.events.push_robot.params["velocity_range"] = {"x": (-2.5, 2.5), "y": (-2.5, 2.5)}
        # self.events.push_robot.interval_range_s = (3.0, 20.0)

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*torso_link"


@configclass
class H1RoughEnvCfg_PLAY(H1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
