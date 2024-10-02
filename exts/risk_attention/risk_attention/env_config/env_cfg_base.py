# Copyright (c) 2022-2024, The IsaacLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import torch

from omni.isaac.lab_assets import ISAACLAB_ASSETS_EXT_DIR
from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedEnvCfg, ViewerCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

import risk_attention.mdp as mdp
from risk_attention.sensors import stereolabs
from . import helper_configurations

# Reset cuda memory
torch.cuda.empty_cache()

ISAAC_GYM_JOINT_NAMES = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]

TERRAIN_MESH_PATH = ["/World/ground"]

##
# Scene definition
##
@configclass
class RiskAttentionSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # NAVIGATION TERRAIN - Must be a saved USD file
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="usd",
    #     usd_path=os.path.join(ISAACLAB_ASSETS_EXT_DIR, "Terrains", "navigation_terrain.usd"),
    #     max_init_terrain_level=None,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #         project_uvw=True,
    #     ),
    #     debug_vis=True,
    #     usd_uniform_env_spacing=10.0,  # 10m spacing between environment origins in the usd environment
    # )
        
    # DEMO TERRAIN - Just a plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # robots
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # TODO: Decide on how to downsample the camera data.
    forwards_zed_camera = ZED_X_MINI_WIDE_RAYCASTER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=TERRAIN_MESH_PATH,
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            # TODO: Find out the real rotation angle of this camera on the robot.
            # pos=(0.4761, 0.0035, 0.1055), rot=(0.9961947, 0.0, 0.087155, 0.0), convention="world"  # 10 degrees
            pos=(0.4761, 0.0035, 0.1055),
            rot=(0.9914449, 0.0, 0.1305262, 0.0),
            convention="world",  # 15 degrees
        ),
    )
    rear_zed_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=TERRAIN_MESH_PATH,
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(-0.4641, 0.0035, 0.1055), 
            rot=(-0.001, 0.132, -0.005, 0.991), 
            convention="world", # 10 degrees
        ),  
    )
    right_zed_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=TERRAIN_MESH_PATH,
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.0203, -0.1056, 0.1748),
            rot=(0.6963642, 0.1227878, 0.1227878, -0.6963642),
            convention="world",  # 20 degrees
        ),
    )
    left_zed_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=TERRAIN_MESH_PATH,
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.0217, 0.1335, 0.1748),
            rot=(0.6963642, -0.1227878, 0.1227878, 0.6963642),
            convention="world",  # 20 degrees
        ),
    )

    foot_scanner_lf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.FootScanPatternCfg(),
        debug_vis=False,
        track_mesh_transforms=False,
        mesh_prim_paths=TERRAIN_MESH_PATH,
        max_distance=100.0,
    )

    foot_scanner_rf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.FootScanPatternCfg(),
        debug_vis=False,
        track_mesh_transforms=False,
        mesh_prim_paths=TERRAIN_MESH_PATH,
        max_distance=100.0,
    )

    foot_scanner_lh = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LH_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.FootScanPatternCfg(),
        debug_vis=False,
        track_mesh_transforms=False,
        mesh_prim_paths=TERRAIN_MESH_PATH,
        max_distance=100.0,
    )

    foot_scanner_rh = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RH_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.FootScanPatternCfg(),
        debug_vis=False,
        track_mesh_transforms=False,
        mesh_prim_paths=TERRAIN_MESH_PATH,
        max_distance=100.0,
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
    )

    def __post_init__(self):
        """Post initialization."""
        self.robot.init_state.joint_pos = {
            "LF_HAA": -0.13859,
            "LH_HAA": -0.13859,
            "RF_HAA": 0.13859,
            "RH_HAA": 0.13859,
            ".*F_HFE": 0.480936,  # both front HFE
            ".*H_HFE": -0.480936,  # both hind HFE
            ".*F_KFE": -0.761428,
            ".*H_KFE": 0.761428,
        }
        # TODO: Is this necessary?
        self.robot.spawn.scale = [1.15, 1.15, 1.15]


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_command = mdp.PerceptiveNavigationSE2ActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False
        ),
        low_level_decimation=4,
        low_level_policy_file=os.path.join(
            ISAACLAB_ASSETS_EXT_DIR, "Robots/RSL-ETHZ/ANYmal-D", "perceptive_locomotion_jit.pt"
        ),
        reorder_joint_list=ISAAC_GYM_JOINT_NAMES,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class LocomotionPolicyCfg(ObsGroup):
        """
        Observations for locomotion policy group. These are fixed when training a navigation 
        policy using a pre-trained locomotion policy.
        """
        # Proprioception
        wild_anymal = ObsTerm(
            func=mdp.wild_anymal,
            params={
                "action_term": "velocity_command",
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES),
            },
        )
        # Exterocpetion
        foot_scan_lf = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foot_scanner_lf"), "offset": 0.05},
            scale=10.0,
        )
        foot_scan_rf = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foot_scanner_rf"), "offset": 0.05},
            scale=10.0,
        )
        foot_scan_lh = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foot_scanner_lh"), "offset": 0.05},
            scale=10.0,
        )
        foot_scan_rh = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foot_scanner_rh"), "offset": 0.05},
            scale=10.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class NavigationPolicyCfg(ObsGroup):
        """Observations for navigation policy group."""

        # TODO: Add noise to the observations once training works, eg:
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_command"})

        forwards_depth_image = ObsTerm(
            func=mdp.flattened_depth_img,
            params={"sensor_cfg": SceneEntityCfg("forwards_zed_camera")},
        )
        rear_depth_image = ObsTerm(
            func=mdp.flattened_depth_img,
            params={"sensor_cfg": SceneEntityCfg("rear_zed_camera")},
        )
        right_depth_image = ObsTerm(
            func=mdp.flattened_depth_img,
            params={"sensor_cfg": SceneEntityCfg("right_zed_camera")},
        )
        left_depth_image = ObsTerm(
            func=mdp.flattened_depth_img,
            params={"sensor_cfg": SceneEntityCfg("left_zed_camera")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    low_level_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()
    policy: NavigationPolicyCfg = NavigationPolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    reset_base = EventTerm(
        func=mdp.TerrainAnalysisRootReset(
            cfg=mdp.TerrainAnalysisCfg(
                semantic_cost_mapping=None,
                raycaster_sensor="forwards_zed_camera",
                viz_graph=True,
                sample_points=10000,
                height_diff_threshold=0.2,
            )
        ),
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "yaw_range": (-3.14, 3.14),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.
    NOTE: all reward get multiplied with weight*dt --> consider this!
    and are normalized over max episode length (in wandb logging)
    NOTE: Wandb --> Eposiode Rewards are in seconds!
    NOTE: Wandb Train Mean Reward --> based on episode length --> Rewards * Episode Length
    """

    # TODO: Go through these, add to isaac-nav-suite, and document / clean up.

    # -- tasks
    # sparse only on termination of task
    goal_reached_rew = RewTerm(
        func=mdp.is_terminated_term,  # returns 1 if the goal is reached and env has NOT timed out
        params={"term_keys": "goal_reached"},
        weight=1000.0,  # make it big
    )

    stepped_goal_progress = mdp.SteppedProgressCfg(
        step=0.05,
        weight=1.0,
    )
    near_goal_stability = RewTerm(
        func=mdp.near_goal_stability,
        weight=2.0,  # Dense Reward of [0.0, 0.1] --> Max Episode Reward: 1.0
    )
    near_goal_angle = RewTerm(
        func=mdp.near_goal_angle,
        weight=1.0,  # Dense Reward of [0.0, 0.025]  --> Max Episode Reward: 0.25
    )

    # -- penalties
    lateral_movement = RewTerm(
        func=mdp.lateral_movement,
        weight=-0.1,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )
    backward_movement = RewTerm(
        func=mdp.backwards_movement,
        weight=-0.1,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )
    episode_termination = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["base_contact", "leg_contact"]},
        weight=-200.0,  # Sparse Reward of {-20.0, 0.0} --> Max Episode Penalty: -20.0
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )
    no_robot_movement = RewTerm(
        func=mdp.no_robot_movement,
        weight=-0.1,  # Dense Reward of [-0.1, 0.0] --> Max Episode Penalty: -1.0
        params={"goal_distance_thresh": 0.5},
    )


@configclass
class TerminationsCfg:
    """
    Termination terms for the MDP. 
    NOTE: time_out flag: if set to True, there won't be any termination penalty added for 
          the termination, but in the RSL_RL library time_out flag has implications for how 
          the reward is handled before the optimization step. If time_out is True, the rewards
          are bootstrapped !!!
    NOTE: When robot arrives at goal, it should stay there until timeout is reached, thus no 
          termimation config for training.
    NOTE: Wandb Episode Termination --> independent of num robots, episode length, etc.
    """

    # TODO: Go through these and document / clean up.

    time_out = DoneTerm(
        func=mdp.proportional_time_out,
        params={
            "max_speed": 1.0,
            "safety_factor": 4.0,
        },
        time_out=True, # No termination penalty for time_out = True
    )

    goal_reached = DoneTerm(
        func=mdp.stayed_at_goal,
        params={
            "time_threshold": 2.0,
            "distance_threshold": 0.5,
            "angle_threshold": 0.3,
            "speed_threshold": 0.6,
        },
        time_out=False,
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
            "threshold": 0.0,
        },
        time_out=False,
    )

    leg_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH", ".*HIP", ".*SHANK"]),
            "threshold": 0.0,
        },
        time_out=False,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.
    NOTE: steps = learning_iterations * num_steps_per_env)
    """
        
    terrain_levels = CurrTerm(
        func=mdp.modify_terrain_level,
        params={
            "initial_config": {
                "promote_ratio": 0.01,
                "demote_ratio": 0.01,
            },
            "final_config": {
                "promote_ratio": 0.2,
                "demote_ratio": 0.2,
            },
            "start_step": 500 * 48,
            "end_step": 1500 * 48,
        },
    )

    initial_heading_pertubation = CurrTerm(
        func=mdp.modify_heading_randomization_linearly,
        params={
            "term_name": "reset_base",
            "initial_perturbation": 0.0,
            "final_perturbation": 3.0,
            "start_step": 0,
            "end_step": 500 * 48,
        },
    )

    goal_conditions_ramp = CurrTerm(
        func=mdp.modify_goal_conditions,
        params={
            "term_name": "goal_reached",
            "time_range": (2.0, 2.0),
            "distance_range": (1, 0.5),
            "angle_range": (0.6, 0.3),
            "speed_range": (1.3, 0.6),
            "step_range": (0, 500 * 48),
        },
    )

    # Increase goal distance & resample trajectories
    goal_distances = CurrTerm(
        func=mdp.modify_goal_distance_in_steps,
        params={
            "update_rate_steps": 100 * 48,
            "initial_config": {
                "num_paths": 100,
                "min_path_length": 1.0,
                "max_path_length": 5.0,
            },
            "final_config": {
                "num_paths": 100,
                "min_path_length": 5.0,
                "max_path_length": 15.0,
            },
            "start_step": 50 * 48,
            "end_step": 1500 * 48,
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # TODO: Configure Goal Command With Terrain Analysis
    command: mdp.NullCommandCfg = mdp.NullCommandCfg()
    
    # goal_command = mdp.GoalCommandCfg(
    #     asset_name="robot",
    #     z_offset_spawn=0.2,
    #     etc...
    # )


@configclass
class MyViewerCfg(ViewerCfg):
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (0.0, 70.0, 70.0)
    lookat: tuple[float, float, float] = (0.0, 10.0, 0.0)
    # resolution: tuple[int, int] = (1920, 1080)   # FHD
    resolution: tuple[int, int] = (1280, 720)  # HD
    origin_type: str = "world"  # "world", "env", "asset_root"
    env_index: int = 1
    asset_name: str = "robot"


##
# Environment configuration
##


@configclass
class RiskAttentionEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the navigation environment."""

    # Scene settings
    scene: RiskAttentionSceneCfg = RiskAttentionSceneCfg(num_envs=100, env_spacing=8)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    viewer: MyViewerCfg = MyViewerCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.fz_low_level_planner = 10  # Hz
        self.episode_length_s = 20
        self.use_path_follower_controller = False  # default is learned policy

        # DO NOT CHANGE
        self.decimation = int(200 / self.fz_low_level_planner)  # low/high level planning runs at 25Hz <- This is
        # setting how many times the high level actions (navigation policy) are applied to the sim before being
        # recalculated. This means the MDP stuff is recalculated every 0.005 * 20 = 0.1 seconds -> 10Hz.

        self.low_level_decimation = 4  # low level controller runs at 50Hz <- similar to above, the low_level actions
        # are calculated every sim.dt * low_level_decimation, so 0.005 * 4 = 0.02 seconds, or 50hz.

        # simulation settings
        self.sim.dt = 0.005  # In seconds
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.forwards_zed_camera is not None:
            self.scene.forwards_zed_camera.update_period = self.decimation * self.sim.dt
        if self.scene.rear_zed_camera is not None:
            self.scene.rear_zed_camera.update_period = self.decimation * self.sim.dt
        if self.scene.right_zed_camera is not None:
            self.scene.right_zed_camera.update_period = self.decimation * self.sim.dt
        if self.scene.left_zed_camera is not None:
            self.scene.left_zed_camera.update_period = self.decimation * self.sim.dt

        # store the image dimensions so the policy pre processing CNN can be initialised correctly
        # list of tuples (height, width), the policy will split observations based on this
        self.depth_image_size = DEPTH_IMAGE_SIZE


######################################################################
# Anymal D - TRAIN & PLAY & DEV Configuration
######################################################################
@configclass
class RiskAttentionEnvCfg_TRAIN(RiskAttentionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Change number of environments
        # self.scene.num_envs = 50


@configclass
class RiskAttentionEnvCfg_PLAY(RiskAttentionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # default play configuration
        helper_configurations.add_play_configuration(self)


@configclass
class RiskAttentionEnvCfg_DEV(RiskAttentionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 2