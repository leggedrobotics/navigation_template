# Copyright (c) 2022-2024, The IsaacLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import torch

from nav_collectors.collectors import TrajectorySamplingCfg
from nav_collectors.terrain_analysis import TerrainAnalysisCfg
from nav_tasks.sensors import ZED_X_MINI_WIDE_RAYCASTER_CFG, FootScanPatternCfg, adjust_ray_caster_camera_image_size
from nav_tasks.terrains import RandomMazeTerrainCfg, MeshPillarTerrainCfg

import graph_nav.mdp as mdp

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from omni.isaac.lab.utils import configclass
from nav_collectors.terrain_analysis import TerrainAnalysisCfg
from nav_collectors.collectors import TrajectorySamplingCfg
from nav_tasks.sensors import adjust_ray_caster_camera_image_size, ZED_X_MINI_WIDE_RAYCASTER_CFG, FootScanPatternCfg

import navigation_template.mdp as mdp

from .helper_configurations import add_play_configuration

# Reset cuda memory
torch.cuda.empty_cache()

from nav_tasks.mdp.actions.navigation_se2_actions_cfg import ISAAC_GYM_JOINT_NAMES

TERRAIN_MESH_PATH: list[str | RayCasterCfg.RaycastTargetCfg] = ["/World/ground"]

IMAGE_SIZE_DOWNSAMPLE_FACTOR = 10


# Terrain Options
FLAT_PLANE = TerrainImporterCfg(
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

GENERATED_SUBTERRAINS = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    terrain_generator=TerrainGeneratorCfg(
        curriculum=True,
        size=(20.0, 20.0),
        num_rows=4,
        num_cols=3,
        sub_terrains={
            "random_maze__rng": RandomMazeTerrainCfg(
                resolution=1.0,
                maze_height=1.0,
                randomization={
                    "range": {
                        "length": [0.5, 1],
                        "width": [0.5, 1],
                        "height": [0.01, 1],
                    },
                    "max_increase": 1.0,
                    "max_decrease": 1.0,
                },
            ),
            "random_maze": RandomMazeTerrainCfg(
                resolution=1.0,
                maze_height=1.0,
            ),
            "mesh_pillar": MeshPillarTerrainCfg(
                box_objects=MeshPillarTerrainCfg.BoxCfg(
                    width=(0.5, 1.0),
                    length=(0.5, 1.0),
                    max_yx_angle=(0.0, 0.0),
                    num_objects=(20, 20),
                    height=(0.5, 1.0),
                ),
                cylinder_cfg=MeshPillarTerrainCfg.CylinderCfg(
                    radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(0.5, 5), num_objects=(20, 20)
                ),
            ),
        },
    ),
)


##
# Scene definition
##
@configclass
class NavigationTemplateSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene for training a perceptive navigation policy on an AnymalD Robot."""

    # DEMO TERRAIN - Just a plane
    terrain = GENERATED_SUBTERRAINS

    # ROBOTS
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # SENSORS

    # Stereolabs Cameras for Navigation Policy
    forwards_zed_camera = ZED_X_MINI_WIDE_RAYCASTER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=TERRAIN_MESH_PATH,
        update_period=0,
        debug_vis=False,
        offset=RayCasterCameraCfg.OffsetCfg(
            # The camera can be mounted at either 10 or 15 degrees on the robot.
            # pos=(0.4761, 0.0035, 0.1055), rot=(0.9961947, 0.0, 0.087155, 0.0), convention="world"  # 10 degrees
            pos=(0.4761, 0.0035, 0.1055),
            rot=(0.9914449, 0.0, 0.1305262, 0.0),
            convention="world",  # 15 degrees
        ),
    )
    rear_zed_camera = ZED_X_MINI_WIDE_RAYCASTER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=TERRAIN_MESH_PATH,
        update_period=0,
        debug_vis=False,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(-0.4641, 0.0035, 0.1055),
            rot=(-0.001, 0.132, -0.005, 0.991),
            convention="world",  # 10 degrees
        ),
    )
    right_zed_camera = ZED_X_MINI_WIDE_RAYCASTER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=TERRAIN_MESH_PATH,
        update_period=0,
        debug_vis=False,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.0203, -0.1056, 0.1748),
            rot=(0.6963642, 0.1227878, 0.1227878, -0.6963642),
            convention="world",  # 20 degrees
        ),
    )
    left_zed_camera = ZED_X_MINI_WIDE_RAYCASTER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=TERRAIN_MESH_PATH,
        update_period=0,
        debug_vis=False,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.0217, 0.1335, 0.1748),
            rot=(0.6963642, -0.1227878, 0.1227878, 0.6963642),
            convention="world",  # 20 degrees
        ),
    )

    # Foot Scanners for Locomotion Policy
    foot_scanner_lf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=FootScanPatternCfg(),
        debug_vis=False,
        track_mesh_transforms=False,
        mesh_prim_paths=TERRAIN_MESH_PATH,
        max_distance=100.0,
    )

    foot_scanner_rf = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=FootScanPatternCfg(),
        debug_vis=False,
        track_mesh_transforms=False,
        mesh_prim_paths=TERRAIN_MESH_PATH,
        max_distance=100.0,
    )

    foot_scanner_lh = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LH_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=FootScanPatternCfg(),
        debug_vis=False,
        track_mesh_transforms=False,
        mesh_prim_paths=TERRAIN_MESH_PATH,
        max_distance=100.0,
    )

    foot_scanner_rh = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RH_FOOT",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=FootScanPatternCfg(),
        debug_vis=False,
        track_mesh_transforms=False,
        mesh_prim_paths=TERRAIN_MESH_PATH,
        max_distance=100.0,
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # LIGHTS
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
        # Downsample the camera data to a usable size for the project.
        self.forwards_zed_camera = adjust_ray_caster_camera_image_size(
            self.forwards_zed_camera, IMAGE_SIZE_DOWNSAMPLE_FACTOR, IMAGE_SIZE_DOWNSAMPLE_FACTOR
        )
        self.rear_zed_camera = adjust_ray_caster_camera_image_size(
            self.rear_zed_camera, IMAGE_SIZE_DOWNSAMPLE_FACTOR, IMAGE_SIZE_DOWNSAMPLE_FACTOR
        )
        self.right_zed_camera = adjust_ray_caster_camera_image_size(
            self.right_zed_camera, IMAGE_SIZE_DOWNSAMPLE_FACTOR, IMAGE_SIZE_DOWNSAMPLE_FACTOR
        )
        self.left_zed_camera = adjust_ray_caster_camera_image_size(
            self.left_zed_camera, IMAGE_SIZE_DOWNSAMPLE_FACTOR, IMAGE_SIZE_DOWNSAMPLE_FACTOR
        )


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
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=ISAAC_GYM_JOINT_NAMES, preserve_order=True),
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

    # Observation Groups
    low_level_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()
    policy: NavigationPolicyCfg = NavigationPolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    reset_base = EventTerm(
        func=mdp.reset_robot_position,
        mode="reset",
        params={
            "yaw_range": (-3.0, 3.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
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

    # -- rewards
    # Sparse: only when the "stayed_at_goal" condition is met, per the goal_reached term in TerminationsCfg
    goal_reached_rew = RewTerm(
        func=mdp.is_terminated_term,  # returns 1 if the goal is reached and env has NOT timed out # type: ignore
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
        func=mdp.is_terminated_term,  # type: ignore
        params={"term_keys": ["base_contact", "leg_contact"]},
        weight=-200.0,  # Sparse Reward of {-20.0, 0.0} --> Max Episode Penalty: -20.0
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )


@configclass
class TerminationsCfg:
    """
    Termination terms for the MDP. 

    NOTE: time_out flag: if set to True, there won't be any termination penalty added for 
          the termination, but in the RSL_RL library time_out flag has implications for how 
          the reward is handled before the optimization step. If time_out is True, the rewards
          are bootstrapped!
    NOTE: Wandb Episode Termination --> independent of num robots, episode length, etc.
    """

    time_out = DoneTerm(
        func=mdp.proportional_time_out,
        params={
            "max_speed": 1.0,
            "safety_factor": 4.0,
        },
        time_out=True,  # No termination penalty for time_out = True
    )

    goal_reached = DoneTerm(
        func=mdp.StayedAtGoal,  # type: ignore
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

    initial_heading_pertubation = CurrTerm(
        func=mdp.modify_heading_randomization_linearly,
        params={
            "event_term_name": "reset_base",
            "initial_perturbation": 0.0,
            "final_perturbation": 3.0,
            "start_step": 0,
            "end_step": 500 * 48,
        },
    )

    goal_conditions_ramp = CurrTerm(
        func=mdp.modify_goal_conditions,
        params={
            "termination_term_name": "goal_reached",
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
                "min_path_length": 1.0,
                "max_path_length": 5.0,
            },
            "final_config": {
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

    goal_command = mdp.GoalCommandCfg(
        asset_name="robot",
        z_offset_spawn=0.1,
        trajectory_config = {
            "num_paths": [1000],
            "max_path_length": [10.0],
            "min_path_length": [2.0],
        },
        traj_sampling=TrajectorySamplingCfg(
            terrain_analysis=TerrainAnalysisCfg(
                raycaster_sensor="forwards_zed_camera", 
                max_terrain_size=100.0, 
                semantic_cost_mapping=None,
                viz_graph=False,
                viz_height_map=False,
            )
        ),
        resampling_time_range=(1.0e9, 1.0e9),  # No resampling
        debug_vis=True,
    )


@configclass
class DefaultViewerCfg(ViewerCfg):
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (0.0, 50.0, 50.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    resolution: tuple[int, int] = (1280, 720)  # (1280, 720) HD, (1920, 1080) FHD
    origin_type: str = "world"  # "world", "env", "asset_root"
    env_index: int = 1
    asset_name: str = "robot"


##
# Environment configuration
##


@configclass
class NavigationTemplateEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # Scene settings
    scene: NavigationTemplateSceneCfg = NavigationTemplateSceneCfg(num_envs=100, env_spacing=8)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    viewer: DefaultViewerCfg = DefaultViewerCfg()

    def __post_init__(self):
        """Post initialization."""

        ###### DO NOT CHANGE ######

        # Simulation settings
        self.sim.dt = 0.005  # In seconds
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # General settings
        self.episode_length_s = 20

        # This sets how many times the high-level actions (navigation policy)
        # are applied to the sim before being recalculated.
        # self.sim.dt * self.decimation = 0.005 * 20 = 0.1 seconds -> 10Hz.
        self.fz_low_level_planner = 10  # Hz
        self.decimation = int(200 / self.fz_low_level_planner)

        # Similar to above, the low-level actions (locomotion controller) are calculated every:
        # self.sim.dt * self.low_level_decimation, so 0.005 * 4 = 0.02 seconds, or 50Hz.
        self.low_level_decimation = 4

        ###### /DO NOT CHANGE ######

        # update sensor update periods
        # We tick contact sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # We tick the cameras based on the navigation policy update period.
        if self.scene.forwards_zed_camera is not None:
            self.scene.forwards_zed_camera.update_period = self.decimation * self.sim.dt
        if self.scene.rear_zed_camera is not None:
            self.scene.rear_zed_camera.update_period = self.decimation * self.sim.dt
        if self.scene.right_zed_camera is not None:
            self.scene.right_zed_camera.update_period = self.decimation * self.sim.dt
        if self.scene.left_zed_camera is not None:
            self.scene.left_zed_camera.update_period = self.decimation * self.sim.dt


######################################################################
# Anymal D - TRAIN & PLAY & DEV Configuration Modifications
######################################################################
@configclass
class NavigationTemplateEnvCfg_TRAIN(NavigationTemplateEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Change number of environments
        # self.scene.num_envs = 50


@configclass
class NavigationTemplateEnvCfg_PLAY(NavigationTemplateEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # default play configuration
        add_play_configuration(self)


@configclass
class NavigationTemplateEnvCfg_DEV(NavigationTemplateEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2
