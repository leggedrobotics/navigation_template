# Copyright (c) 2022-2024, The IsaacLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.envs import ViewerCfg
from omni.isaac.lab.utils import configclass

@configclass
class PlayViewerCfg(ViewerCfg):
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (0.0, 7.0, 7.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    resolution: tuple[int, int] = (1920, 1080)  # FHD
    # resolution: tuple[int, int] = (1280, 720)  # HD
    origin_type: str = "asset_root"  # "world", "env", "asset_root"
    env_index: int = 1
    asset_name: str = "robot"


def add_play_configuration(self):
    """Configuration for play mode"""
    # number of environments
    self.scene.num_envs = 10

    # TODO: switch terrain

    # TODO: Add commands just for play mode

    # Curriculum (disable all)
    for key in list(self.curriculum.__dict__.keys()):
        if not key.startswith("__"):
            delattr(self.curriculum, key)

    self.events.reset_base.params["yaw_range"] = (0, 0)

    # terminate straight away
    self.terminations.goal_reached.params = {
        "time_threshold": 0.1,
        "distance_threshold": 0.5,
        "angle_threshold": 0.3,
        "speed_threshold": 0.6,
    }

    self.viewer = PlayViewerCfg()

