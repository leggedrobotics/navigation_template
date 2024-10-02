from __future__ import annotations

import torch
import torch.types
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import Camera, RayCasterCamera

if TYPE_CHECKING:
    from omni.isaac.lab.envs import BaseEnv

def depth_camera(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera depth image from the given sensor w.r.t. the asset's root frame.
    Also removes nan/inf values and sets them to the maximum distance of the sensor

    Args:
        env: The environment object.
        sensor_cfg: The name of the sensor.

    Returns:
        The depth image."""
    # extract the used quantities (to enable type-hinting)
    sensor: Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    try:
        if isinstance(sensor, Camera):
            depth_img = sensor.data.output["distance_to_image_plane"].clone()
        elif isinstance(sensor, RayCasterCamera):
            depth_img = sensor.camera_data.output["distance_to_image_plane"].clone()
        else:
            raise ValueError("The sensor type is not supported.")
    except AttributeError:
        depth_img = torch.zeros((env.num_envs, sensor.image_shape[0], sensor.image_shape[1])).to(env.device)
        print("WARNING: no image received, returning zero tensor")
    depth_img[torch.isnan(depth_img)] = sensor.cfg.max_distance
    depth_img[torch.isinf(depth_img)] = sensor.cfg.max_distance
    return depth_img


def flattened_depth_img(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    depth_img = depth_camera(env, sensor_cfg)
    depth_img = depth_img.flatten(start_dim=1)
    return depth_img