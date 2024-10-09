from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg


def expanded_generated_commands(env: ManagerBasedRLEnvCfg, command_name: str, size: int) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name, with a dimension added such
    that the total size is equal to the given size.
    Useful for expanding the generated command to similar shape to the other observations.
    """
    command = env.command_manager.get_command(command_name)
    dim_expansion = size // command[0].numel()  # Expand each eviroment's command to the given size
    return command.unsqueeze(2).expand(-1, -1, dim_expansion).reshape(command.shape[0], -1)