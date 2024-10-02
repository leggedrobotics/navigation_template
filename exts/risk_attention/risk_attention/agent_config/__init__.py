# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from risk_attention.exts.risk_attention.risk_attention.env_config import env_cfg_base
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Navigation-RiskAttention-PPO-Anymal-D-DEV",
    entry_point="omni.isaac.lab.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.RiskAttentionEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",
    },
)
gym.register(
    id="Isaac-Navigation-RiskAttention-PPO-Anymal-D-TRAIN",
    entry_point="omni.isaac.lab.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.RiskAttentionEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfg",
    },
)
gym.register(
    id="Isaac-Navigation-RiskAttention-PPO-Anymal-D-PLAY",
    entry_point="omni.isaac.lab.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.RiskAttentionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",
    },
)