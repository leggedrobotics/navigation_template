

import omni.isaac.lab.terrains as terrain_gen

import nav_tasks.terrains as nav_terrains

##
# Terrain Generator
##

DENO_NAV_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(50.0, 50.0),
    border_width=1.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "maze": nav_terrains.RandomMazeTerrainCfg()
    },
)