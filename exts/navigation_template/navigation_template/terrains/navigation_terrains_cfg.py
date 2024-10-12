

from omni.isaac.lab.terrains import TerrainGeneratorCfg

from nav_tasks.terrains import RandomMazeTerrainCfg, MeshPillarTerrainCfg

DEMO_NAV_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(50.0, 50.0),
    border_width=10.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "maze": RandomMazeTerrainCfg()
    },
)


DEMO_NAV_CURRICULUM_TERRAIN_CFG = TerrainGeneratorCfg(
        curriculum=True,
        size=(40.0, 40.0),
        border_width=10.0,
        num_rows=4,
        num_cols=3,
        sub_terrains={
            "random_maze__rng": RandomMazeTerrainCfg(
                resolution=1.0,
                maze_height=1.0,
                length_range=(0.5, 1),
                width_range=(0.5, 1),
                height_range=(0.01, 1),
                max_increase=1.0,
                max_decrease=1.0,
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
)