# Author: Roman
# Refactored from: https://deepnote.com/workspace/Manipulation-ac8201a1-470a-4c77-afd0-2cc45bc229ff/project/0762b167-402a-4362-9702-7d559f0e73bb/notebook/irisbuilder-3c25c10bc29d4c9493e48eaced475d03?secondary-sidebar-autoopen=true&secondary-sidebar=agent

import time

import numpy as np
import yaml

from manipulation import running_as_notebook
from manipulation.scenarios import AddIiwa, AddWsg
from manipulation.utils import ConfigureParser
from pydrake.all import (
    CollisionCheckerParams,
    CommonSampledIrisOptions,
    HPolyhedron,
    Hyperellipsoid,
    IrisFromCliqueCoverOptions,
    IrisInConfigurationSpaceFromCliqueCover,
    IrisOptions,
    IrisZoOptions,
    RandomGenerator,
    SaveIrisRegionsYamlFile,
    SceneGraphCollisionChecker,
)
from termcolor import colored

# def compute_iris_region(plant, plant_context, q_samples):
#     E = Hyperellipsoid.MinimumVolumeCircumscribedEllipsoid(q_samples)
#     plant.SetPositions(plant_context, E.center())
#     # diagram.ForcedPublish(context)
#     options = IrisOptions()

#     # You'll see a few glancing collisions in the resulting region; increase this number
#     # to reduce them (at the cost of IRIS running for longer)
#     options.num_collision_infeasible_samples = 10
#     options.random_seed = 1235
#     options.starting_ellipse = E
#     options.iteration_limit = 1
#     region = IrisInConfigurationSpace(plant, plant_context, options)
#     # AnimateIris(diagram, context, plant, region, speed=0.1)
#     return region


def compute_iris_regions(station, use_zo=True):
    print("\n" + "=" * 60)
    print("DEBUG: compute_iris_regions starting")
    print("=" * 60)

    opt_diagram = station.internal_station.get_optimization_diagram()
    opt_plant = station.get_optimization_plant()

    iiwa_model_index = opt_plant.GetModelInstanceByName("iiwa")
    robot_instances = [iiwa_model_index]

    params = CollisionCheckerParams()
    params.model = opt_diagram
    params.robot_model_instances = robot_instances
    params.edge_step_size = 0.05
    checker = SceneGraphCollisionChecker(params=params)

    if use_zo:
        iris_options = IrisZoOptions()
        common_options = CommonSampledIrisOptions()
        common_options.termination_threshold = -1
        common_options.relative_termination_threshold = -1
        common_options.max_iterations = 1

        iris_options.sampled_iris_options = common_options
    else:
        iris_options = IrisOptions()
        iris_options.termination_threshold = -1
        iris_options.relative_termination_threshold = -1
        iris_options.iteration_limit = 1

    options = IrisFromCliqueCoverOptions()
    options.coverage_termination_threshold = 0.85
    options.iris_options = iris_options

    regions = []
    generator = RandomGenerator(1235)

    regions = IrisInConfigurationSpaceFromCliqueCover(
        checker=checker,
        options=options,
        generator=generator,
        sets=regions,
    )
    print("=" * 60)
    print(f"[DEBUG] Done. Generated {len(regions)} regions.")

    for i, region in enumerate(regions):
        b_finite = np.all(np.isfinite(region.b()))
        print(
            f"  Region {i}: dim={region.ambient_dimension()}, "
            f"A.shape={region.A().shape}, b finite={b_finite}, "
            f"Chebyshev center (deg)={np.rad2deg(region.ChebyshevCenter()).round(2)}"
        )

    named_regions = {f"region_{i}": r for i, r in enumerate(regions)}
    SaveIrisRegionsYamlFile("iris_regions.yaml", named_regions)

    return named_regions
