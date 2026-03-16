# Author: Roman
# Refactored from: https://deepnote.com/workspace/Manipulation-ac8201a1-470a-4c77-afd0-2cc45bc229ff/project/0762b167-402a-4362-9702-7d559f0e73bb/notebook/irisbuilder-3c25c10bc29d4c9493e48eaced475d03?secondary-sidebar-autoopen=true&secondary-sidebar=agent

import time

import numpy as np

from manipulation import running_as_notebook
from manipulation.scenarios import AddIiwa, AddWsg
from manipulation.utils import ConfigureParser
from pydrake.all import (
    AddDefaultVisualization,
    AddMultibodyPlantSceneGraph,
    Context,
    Diagram,
    DiagramBuilder,
    HPolyhedron,
    Hyperellipsoid,
    InverseKinematics,
    IrisInConfigurationSpace,
    IrisOptions,
    MathematicalProgram,
    MultibodyPlant,
    Parser,
    Rgba,
    RigidTransform,
    Solve,
    Sphere,
    StartMeshcat,
)

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


def compute_iris_regions():
    # Run the algorithm to populate the regions list
    IrisInConfigurationSpaceFromCliqueCover(
        checker=checker, options=options, generator=generator, sets=regions
    )


# def ScaleHPolyhedron(hpoly, scale_factor):
#     # Shift to the center.
#     xc = hpoly.ChebyshevCenter()
#     A = hpoly.A()
#     b = hpoly.b() - A @ xc
#     # Scale
#     b = scale_factor * b
#     # Shift back
#     b = b + A @ xc
#     return HPolyhedron(A, b)


# def _CheckNonEmpty(region):
#     prog = MathematicalProgram()
#     x = prog.NewContinuousVariables(region.ambient_dimension())
#     region.AddPointInSetConstraints(prog, x)
#     result = Solve(prog)
#     assert result.is_success()


# def _CalcRegion(name, seed):
#     builder = DiagramBuilder()
#     plant = AddMultibodyPlantSceneGraph(builder, 0.0)[0]
#     LoadRobot(plant)
#     plant.Finalize()
#     diagram = builder.Build()
#     diagram_context = diagram.CreateDefaultContext()
#     plant_context = plant.GetMyContextFromRoot(diagram_context)
#     plant.SetPositions(plant_context, seed)
#     if use_existing_regions_as_obstacles:
#         iris_options.configuration_obstacles = [
#             ScaleHPolyhedron(r, regions_as_obstacles_scale_factor)
#             for k, r in iris_regions.items()
#             if k != name
#         ]
#         for h in iris_options.configuration_obstacles:
#             _CheckNonEmpty(h)
#     else:
#         iris_options.configuration_obstacles = None
#     display(f"Computing region for seed: {name}")
#     start_time = time.time()
#     hpoly = IrisInConfigurationSpace(plant, plant_context, iris_options)
#     display(
#         f"Finished seed {name}; Computation time: {(time.time() - start_time):.2f} seconds"
#     )

#     _CheckNonEmpty(hpoly)
#     reduced = hpoly.ReduceInequalities()
#     _CheckNonEmpty(reduced)

#     return reduced

# def GenerateRegion(name, seed):
#     global iris_regions
#     iris_regions[name] = _CalcRegion(name, seed)
#     SaveIrisRegionsYamlFile(f"{iris_filename}.autosave", iris_regions)


# def GenerateRegions(seed_dict, verbose=True):
#     if use_existing_regions_as_obstacles:
#         # Then run serially
#         for k, v in seed_dict.items():
#             GenerateRegion(k, v)
#         return

#     loop_time = time.time()
#     with mp.Pool(processes=num_parallel) as pool:
#         new_regions = pool.starmap(_CalcRegion, [[k, v] for k, v in seed_dict.items()])

#     if verbose:
#         print("Loop time:", time.time() - loop_time)

#     global iris_regions
#     iris_regions.update(dict(list(zip(seed_dict.keys(), new_regions))))
