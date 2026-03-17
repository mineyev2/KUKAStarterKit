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


def compute_iris_regions(station):
    print("\n" + "=" * 60)
    print("DEBUG: compute_iris_regions starting")
    print("=" * 60)

    opt_diagram = station.internal_station.get_optimization_diagram()
    opt_plant = station.get_optimization_plant()
    internal_plant = station.get_internal_plant()

    # ----------------------------------------------------------------
    # 1. Print all model instances in both plants
    # ----------------------------------------------------------------
    from pydrake.all import ModelInstanceIndex

    # ----------------------------------------------------------------
    # 2. Set up collision checker
    # ----------------------------------------------------------------
    iiwa_model_index = opt_plant.GetModelInstanceByName("iiwa")
    print(f"\n[DEBUG] iiwa model instance index in opt_plant: {iiwa_model_index}")
    robot_instances = [iiwa_model_index]

    params = CollisionCheckerParams()
    params.model = opt_diagram
    params.robot_model_instances = robot_instances
    params.edge_step_size = 0.05
    checker = SceneGraphCollisionChecker(params=params)
    print(
        f"[DEBUG] Checker created. checker.plant().num_positions() = {checker.plant().num_positions()}"
    )

    opt_plant_context = station.internal_station.get_optimization_plant_context()
    q_opt = opt_plant.GetPositions(opt_plant_context)
    q_internal = internal_plant.GetPositions(station.get_internal_plant_context())
    print(
        f"\n[DEBUG] q from opt_plant_context    (len={len(q_opt)}):      {np.rad2deg(q_opt).round(2)} deg"
    )
    print(
        f"[DEBUG] q from internal_plant_context (len={len(q_internal)}): {np.rad2deg(q_internal).round(2)} deg"
    )

    # ----------------------------------------------------------------
    # 9. Build IRIS options using opt_plant limits
    # NOTE: Do NOT set iris_options.starting_ellipse here.
    # IrisFromCliqueCover manages its own per-seed ellipsoid internally.
    # Overriding it with a global sphere causes the MVIE to start from
    # a mismatched position and collapse to zero volume.
    # ----------------------------------------------------------------
    iris_options = IrisOptions()
    iris_options.termination_threshold = -1
    iris_options.relative_termination_threshold = -1
    iris_options.iteration_limit = 1

    options = IrisFromCliqueCoverOptions()
    options.coverage_termination_threshold = 0.9
    options.iris_options = iris_options

    print(f"\n[DEBUG] IRIS options summary:")
    print(
        f"  options.coverage_termination_threshold: {options.coverage_termination_threshold}"
    )
    print(
        f"  options.num_points_per_coverage_check: {options.num_points_per_coverage_check}"
    )

    regions = []
    generator = RandomGenerator(1235)

    print("\n[DEBUG] Calling IrisInConfigurationSpaceFromCliqueCover ...")
    print("=" * 60)
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

    with open("iris_regions.yaml", "w") as f:
        yaml.dump(regions, f)

    return regions


def compute_iris_regions_v2(station):
    """
    Debugging v2: fixes negative volume change by:
      - minimum_clique_size = nq + 2 (well-conditioned circumscribed ellipsoid in nq-D space)
      - num_points_per_visibility_round = 500 (denser visibility graph -> larger cliques)
      - termination thresholds disabled (let IRIS run all iterations even if volume shrinks)
    """
    nq = station.get_optimization_plant().num_positions()
    print(f"\n{'='*60}")
    print(f"[v2] compute_iris_regions_v2 | nq={nq}")
    print(f"{'='*60}")

    opt_diagram = station.internal_station.get_optimization_diagram()
    opt_plant = station.get_optimization_plant()

    iiwa_model_index = opt_plant.GetModelInstanceByName("iiwa")
    params = CollisionCheckerParams()
    params.model = opt_diagram
    params.robot_model_instances = [iiwa_model_index]
    params.edge_step_size = 0.05
    checker = SceneGraphCollisionChecker(params=params)

    lb_opt = opt_plant.GetPositionLowerLimits()
    ub_opt = opt_plant.GetPositionUpperLimits()
    bounding_box = HPolyhedron.MakeBox(lb_opt, ub_opt)

    # Hypothesis 1: minimum_clique_size must be >= nq+1 for a full-rank circumscribed
    # ellipsoid. 3 points in 7D -> under-determined ellipsoid -> huge -> gets cut ->
    # volume collapses to near zero on first IRIS iteration.
    # Hypothesis 2: disable termination thresholds so IRIS can recover across iterations.
    iris_options = IrisOptions()
    iris_options.iteration_limit = 10
    iris_options.bounding_region = bounding_box
    iris_options.termination_threshold = -1e10
    iris_options.relative_termination_threshold = -1e10

    clique_size = nq + 2  # 9 for 7-DOF; nq+1 spans the space, +1 for margin
    options = IrisFromCliqueCoverOptions()
    options.num_points_per_coverage_check = 1000
    options.coverage_termination_threshold = 0.7
    options.minimum_clique_size = clique_size
    options.num_points_per_visibility_round = 500
    options.iris_options = iris_options

    print(f"[v2] minimum_clique_size             = {options.minimum_clique_size}")
    print(
        f"[v2] num_points_per_visibility_round = {options.num_points_per_visibility_round}"
    )
    print(
        f"[v2] iris termination_threshold      = {iris_options.termination_threshold}"
    )
    print(f"[v2] iris iteration_limit            = {iris_options.iteration_limit}")

    regions = []
    generator = RandomGenerator(1235)

    print(f"\n[v2] Calling IrisInConfigurationSpaceFromCliqueCover ...")
    print("=" * 60)
    regions = IrisInConfigurationSpaceFromCliqueCover(
        checker=checker,
        options=options,
        generator=generator,
        sets=regions,
    )
    print("=" * 60)
    print(f"[v2] Done. Generated {len(regions)} regions.")

    for i, region in enumerate(regions):
        b_finite = np.all(np.isfinite(region.b()))
        print(
            f"  Region {i}: dim={region.ambient_dimension()}, "
            f"A.shape={region.A().shape}, b_finite={b_finite}, "
            f"Chebyshev center (deg)={np.rad2deg(region.ChebyshevCenter()).round(2)}"
        )

    return regions


def compute_iris_regions_v3(station):
    """
    Debugging v3: diagnose whether basic IrisInConfigurationSpace works at all.
      Step 1: call IrisInConfigurationSpace directly with a known-good seed.
      Step 2: if that works, try clique cover with minimum_clique_size=3 + disabled thresholds.
    This isolates whether the issue is IRIS itself or the clique cover wrapper.
    """
    from pydrake.all import IrisInConfigurationSpace

    nq = station.get_optimization_plant().num_positions()
    print(f"\n{'='*60}")
    print(f"[v3] compute_iris_regions_v3 | nq={nq}")
    print(f"{'='*60}")

    opt_diagram = station.internal_station.get_optimization_diagram()
    opt_plant = station.get_optimization_plant()

    iiwa_model_index = opt_plant.GetModelInstanceByName("iiwa")
    params = CollisionCheckerParams()
    params.model = opt_diagram
    params.robot_model_instances = [iiwa_model_index]
    params.edge_step_size = 0.05
    checker = SceneGraphCollisionChecker(params=params)

    lb_opt = opt_plant.GetPositionLowerLimits()
    ub_opt = opt_plant.GetPositionUpperLimits()
    bounding_box = HPolyhedron.MakeBox(lb_opt, ub_opt)

    # --- Step 1: standalone IrisInConfigurationSpace with known-good seed ---
    print("\n[v3] Step 1: standalone IrisInConfigurationSpace with q_home seed")
    q_home = np.array([0.0, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0])
    is_free = checker.CheckConfigCollisionFree(q_home)
    print(f"[v3]   q_home collision-free: {is_free}")

    if is_free:
        plant = checker.plant()
        plant_context = checker.plant_context(0)
        plant.SetPositions(plant_context, q_home)

        iris_opts = IrisOptions()
        iris_opts.iteration_limit = 5
        iris_opts.bounding_region = bounding_box
        iris_opts.termination_threshold = -1e10
        iris_opts.relative_termination_threshold = -1e10

        try:
            region = IrisInConfigurationSpace(plant, plant_context, iris_opts)
            cheb = region.ChebyshevCenter()
            print(
                f"[v3]   SUCCESS: dim={region.ambient_dimension()}, "
                f"A.shape={region.A().shape}, "
                f"Chebyshev center (deg)={np.rad2deg(cheb).round(2)}"
            )
        except Exception as e:
            print(f"[v3]   FAILED with exception: {e}")
    else:
        print("[v3]   q_home is in collision, skipping Step 1")

    # --- Step 2: clique cover with minimum_clique_size=3, thresholds disabled ---
    print("\n[v3] Step 2: clique cover, minimum_clique_size=3, thresholds disabled")
    iris_options = IrisOptions()
    iris_options.iteration_limit = 5
    iris_options.bounding_region = bounding_box
    iris_options.termination_threshold = -1e10
    iris_options.relative_termination_threshold = -1e10

    options = IrisFromCliqueCoverOptions()
    options.num_points_per_coverage_check = 500
    options.coverage_termination_threshold = 0.3
    options.minimum_clique_size = 3
    options.num_points_per_visibility_round = 200
    options.iris_options = iris_options

    regions = []
    generator = RandomGenerator(1235)

    print(f"[v3]   minimum_clique_size=3, termination_threshold=-1e10")
    print("=" * 60)
    regions = IrisInConfigurationSpaceFromCliqueCover(
        checker=checker,
        options=options,
        generator=generator,
        sets=regions,
    )
    print("=" * 60)
    print(f"[v3] Step 2 done. Generated {len(regions)} regions.")

    for i, region in enumerate(regions):
        b_finite = np.all(np.isfinite(region.b()))
        print(
            f"  Region {i}: dim={region.ambient_dimension()}, "
            f"A.shape={region.A().shape}, b_finite={b_finite}, "
            f"Chebyshev center (deg)={np.rad2deg(region.ChebyshevCenter()).round(2)}"
        )

    return regions


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
