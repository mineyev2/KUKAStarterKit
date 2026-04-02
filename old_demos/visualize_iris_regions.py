"""
Interactive IRIS region visualizer using robot skeleton line segments.

- "Region" slider (1..n_regions): moving it clears everything and draws a
  fresh sample from only that region.
- "Show All" button: clears everything and draws num_samples configs sampled
  randomly across all regions (region picked uniformly at random per config).
- "Num samples" slider: how many configs to draw on the next action.

Usage:
    python visualize_iris_regions.py
    python visualize_iris_regions.py --max_samples 500
    python visualize_iris_regions.py --iris_yaml path/to/iris_regions.yaml
"""

import argparse
import time

import numpy as np

from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    ConstantVectorSource,
    DiagramBuilder,
    LoadIrisRegionsYamlFile,
    MeshcatVisualizer,
    RandomGenerator,
    Rgba,
    RigidTransform,
    Simulator,
    Sphere,
)
from termcolor import colored

from iiwa_setup.iiwa import IiwaHardwareStationDiagram

REGION_COLORS = [
    Rgba(1.0, 0.2, 0.2, 0.9),  # red
    Rgba(0.2, 0.9, 0.2, 0.9),  # green
    Rgba(0.2, 0.4, 1.0, 0.9),  # blue
    Rgba(1.0, 0.85, 0.1, 0.9),  # yellow
    Rgba(1.0, 0.2, 1.0, 0.9),  # magenta
    Rgba(0.1, 1.0, 1.0, 0.9),  # cyan
    Rgba(1.0, 0.5, 0.1, 0.9),  # orange
    Rgba(0.6, 0.1, 1.0, 0.9),  # purple
    Rgba(0.3, 0.8, 0.5, 0.9),  # teal
    Rgba(0.8, 0.5, 0.3, 0.9),  # brown-orange
]

SKELETON_LINKS = [
    "iiwa_link_0",
    "iiwa_link_1",
    "iiwa_link_2",
    "iiwa_link_3",
    "iiwa_link_4",
    "iiwa_link_5",
    "iiwa_link_6",
    "iiwa_link_7",
    "microscope_tip_link",
]

_seed_counter = 0


def fresh_generator():
    global _seed_counter
    _seed_counter += 1
    return RandomGenerator(_seed_counter)


def sample_from_hpolyhedron(region, n_samples: int):
    generator = fresh_generator()
    center = region.ChebyshevCenter()
    samples = [center.copy()]
    prev = center.copy()
    for _ in range(n_samples - 1):
        prev = region.UniformSample(generator, prev)
        samples.append(prev.copy())
    return np.array(samples)


def draw_skeleton(meshcat, plant, plant_context, q, path, color):
    plant.SetPositions(plant_context, q)
    positions = []
    for link_name in SKELETON_LINKS:
        try:
            body = plant.GetBodyByName(link_name)
            X_WB = plant.EvalBodyPoseInWorld(plant_context, body)
            positions.append(X_WB.translation())
        except Exception:
            pass
    if len(positions) >= 2:
        meshcat.SetLine(path, np.array(positions).T, line_width=3.0, rgba=color)


def draw_tip_point(meshcat, plant, plant_context, q, path, color):
    plant.SetPositions(plant_context, q)
    body = plant.GetBodyByName("microscope_tip_link")
    X_WB = plant.EvalBodyPoseInWorld(plant_context, body)
    meshcat.SetObject(path, Sphere(0.005), color)
    meshcat.SetTransform(path, RigidTransform(X_WB.translation()))


def draw_config(meshcat, plant, plant_context, q, path, color, skeleton_mode):
    if skeleton_mode:
        draw_skeleton(meshcat, plant, plant_context, q, path, color)
    else:
        draw_tip_point(meshcat, plant, plant_context, q, path, color)


def show_one_region(
    meshcat,
    plant,
    plant_context,
    iris_regions,
    region_names,
    region_idx,
    n_samples,
    skeleton_mode,
):
    """Clear everything, then draw n_samples configs from region_idx."""
    meshcat.Delete("iris_ghosts")
    name = region_names[region_idx]
    color = REGION_COLORS[region_idx % len(REGION_COLORS)]
    samples = sample_from_hpolyhedron(iris_regions[name], n_samples)
    for j, q in enumerate(samples):
        draw_config(
            meshcat,
            plant,
            plant_context,
            q,
            f"iris_ghosts/{name}/config_{j:03d}",
            color,
            skeleton_mode,
        )


def show_all_regions(
    meshcat, plant, plant_context, iris_regions, region_names, n_samples, skeleton_mode
):
    """Clear everything, then draw n_samples configs sampled randomly across all regions."""
    meshcat.Delete("iris_ghosts")
    rng = np.random.default_rng(_seed_counter)
    for k in range(n_samples):
        i = int(rng.integers(len(region_names)))
        name = region_names[i]
        color = REGION_COLORS[i % len(REGION_COLORS)]
        q = sample_from_hpolyhedron(iris_regions[name], 1)[0]
        draw_config(
            meshcat,
            plant,
            plant_context,
            q,
            f"iris_ghosts/all/config_{k:03d}",
            color,
            skeleton_mode,
        )


def main(iris_yaml_path: str, max_samples: int) -> None:
    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    plant_config:
        time_step: 0.005
        contact_model: "hydroelastic_with_fallback"
        discrete_contact_approximation: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            lcm_bus: "default"
            control_mode: position_only
    lcm_buses:
        default:
            lcm_url: ""
    """

    # ------------------------------------------------------------------
    # Load IRIS regions
    # ------------------------------------------------------------------
    print(colored(f"Loading IRIS regions from: {iris_yaml_path}", "cyan"))
    iris_regions = LoadIrisRegionsYamlFile(iris_yaml_path)
    region_names = list(iris_regions.keys())
    n_regions = len(region_names)
    print(colored(f"Loaded {n_regions} regions: {region_names}", "green"))

    # ------------------------------------------------------------------
    # Build diagram (single robot, used only for FK)
    # ------------------------------------------------------------------
    builder = DiagramBuilder()
    scenario = LoadScenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_dist=0.8,
            hemisphere_angle=np.deg2rad(60),
            hemisphere_radius=0.100,
            use_hardware=False,
        ),
    )

    default_position = np.array([1.57079, 0.1, 0, -1.2, 0, 1.6, 0])
    dummy = builder.AddSystem(ConstantVectorSource(default_position))
    builder.Connect(dummy.get_output_port(), station.GetInputPort("iiwa.position"))

    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.AdvanceTo(0.01)

    plant = station.get_internal_plant()
    plant_context = station.get_internal_plant_context()
    meshcat = station.internal_meshcat

    # ------------------------------------------------------------------
    # Meshcat controls
    # ------------------------------------------------------------------
    meshcat.AddSlider("Region", 1, n_regions, 1, 1)
    meshcat.AddSlider("Num samples", 1, max_samples, 1, 20)
    meshcat.AddButton("Show All")
    meshcat.AddButton("Toggle Skeleton / Tips")
    meshcat.AddButton("Exit")

    skeleton_mode = True  # start in skeleton mode
    toggle_clicks = 0

    # Draw initial view: region 1
    n_show = int(round(meshcat.GetSliderValue("Num samples")))
    show_one_region(
        meshcat,
        plant,
        plant_context,
        iris_regions,
        region_names,
        0,
        n_show,
        skeleton_mode,
    )

    prev_region_idx = 0
    show_all_clicks = 0
    last_was_show_all = False  # track whether the last draw was "show all"

    print(colored(f"\nVisualization ready → {meshcat.web_url()}", "cyan"))
    print("Open that URL in a browser.  Press Ctrl-C or click Exit to quit.\n")

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------
    while meshcat.GetButtonClicks("Exit") < 1:
        n_show = int(round(meshcat.GetSliderValue("Num samples")))
        region_idx = int(round(meshcat.GetSliderValue("Region"))) - 1  # 0-indexed

        # Toggle skeleton ↔ tip-points — redraws current view in new mode
        new_toggle = meshcat.GetButtonClicks("Toggle Skeleton / Tips")
        if new_toggle > toggle_clicks:
            toggle_clicks = new_toggle
            skeleton_mode = not skeleton_mode
            mode_str = "Skeleton" if skeleton_mode else "Tip points"
            print(colored(f"View mode → {mode_str}", "cyan"))
            if last_was_show_all:
                show_all_regions(
                    meshcat,
                    plant,
                    plant_context,
                    iris_regions,
                    region_names,
                    n_show,
                    skeleton_mode,
                )
            else:
                show_one_region(
                    meshcat,
                    plant,
                    plant_context,
                    iris_regions,
                    region_names,
                    prev_region_idx,
                    n_show,
                    skeleton_mode,
                )

        # Show All button
        elif meshcat.GetButtonClicks("Show All") > show_all_clicks:
            show_all_clicks = meshcat.GetButtonClicks("Show All")
            show_all_regions(
                meshcat,
                plant,
                plant_context,
                iris_regions,
                region_names,
                n_show,
                skeleton_mode,
            )
            prev_region_idx = -1
            last_was_show_all = True

        # Region slider changed → show only that region
        elif region_idx != prev_region_idx:
            prev_region_idx = region_idx
            last_was_show_all = False
            show_one_region(
                meshcat,
                plant,
                plant_context,
                iris_regions,
                region_names,
                region_idx,
                n_show,
                skeleton_mode,
            )

        time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive IRIS region visualizer (line-segment skeletons)."
    )
    parser.add_argument(
        "--iris_yaml",
        type=str,
        default="iris_regions_old.yaml",
        help="Path to IRIS regions YAML (default: iris_regions.yaml).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Max samples per region — also the slider upper bound (default: 500).",
    )
    args = parser.parse_args()
    main(args.iris_yaml, args.max_samples)
