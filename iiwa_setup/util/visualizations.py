import numpy as np

from pydrake.all import (
    Box,
    CoulombFriction,
    Cylinder,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Sphere,
)


def draw_sphere(meshcat, name, position, radius=0.01):
    rgba = Rgba(0.0, 1.0, 0.1, 0.5)

    meshcat.SetObject(
        name,
        Sphere(radius),
        rgba,
    )
    meshcat.SetTransform(
        name,
        RigidTransform(np.array(position)),
    )


# TODO: These are collisions as well so maybe don't just add into visualizations.py?
def draw_triad(meshcat, name, transform, length=0.1, radius=0.005, opacity=1.0):
    """Draws a coordinate frame triad in Meshcat at the given RigidTransform."""
    meshcat.SetObject(f"{name}/x", Cylinder(radius, length), Rgba(1, 0, 0, opacity))
    meshcat.SetTransform(
        f"{name}/x",
        RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2, 0, 0]),
    )

    meshcat.SetObject(f"{name}/y", Cylinder(radius, length), Rgba(0, 1, 0, opacity))
    meshcat.SetTransform(
        f"{name}/y",
        RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2, 0]),
    )

    meshcat.SetObject(f"{name}/z", Cylinder(radius, length), Rgba(0, 0, 1, opacity))
    meshcat.SetTransform(f"{name}/z", RigidTransform([0, 0, length / 2]))

    # Set the overall frame transform
    meshcat.SetTransform(name, transform)


def add_sphere(
    plant,
    position,
    radius=0.01,
    name="sphere",
    color=[0.0, 1.0, 0.0, 0.2],
    collision=True,
):
    friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.8)

    if radius <= 0:
        radius = 0.01  # default small radius to avoid issues with zero-size geometry
    sphere_shape = Sphere(radius)
    X_WC = RigidTransform(np.array(position))

    if collision:
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            X_WC,
            sphere_shape,
            f"{name}_collision",
            friction,
        )

    # Optional: visualization
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_WC,
        sphere_shape,
        f"{name}_visual",
        color,
    )


def add_floor(plant):
    friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.8)
    floor_thickness = 0.05
    floor_length = 1.5 + 0.3  # extra to extend beyond robot base to the wall
    floor_size = Box(floor_length, 3.0, floor_thickness)
    X_WF = RigidTransform(
        [floor_length / 2 - 0.3, 0, -floor_thickness / 2]
    )  # top surface at z=0

    plant.RegisterCollisionGeometry(
        plant.world_body(),
        X_WF,
        floor_size,
        "floor_collision",
        friction,
    )

    # Optional: visualization
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_WF,
        floor_size,
        "floor_visual",
        [58 / 255.0, 85 / 255.0, 69 / 255.0, 0.3],
    )


def add_wall(plant, X_WF=None):
    if not hasattr(add_wall, "counter"):
        add_wall.counter = 0

    add_wall.counter += 1

    friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.8)
    wall_thickness = 0.01
    wall_height = 1.5
    wall_size = Box(wall_thickness, 3.0, wall_height)

    if X_WF is None:
        X_WF = RigidTransform([-0.3 - wall_thickness / 2, 0, wall_height / 2])

    plant.RegisterCollisionGeometry(
        plant.world_body(),
        X_WF,
        wall_size,
        f"wall_{add_wall.counter}_collision",
        friction,
    )

    # Optional: visualization
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_WF,
        wall_size,
        f"wall_{add_wall.counter}_visual",
        [58 / 255.0, 85 / 255.0, 69 / 255.0, 0.3],
    )
