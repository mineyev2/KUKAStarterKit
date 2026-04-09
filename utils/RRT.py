"""
RRTConnect motion planner for the iiwa arm in joint space.

Uses the station's optimization plant for collision detection.
"""

import time

import numpy as np

from pydrake.all import PiecewisePolynomial, Rgba, RigidTransform, Sphere
from termcolor import colored

from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra


class RRTConnect:
    """
    Bidirectional RRT (RRT-Connect) planner operating in joint configuration space.

    Collision detection uses the station's optimization plant (same plant used
    by GCS / trajectory optimization).
    """

    def __init__(
        self,
        station,
        vel_limits: np.ndarray,
        acc_limits: np.ndarray,
        step_size: float = 0.05,
        max_iter: int = 10000,
        goal_bias: float = 0.1,
        num_collision_interp_steps: int = 10,
    ):
        """
        Args:
            station: IiwaHardwareStationDiagram instance.
            vel_limits: (7,) joint velocity limits (rad/s).
            acc_limits: (7,) joint acceleration limits (rad/s^2).
            step_size: Max step size in joint space (rad) per RRT extension.
            max_iter: Maximum number of tree extension iterations.
            goal_bias: Probability of sampling the goal directly.
            num_collision_interp_steps: Interpolation steps used for edge collision check.
        """
        self.station = station
        self.vel_limits = vel_limits
        self.acc_limits = acc_limits
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.num_collision_interp_steps = num_collision_interp_steps

        # Pull joint limits from the internal plant
        plant = station.get_internal_plant()
        self.q_lower = plant.GetPositionLowerLimits()
        self.q_upper = plant.GetPositionUpperLimits()
        self.num_joints = plant.num_positions()

        # Collision-check resources (optimization plant)
        self._opt_plant = station.internal_station.get_optimization_plant()
        self._opt_plant_context = (
            station.internal_station.get_optimization_plant_context()
        )
        self._opt_sg = station.internal_station.get_optimization_diagram_sg()
        self._opt_sg_context = (
            station.internal_station.get_optimization_diagram_sg_context()
        )
        self._iiwa_model = self._opt_plant.GetModelInstanceByName("iiwa")

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Return True if q is collision-free."""
        self._opt_plant.SetPositions(self._opt_plant_context, self._iiwa_model, q)
        query_object = self._opt_sg.get_query_output_port().Eval(self._opt_sg_context)
        return not query_object.HasCollisions()

    def is_edge_collision_free(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        """Return True if the straight-line edge in joint space is collision-free."""
        for t in np.linspace(0.0, 1.0, self.num_collision_interp_steps):
            q_interp = q_from + t * (q_to - q_from)
            if not self.is_collision_free(q_interp):
                return False
        return True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _random_config(self) -> np.ndarray:
        return np.random.uniform(self.q_lower, self.q_upper)

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest(tree_nodes: list, q: np.ndarray) -> int:
        """Return index of nearest node in tree to q."""
        dists = [np.linalg.norm(node - q) for node in tree_nodes]
        return int(np.argmin(dists))

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """Move from q_from toward q_to by at most step_size."""
        diff = q_to - q_from
        dist = np.linalg.norm(diff)
        if dist <= self.step_size:
            return q_to.copy()
        return q_from + (diff / dist) * self.step_size

    # ------------------------------------------------------------------
    # RRT-Connect core
    # ------------------------------------------------------------------

    TRAPPED = 0
    ADVANCED = 1
    REACHED = 2

    def _extend(self, nodes: list, parents: list, q_target: np.ndarray):
        """
        Extend tree toward q_target by one step.

        Returns:
            status: TRAPPED | ADVANCED | REACHED
            new_node_idx: index of the added node (or -1 if TRAPPED)
        """
        idx_near = self._nearest(nodes, q_target)
        q_near = nodes[idx_near]
        q_new = self._steer(q_near, q_target)

        if not self.is_edge_collision_free(q_near, q_new):
            return self.TRAPPED, -1

        nodes.append(q_new)
        parents.append(idx_near)
        new_idx = len(nodes) - 1

        if np.linalg.norm(q_new - q_target) < 1e-6:
            return self.REACHED, new_idx
        return self.ADVANCED, new_idx

    def _connect(self, nodes: list, parents: list, q_target: np.ndarray):
        """
        Greedily extend tree toward q_target until REACHED or TRAPPED.

        Returns:
            status: TRAPPED | REACHED
            last_node_idx: index of the last added node (-1 if TRAPPED on first step)
        """
        status = self.ADVANCED
        last_idx = -1
        while status == self.ADVANCED:
            status, last_idx = self._extend(nodes, parents, q_target)
        return status, last_idx

    @staticmethod
    def _extract_path(nodes_a, parents_a, idx_a, nodes_b, parents_b, idx_b):
        """
        Reconstruct path from root of A to root of B through the bridge.

        Walking up parents from idx_a gives [bridge, ..., root_A].
        Reversed: [root_A, ..., bridge].

        Walking up parents from idx_b gives [bridge, ..., root_B].
        We want [bridge, ..., root_B] as-is (bridge→root_B direction).
        Skip path_b[0] (the bridge) to avoid duplicating the junction.
        """
        # Walk up tree A: [bridge, ..., root_A] → reverse → [root_A, ..., bridge]
        path_a = []
        i = idx_a
        while i != -1:
            path_a.append(nodes_a[i])
            i = parents_a[i]
        path_a.reverse()

        # Walk up tree B: [bridge, ..., root_B] — keep this order (bridge→root_B)
        path_b = []
        i = idx_b
        while i != -1:
            path_b.append(nodes_b[i])
            i = parents_b[i]

        # Concatenate: root_A → ... → bridge → ... → root_B
        # Skip path_b[0] (bridge) since path_a already ends there
        return path_a + path_b[1:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray):
        """
        Plan a collision-free path from q_start to q_goal.

        Returns:
            path: list of np.ndarray (joint configs) from start to goal,
                  or None if planning failed.
        """
        if not self.is_collision_free(q_start):
            print(colored("❌ RRTConnect: q_start is in collision!", "red"))
            return None
        if not self.is_collision_free(q_goal):
            print(colored("❌ RRTConnect: q_goal is in collision!", "red"))
            return None

        # Tree A: rooted at start. Tree B: rooted at goal.
        nodes_a, parents_a = [q_start.copy()], [-1]
        nodes_b, parents_b = [q_goal.copy()], [-1]
        # After each swap, A and B alternate roles. Track so we can
        # orient the final path correctly (always start→goal).
        a_is_start = True

        t0 = time.time()
        for iteration in range(self.max_iter):
            # Sample
            if np.random.rand() < self.goal_bias:
                q_rand = q_goal.copy()
            else:
                q_rand = self._random_config()

            # Extend tree A toward sample
            status_a, idx_a_new = self._extend(nodes_a, parents_a, q_rand)
            if status_a == self.TRAPPED:
                continue

            # Try to connect tree B to the new node in tree A
            q_bridge = nodes_a[idx_a_new]
            status_b, idx_b_last = self._connect(nodes_b, parents_b, q_bridge)

            if status_b == self.REACHED:
                elapsed = time.time() - t0
                print(
                    colored(
                        f"✓ RRTConnect found path after {iteration + 1} iterations "
                        f"({elapsed:.2f}s), "
                        f"tree sizes: A={len(nodes_a)}, B={len(nodes_b)}",
                        "green",
                    )
                )
                # _extract_path returns root_A → ... → root_B.
                # If A is the start tree that's already start→goal; otherwise reverse.
                path = self._extract_path(
                    nodes_a,
                    parents_a,
                    idx_a_new,
                    nodes_b,
                    parents_b,
                    idx_b_last,
                )
                if not a_is_start:
                    path = path[::-1]
                return path

            # Swap trees to alternate growth
            nodes_a, nodes_b = nodes_b, nodes_a
            parents_a, parents_b = parents_b, parents_a
            a_is_start = not a_is_start

        elapsed = time.time() - t0
        print(
            colored(
                f"❌ RRTConnect failed after {self.max_iter} iterations ({elapsed:.2f}s).",
                "red",
            )
        )
        return None

    def plan_to_trajectory(self, q_start: np.ndarray, q_goal: np.ndarray):
        """
        Plan a path and reparameterize it with TOPPRA.

        Returns:
            trajectory: PathParameterizedTrajectory (starts at t=0), or None on failure.
            path: raw list of configs, or None on failure.
        """
        path = self.plan(q_start, q_goal)
        if path is None:
            return None, None

        configs = np.array(path).T  # shape (7, N)
        n = configs.shape[1]
        # Uniform pseudo-time breakpoints — TOPPRA will retime
        t_breaks = np.linspace(0.0, float(n - 1), n)
        geometric_path = PiecewisePolynomial.FirstOrderHold(t_breaks, configs)

        controller_plant = self.station.get_iiwa_controller_plant()
        trajectory, success = reparameterize_with_toppra(
            geometric_path,
            controller_plant,
            velocity_limits=self.vel_limits,
            acceleration_limits=self.acc_limits,
        )
        if not success:
            return None, None
        return trajectory, path


def plot_rrt_raw_path_in_meshcat(
    station,
    path: list,
    name: str = "rrt_raw_path",
    rgba: Rgba = Rgba(1.0, 0.4, 0.0, 1.0),
    dot_radius: float = 0.005,
    line_width: float = 2.0,
):
    """
    Visualize the raw (unsmoothed) RRT path in Meshcat.

    Draws:
      - A sphere at each waypoint's end-effector position.
      - A polyline connecting all waypoints in order.

    Args:
        station: IiwaHardwareStationDiagram.
        path: List of (7,) joint-config arrays returned by RRTConnect.plan().
        name: Meshcat path prefix used for the objects.
        rgba: Color for both dots and line.
        dot_radius: Radius of the sphere markers.
        line_width: Width of the connecting polyline.
    """
    if not path:
        return

    meshcat = station.internal_meshcat
    plant = station.get_internal_plant()
    context = station.get_internal_plant_context()

    # Forward-kinematics for every waypoint
    tip_body = plant.GetBodyByName("microscope_tip_link")
    positions = []
    for q in path:
        plant.SetPositions(context, q)
        X_WB = plant.EvalBodyPoseInWorld(context, tip_body)
        positions.append(X_WB.translation())

    # Clear stale visuals under this name
    meshcat.Delete(name)

    # Draw a sphere at each waypoint
    sphere = Sphere(dot_radius)
    for i, pos in enumerate(positions):
        meshcat.SetObject(f"{name}/dots/{i}", sphere, rgba)
        meshcat.SetTransform(f"{name}/dots/{i}", RigidTransform(pos))

    # Draw a polyline connecting the waypoints
    pts = np.array(positions).T  # shape (3, N)
    meshcat.SetLine(f"{name}/line", pts, line_width=line_width, rgba=rgba)


def plan_rrt_async(
    station,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    result_dict: dict,
    step_size: float = 0.05,
    max_iter: int = 10000,
):
    """
    Background-thread wrapper: runs RRTConnect and stores results in result_dict.

    result_dict keys set on completion:
        ready     bool
        success   bool
        trajectory  PathParameterizedTrajectory | None
        path        list[np.ndarray] | None
    """
    planner = RRTConnect(
        station=station,
        vel_limits=vel_limits,
        acc_limits=acc_limits,
        step_size=step_size,
        max_iter=max_iter,
    )
    trajectory, path = planner.plan_to_trajectory(q_start, q_goal)
    result_dict["trajectory"] = trajectory
    result_dict["path"] = path
    result_dict["success"] = trajectory is not None
    result_dict["ready"] = True
