"""
RRT*-Connect motion planner for the iiwa arm in joint space.

Extends RRT-Connect with path-cost optimization: each new node selects the
best parent among nearby nodes (minimum cost from root) and rewires nearby
nodes through the new node if it lowers their cost.

Author: Roman Mineyev
"""

import time

import numpy as np

from termcolor import colored

from utils.RRT import RRTConnect


class RRTStarConnect(RRTConnect):
    """
    RRT*-Connect: bidirectional RRT with rewiring for path-cost optimization.

    On each iteration:
      1. Extend tree A one step with RRT* (best parent + rewire nearby).
      2. Greedily connect tree B to the new node (also with rewiring per step).
      3. Swap trees and repeat.

    The rewiring radius shrinks as the tree grows following the RRT* schedule,
    ensuring asymptotic optimality.
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
        rewire_radius: float = 0.3,
    ):
        """
        Args:
            rewire_radius: Max rewire radius (rad). Adaptive radius is clamped
                           to this value and shrinks with tree size per RRT* schedule.
            All other args: same as RRTConnect.
        """
        super().__init__(
            station=station,
            vel_limits=vel_limits,
            acc_limits=acc_limits,
            step_size=step_size,
            max_iter=max_iter,
            goal_bias=goal_bias,
            num_collision_interp_steps=num_collision_interp_steps,
        )
        self.rewire_radius = rewire_radius

    # ------------------------------------------------------------------
    # RRT* helpers
    # ------------------------------------------------------------------

    def _near_nodes(self, nodes: list, q: np.ndarray, radius: float) -> list:
        """Return indices of all nodes within radius of q."""
        return [i for i, node in enumerate(nodes) if np.linalg.norm(node - q) <= radius]

    def _adaptive_radius(self, n: int) -> float:
        """
        Compute the RRT* rewire radius for a tree of size n.
        Shrinks as O((log n / n)^(1/d)) to ensure asymptotic optimality.
        """
        d = self.num_joints
        gamma = self.rewire_radius * 2.0
        r = gamma * (np.log(n + 1) / (n + 1)) ** (1.0 / d)
        return min(r, self.rewire_radius)

    def _extend_star(
        self, nodes: list, parents: list, costs: list, q_target: np.ndarray
    ):
        """
        RRT*-style single-step extension toward q_target.

        Steps:
          1. Steer from nearest node toward q_target.
          2. Among nearby nodes, choose the parent with lowest cost-from-root.
          3. Add the new node.
          4. Rewire nearby nodes through the new node if it lowers their cost.

        Returns:
            (status, new_node_idx) — same convention as base _extend.
        """
        idx_near = self._nearest(nodes, q_target)
        q_near = nodes[idx_near]
        q_new = self._steer(q_near, q_target)

        if not self.is_edge_collision_free(q_near, q_new):
            return self.TRAPPED, -1

        # Candidate nearby nodes for parent selection / rewiring
        radius = self._adaptive_radius(len(nodes))
        near_indices = self._near_nodes(nodes, q_new, radius)

        # Choose best parent (lowest cost-from-root + edge cost)
        best_parent = idx_near
        best_cost = costs[idx_near] + np.linalg.norm(q_new - q_near)
        for i in near_indices:
            c = costs[i] + np.linalg.norm(q_new - nodes[i])
            if c < best_cost and self.is_edge_collision_free(nodes[i], q_new):
                best_cost = c
                best_parent = i

        # Add new node
        nodes.append(q_new)
        parents.append(best_parent)
        costs.append(best_cost)
        new_idx = len(nodes) - 1

        # Rewire: redirect nearby nodes through q_new if cheaper
        for i in near_indices:
            c = best_cost + np.linalg.norm(nodes[i] - q_new)
            if c < costs[i] and self.is_edge_collision_free(q_new, nodes[i]):
                parents[i] = new_idx
                costs[i] = c

        if np.linalg.norm(q_new - q_target) < 1e-6:
            return self.REACHED, new_idx
        return self.ADVANCED, new_idx

    def _connect_star(
        self, nodes: list, parents: list, costs: list, q_target: np.ndarray
    ):
        """
        Greedy connect using RRT* extensions (rewiring on every step).

        Returns (status, last_node_idx) — same as base _connect.
        """
        status = self.ADVANCED
        last_idx = -1
        while status == self.ADVANCED:
            status, last_idx = self._extend_star(nodes, parents, costs, q_target)
        return status, last_idx

    # ------------------------------------------------------------------
    # Public API (overrides RRTConnect.plan)
    # ------------------------------------------------------------------

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray):
        """
        Plan a collision-free, cost-optimized path from q_start to q_goal.

        Returns:
            path: list of np.ndarray (joint configs) start→goal, or None on failure.
        """
        if not self.is_collision_free(q_start):
            print(colored("❌ RRTStarConnect: q_start is in collision!", "red"))
            return None
        if not self.is_collision_free(q_goal):
            print(colored("❌ RRTStarConnect: q_goal is in collision!", "red"))
            return None

        # costs[i] = path length from root to nodes[i]
        nodes_a, parents_a, costs_a = [q_start.copy()], [-1], [0.0]
        nodes_b, parents_b, costs_b = [q_goal.copy()], [-1], [0.0]
        a_is_start = True

        t0 = time.time()
        for iteration in range(self.max_iter):
            if np.random.rand() < self.goal_bias:
                q_rand = q_goal.copy()
            else:
                q_rand = self._random_config()

            # RRT* extend tree A (rewiring included)
            status_a, idx_a_new = self._extend_star(nodes_a, parents_a, costs_a, q_rand)
            if status_a == self.TRAPPED:
                continue

            # Greedy RRT* connect tree B to the new node in tree A
            q_bridge = nodes_a[idx_a_new]
            status_b, idx_b_last = self._connect_star(
                nodes_b, parents_b, costs_b, q_bridge
            )

            if status_b == self.REACHED:
                elapsed = time.time() - t0
                print(
                    colored(
                        f"✓ RRTStarConnect found path after {iteration + 1} iterations "
                        f"({elapsed:.2f}s), "
                        f"tree sizes: A={len(nodes_a)}, B={len(nodes_b)}",
                        "green",
                    )
                )
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

            # Swap trees
            nodes_a, nodes_b = nodes_b, nodes_a
            parents_a, parents_b = parents_b, parents_a
            costs_a, costs_b = costs_b, costs_a
            a_is_start = not a_is_start

        elapsed = time.time() - t0
        print(
            colored(
                f"❌ RRTStarConnect failed after {self.max_iter} iterations ({elapsed:.2f}s).",
                "red",
            )
        )
        return None


def plan_rrt_star_async(
    station,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    result_dict: dict,
    step_size: float = 0.05,
    max_iter: int = 10000,
    rewire_radius: float = 0.3,
):
    """
    Background-thread wrapper for RRTStarConnect.

    result_dict keys set on completion: ready, success, trajectory, path.
    """
    planner = RRTStarConnect(
        station=station,
        vel_limits=vel_limits,
        acc_limits=acc_limits,
        step_size=step_size,
        max_iter=max_iter,
        rewire_radius=rewire_radius,
    )
    trajectory, path = planner.plan_to_trajectory(q_start, q_goal)
    result_dict["trajectory"] = trajectory
    result_dict["path"] = path
    result_dict["success"] = trajectory is not None
    result_dict["ready"] = True
