"""
Interactive state machine diagram for scan_object_and_save_frames.py.

Usage:
    python utils/view_state_machine.py

Requires: networkx, matplotlib
Optional: pydot + graphviz (for better layout)
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------------------
# State transitions extracted from scan_object_and_save_frames.py
# (from_state, to_state, label)
# ---------------------------------------------------------------------------
TRANSITIONS = [
    ("IDLE", "COMPUTING_MOVE_TO_PRESCAN", "btn: Move to Prescan"),
    ("COMPUTING_MOVE_TO_PRESCAN", "MOVING_TO_PRESCAN", "plan ready"),
    ("MOVING_TO_PRESCAN", "WAITING_TO_GO_TO_START", "traj done"),
    ("MOVING_TO_PRESCAN", "DONE", "all scans done"),
    ("WAITING_TO_GO_TO_START", "COMPUTING_MOVE_TO_START", "btn: Move to Scan"),
    ("COMPUTING_MOVE_TO_START", "MOVING_TO_START", "plan ready"),
    ("MOVING_TO_START", "WAITING_FOR_NEXT_SCAN", "traj done"),
    ("WAITING_FOR_NEXT_SCAN", "DONE", "all scans done"),
    ("WAITING_FOR_NEXT_SCAN", "COMPUTING_IKS", "start IK threads"),
    ("COMPUTING_IKS", "MOVING_ALONG_HEMISPHERE", "valid + skip_opt"),
    ("COMPUTING_IKS", "PLANNING_ALONG_ALTERNATE_PATH", "invalid + skip_opt"),
    ("COMPUTING_IKS", "MOVING_DOWN_OPTICAL_AXIS", "IKs ready + !skip_opt"),
    (
        "PLANNING_ALONG_ALTERNATE_PATH",
        "COMPUTING_ALONG_ALTERNATE_PATH",
        "start GCS thread",
    ),
    ("COMPUTING_ALONG_ALTERNATE_PATH", "MOVING_ALONG_ALTERNATE_PATH", "plan success"),
    (
        "COMPUTING_ALONG_ALTERNATE_PATH",
        "WAITING_FOR_NEXT_SCAN",
        "plan failed (skip scan)",
    ),
    ("MOVING_ALONG_ALTERNATE_PATH", "WAITING_FOR_NEXT_SCAN", "traj done"),
    ("MOVING_ALONG_HEMISPHERE", "WAITING_FOR_NEXT_SCAN", "traj done"),
    ("MOVING_DOWN_OPTICAL_AXIS", "MOVING_ALONG_HEMISPHERE", "opt done + valid"),
    ("MOVING_DOWN_OPTICAL_AXIS", "PLANNING_ALONG_ALTERNATE_PATH", "opt done + invalid"),
]

# ---------------------------------------------------------------------------
# Node styling
# ---------------------------------------------------------------------------
NODE_COLORS = {
    "IDLE": "#aed6f1",  # light blue
    "DONE": "#a9dfbf",  # light green
    "WAITING_FOR_NEXT_SCAN": "#f9e79f",  # yellow
    "WAITING_TO_GO_TO_START": "#f9e79f",
    "COMPUTING_MOVE_TO_PRESCAN": "#d7bde2",  # purple
    "COMPUTING_MOVE_TO_START": "#d7bde2",
    "COMPUTING_IKS": "#d7bde2",
    "COMPUTING_ALONG_ALTERNATE_PATH": "#d7bde2",
    "MOVING_TO_PRESCAN": "#fad7a0",  # orange
    "MOVING_TO_START": "#fad7a0",
    "MOVING_ALONG_HEMISPHERE": "#fad7a0",
    "MOVING_DOWN_OPTICAL_AXIS": "#fad7a0",
    "MOVING_ALONG_ALTERNATE_PATH": "#fad7a0",
    "PLANNING_MOVE_TO_PRESCAN": "#d5f5e3",  # mint
    "PLANNING_MOVE_TO_START": "#d5f5e3",
    "PLANNING_ALONG_ALTERNATE_PATH": "#d5f5e3",
}

# Manual positions for a clean top-to-bottom layout
# (x, y) in data coordinates
POS = {
    "IDLE": (0, 10),
    "COMPUTING_MOVE_TO_PRESCAN": (0, 9),
    "MOVING_TO_PRESCAN": (0, 8),
    "WAITING_TO_GO_TO_START": (0, 7),
    "COMPUTING_MOVE_TO_START": (0, 6),
    "MOVING_TO_START": (0, 5),
    "WAITING_FOR_NEXT_SCAN": (0, 4),
    "COMPUTING_IKS": (0, 3),
    "MOVING_DOWN_OPTICAL_AXIS": (-3, 2),
    "MOVING_ALONG_HEMISPHERE": (0, 1),
    "PLANNING_ALONG_ALTERNATE_PATH": (3, 2),
    "COMPUTING_ALONG_ALTERNATE_PATH": (3, 1),
    "MOVING_ALONG_ALTERNATE_PATH": (3, 0),
    "DONE": (2, 8),
}


def build_graph():
    G = nx.MultiDiGraph()
    for src, dst, label in TRANSITIONS:
        G.add_edge(src, dst, label=label)
    return G


def draw(G):
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_title(
        "State Machine — scan_object_and_save_frames.py", fontsize=14, fontweight="bold"
    )
    ax.axis("off")

    nodes = list(G.nodes())
    colors = [NODE_COLORS.get(n, "#f0f0f0") for n in nodes]
    pos = {n: POS[n] for n in nodes if n in POS}
    # Any node missing a manual position falls back to spring
    missing = [n for n in nodes if n not in POS]
    if missing:
        spring = nx.spring_layout(G, seed=42)
        for n in missing:
            pos[n] = spring[n]

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=3000, node_shape="s", alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold")

    # Draw edges with slight curvature to separate parallel edges
    edge_list = list(G.edges(data=True, keys=True))
    # Count edges per (src, dst) pair to offset them
    from collections import defaultdict

    pair_count = defaultdict(int)
    pair_idx = defaultdict(int)
    for src, dst, key, _ in edge_list:
        pair_count[(src, dst)] += 1

    for src, dst, key, data in edge_list:
        n = pair_count[(src, dst)]
        i = pair_idx[(src, dst)]
        pair_idx[(src, dst)] += 1
        rad = 0.1 + 0.15 * i if n > 1 else 0.1
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=[(src, dst)],
            connectionstyle=f"arc3,rad={rad}",
            arrows=True,
            arrowsize=15,
            width=1.5,
            edge_color="#555555",
            min_source_margin=30,
            min_target_margin=30,
        )

    # Edge labels — place near midpoint with slight offset
    edge_labels = {(src, dst): data["label"] for src, dst, key, data in edge_list}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        ax=ax,
        edge_labels=edge_labels,
        font_size=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        label_pos=0.4,
    )

    # Legend
    legend_items = [
        mpatches.Patch(color="#aed6f1", label="Entry / terminal"),
        mpatches.Patch(color="#f9e79f", label="Waiting (user input)"),
        mpatches.Patch(color="#d7bde2", label="Computing (background thread)"),
        mpatches.Patch(color="#fad7a0", label="Moving (executing trajectory)"),
        mpatches.Patch(color="#d5f5e3", label="Planning"),
        mpatches.Patch(color="#a9dfbf", label="Done"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    G = build_graph()
    draw(G)
