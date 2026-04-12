from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from gsplat.strategy import DefaultStrategy, MCMCStrategy
from typing_extensions import Literal, assert_never


@dataclass
class Config:
    # ── Viewer ────────────────────────────────────────────────────────────────
    disable_viewer: bool = False

    # ── Checkpoint / export ───────────────────────────────────────────────────
    ckpt: Optional[List[str]] = None
    compression: Optional[Literal["png"]] = None
    render_traj_path: str = "interp"
    render_traj_trim: Optional[
        int
    ] = 5  # frames to drop from each end before path generation; None disables trim
    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = "data/360_v2/garden"
    data_factor: int = 4
    result_dir: str = "results/garden"
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # ── Viewer server ─────────────────────────────────────────────────────────
    port: int = 8080

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = False
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    disable_video: bool = False

    # ── Gaussian initialization ───────────────────────────────────────────────
    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2

    # ── Clipping planes ───────────────────────────────────────────────────────
    near_plane: float = 0.01
    far_plane: float = 1e10

    # ── Densification strategy ────────────────────────────────────────────────
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )

    # ── Rasterization ─────────────────────────────────────────────────────────
    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = False
    antialiased: bool = False
    random_bkgd: bool = False

    # ── Learning rates ────────────────────────────────────────────────────────
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    # ── Regularisation ────────────────────────────────────────────────────────
    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    # ── Pose / camera optimisation ────────────────────────────────────────────
    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0

    # ── Appearance optimisation ───────────────────────────────────────────────
    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6

    # ── Bilateral grid ────────────────────────────────────────────────────────
    use_bilateral_grid: bool = False
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # ── Depth loss ────────────────────────────────────────────────────────────
    depth_loss: bool = False
    depth_lambda: float = 1e-2

    # ── Tensorboard ───────────────────────────────────────────────────────────
    tb_every: int = 100
    tb_save_image: bool = False

    # ── LPIPS ─────────────────────────────────────────────────────────────────
    lpips_net: Literal["vgg", "alex"] = "alex"

    # ── 3DGUT ─────────────────────────────────────────────────────────────────
    with_ut: bool = False
    with_eval3d: bool = False
    use_fused_bilagrid: bool = False

    # ─────────────────────────────────────────────────────────────────────────

    def adjust_steps(self, factor: float) -> None:
        """Scale all step-indexed hyper-parameters by *factor* (e.g. 0.25 for 4-GPU runs)."""
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)
