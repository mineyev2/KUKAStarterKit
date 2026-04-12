"""Runner — orchestrates setup, rasterization, and the training loop.

Heavy output concerns are delegated:
  TrainingLogger  (trainer/logger.py)    — TensorBoard, checkpoints, PLY export
  Evaluator       (trainer/evaluator.py) — eval metrics, videos, compression

The training loop (``train``) is kept readable by extracting one full
forward-backward-optimizer iteration into ``_train_step``.
"""

import math
import os
import time

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import tqdm
import viser
import yaml

from datasets.colmap import Dataset, Parser
from fused_ssim import fused_ssim
from gsplat.compression import PngCompression
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatRenderTabState, GsplatViewer
from nerfview import CameraState, RenderTabState, apply_float_colormap
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Literal, assert_never

from utils import AppearanceOptModule, CameraOptModule, set_random_seed

from .config import Config
from .evaluator import Evaluator
from .logger import TrainingLogger
from .model import create_splats_with_optimizers


class Runner:
    """Orchestrates model setup and the training loop.

    Setup (called once from ``__init__``):
        ``_setup_dirs``        — create the output directory tree
        ``_load_data``         — parse dataset, build train/val splits
        ``_build_model``       — initialise splat parameters and optimizers
        ``_build_aux_modules`` — optional pose/appearance/bilateral-grid modules
        ``_build_viewer``      — start interactive Viser viewer (if enabled)

    Delegates to:
        ``self.logger``    (TrainingLogger)  — all file/tensorboard output
        ``self.evaluator`` (Evaluator)       — metrics, video, compression

    Training:
        ``_build_schedulers`` — construct LR schedulers
        ``_train_step``       — one forward-backward-optimizer iteration
        ``train``             — main loop

    Viewer:
        ``_viewer_render_fn`` — interactive viewer callback
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        local_rank: int,
        world_rank: int,
        world_size: int,
        cfg: Config,
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        self._setup_dirs()
        self._load_data()
        self._build_model()
        self._build_aux_modules()
        self._build_viewer()

        self.logger = TrainingLogger(
            cfg=cfg,
            splats=self.splats,
            stats_dir=self.stats_dir,
            ckpt_dir=self.ckpt_dir,
            ply_dir=self.ply_dir,
            world_rank=world_rank,
            writer=self.writer,
        )
        self.evaluator = Evaluator(
            cfg=cfg,
            rasterize_fn=self._rasterize_splats,
            splats=self.splats,
            valset=self.valset,
            parser=self.parser,
            scene_scale=self.scene_scale,
            render_dir=self.render_dir,
            stats_dir=self.stats_dir,
            writer=self.writer,
            world_rank=world_rank,
            device=self.device,
            compression_method=self.compression_method,
            color_correct=getattr(self, "_color_correct", None),
        )

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _setup_dirs(self) -> None:
        """Create the output directory tree under ``cfg.result_dir``."""
        cfg = self.cfg
        for subdir in ("", "ckpts", "stats", "renders", "ply"):
            os.makedirs(os.path.join(cfg.result_dir, subdir), exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        self.stats_dir = f"{cfg.result_dir}/stats"
        self.render_dir = f"{cfg.result_dir}/renders"
        self.ply_dir = f"{cfg.result_dir}/ply"
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

    def _load_data(self) -> None:
        """Parse the COLMAP dataset and build train/val ``Dataset`` objects."""
        cfg = self.cfg
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

    def _build_model(self) -> None:
        """Initialise splat parameters, optimizers, and densification strategy."""
        cfg = self.cfg
        feature_dim = 32 if cfg.app_opt else None

        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=self.world_rank,
            world_size=self.world_size,
        )
        print("Model initialised. Number of GS:", len(self.splats["means"]))

        cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(cfg.strategy, DefaultStrategy):
            self.strategy_state = cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(cfg.strategy, MCMCStrategy):
            self.strategy_state = cfg.strategy.initialize_state()
        else:
            assert_never(cfg.strategy)

        self.compression_method = None
        if cfg.compression == "png":
            self.compression_method = PngCompression()
        elif cfg.compression is not None:
            raise ValueError(f"Unknown compression strategy: {cfg.compression}")

    def _build_aux_modules(self) -> None:
        """Set up optional pose adjustment, appearance, and bilateral-grid modules."""
        cfg = self.cfg

        # ── Pose optimisation ──────────────────────────────────────────────
        self.pose_optimizers: List[torch.optim.Optimizer] = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if self.world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if self.world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        # ── Appearance optimisation ────────────────────────────────────────
        self.app_optimizers: List[torch.optim.Optimizer] = []
        feature_dim = 32 if cfg.app_opt else None
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if self.world_size > 1:
                self.app_module = DDP(self.app_module)

        # ── Bilateral grid ─────────────────────────────────────────────────
        self.bil_grid_optimizers: List[torch.optim.Optimizer] = []
        if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
            if cfg.use_fused_bilagrid:
                cfg.use_bilateral_grid = True
                from fused_bilagrid import (  # type: ignore[import]
                    BilateralGrid,
                    color_correct,
                    slice as bil_slice,
                    total_variation_loss,
                )
            else:
                cfg.use_bilateral_grid = True
                from lib_bilagrid import (  # type: ignore[import]
                    BilateralGrid,
                    color_correct,
                    slice as bil_slice,
                    total_variation_loss,
                )
            self._color_correct = color_correct
            self._bil_slice = bil_slice
            self._total_variation_loss = total_variation_loss

            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                )
            ]

    def _build_viewer(self) -> None:
        """Start the interactive Viser viewer (skipped when disabled)."""
        if self.cfg.disable_viewer:
            return
        self.server = viser.ViserServer(port=self.cfg.port, verbose=False)
        self.viewer = GsplatViewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            output_dir=Path(self.cfg.result_dir),
            mode="training",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rasterization
    # ─────────────────────────────────────────────────────────────────────────

    def _rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Run Gaussian rasterization and return ``(colors, alphas, info)``."""
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = torch.sigmoid(colors + self.splats["colors"])
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def _build_schedulers(self, max_steps: int) -> List:
        """Construct and return the list of LR schedulers for this run."""
        cfg = self.cfg
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            )
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        return schedulers

    def _train_step(self, step: int, data: dict, schedulers: List) -> dict:
        """One full forward -> loss -> backward -> optimizer -> strategy iteration.

        Returns a dict consumed by ``train`` for logging and viewer updates:
            ``desc``       — progress-bar description string
            ``num_rays``   — pixel count (for viewer rays/sec metric)
            ``sh_degree``  — current SH degree (for PLY export)
            ``loss``       — total scalar loss
            ``l1loss``     — L1 component
            ``ssimloss``   — SSIM component
            ``depthloss``  — depth component (or None)
            ``tvloss``     — TV regularisation component (or None)
            ``pixels``     — ground-truth pixels tensor
            ``colors``     — rendered color tensor
        """
        cfg = self.cfg
        device = self.device

        camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        num_rays = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None

        points = depths_gt = None
        if cfg.depth_loss:
            points = data["points"].to(device)
            depths_gt = data["depths"].to(device)

        height, width = pixels.shape[1:3]

        if cfg.pose_noise:
            camtoworlds = self.pose_perturb(camtoworlds, image_ids)
        if cfg.pose_opt:
            camtoworlds = self.pose_adjust(camtoworlds, image_ids)

        # ── Forward ───────────────────────────────────────────────────────
        sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        renders, alphas, info = self._rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            masks=masks,
        )
        colors = renders[..., :3]
        depths = renders[..., 3:4] if renders.shape[-1] == 4 else None

        if cfg.use_bilateral_grid:
            grid_y, grid_x = torch.meshgrid(
                (torch.arange(height, device=device) + 0.5) / height,
                (torch.arange(width, device=device) + 0.5) / width,
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            colors = self._bil_slice(
                self.bil_grids,
                grid_xy.expand(colors.shape[0], -1, -1, -1),
                colors,
                image_ids.unsqueeze(-1),
            )["rgb"]

        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            colors = colors + bkgd * (1.0 - alphas)

        cfg.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )

        # ── Loss ──────────────────────────────────────────────────────────
        l1loss = F.l1_loss(colors, pixels)
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
        )
        loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

        depthloss = tvloss = None
        if cfg.depth_loss and points is not None and depths_gt is not None:
            pts = torch.stack(
                [
                    points[:, :, 0] / (width - 1) * 2 - 1,
                    points[:, :, 1] / (height - 1) * 2 - 1,
                ],
                dim=-1,
            )
            sampled = (
                F.grid_sample(
                    depths.permute(0, 3, 1, 2), pts.unsqueeze(2), align_corners=True
                )
                .squeeze(3)
                .squeeze(1)
            )
            disp = torch.where(sampled > 0.0, 1.0 / sampled, torch.zeros_like(sampled))
            depthloss = F.l1_loss(disp, 1.0 / depths_gt) * self.scene_scale
            loss = loss + depthloss * cfg.depth_lambda

        if cfg.use_bilateral_grid:
            tvloss = 10 * self._total_variation_loss(self.bil_grids.grids)
            loss = loss + tvloss

        if cfg.opacity_reg > 0.0:
            loss = (
                loss + cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            )
        if cfg.scale_reg > 0.0:
            loss = loss + cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

        loss.backward()

        # ── Progress-bar description ───────────────────────────────────────
        desc = f"loss={loss.item():.3f}| sh degree={sh_degree}| "
        if cfg.depth_loss and depthloss is not None:
            desc += f"depth loss={depthloss.item():.6f}| "
        if cfg.pose_opt and cfg.pose_noise:
            pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
            desc += f"pose err={pose_err.item():.6f}| "

        # ── Sparsify gradients (experimental) ─────────────────────────────
        if cfg.sparse_grad:
            assert cfg.packed, "Sparse gradients only work with packed mode."
            gaussian_ids = info["gaussian_ids"]
            for k in self.splats.keys():
                grad = self.splats[k].grad
                if grad is None or grad.is_sparse:
                    continue
                self.splats[k].grad = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],
                    values=grad[gaussian_ids],
                    size=self.splats[k].size(),
                    is_coalesced=len(Ks) == 1,
                )

        # ── Optimizer step ─────────────────────────────────────────────────
        visibility_mask = None
        if cfg.visible_adam:
            if cfg.packed:
                visibility_mask = torch.zeros_like(
                    self.splats["opacities"], dtype=torch.bool
                )
                visibility_mask.scatter_(0, info["gaussian_ids"], 1)
            else:
                visibility_mask = (info["radii"] > 0).all(-1).any(0)

        for optimizer in self.optimizers.values():
            optimizer.step(visibility_mask) if cfg.visible_adam else optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in (
            self.pose_optimizers + self.app_optimizers + self.bil_grid_optimizers
        ):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        # ── Strategy post-backward ─────────────────────────────────────────
        if isinstance(cfg.strategy, DefaultStrategy):
            cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )
        elif isinstance(cfg.strategy, MCMCStrategy):
            cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                lr=schedulers[0].get_last_lr()[0],
            )
        else:
            assert_never(cfg.strategy)

        return {
            "desc": desc,
            "num_rays": num_rays,
            "sh_degree": sh_degree,
            "loss": loss,
            "l1loss": l1loss,
            "ssimloss": ssimloss,
            "depthloss": depthloss,
            "tvloss": tvloss,
            "pixels": pixels,
            "colors": colors,
        }

    def train(self) -> None:
        """Main training loop."""
        cfg = self.cfg

        if self.world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        schedulers = self._build_schedulers(max_steps)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Precompute step sets for O(1) membership checks
        save_at = {i - 1 for i in cfg.save_steps}
        ply_at = {i - 1 for i in cfg.ply_steps}
        eval_at = {i - 1 for i in cfg.eval_steps}

        global_tic = time.time()
        pbar = tqdm.tqdm(range(max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            out = self._train_step(step, data, schedulers)
            pbar.set_description(out["desc"])

            # ── Logging ───────────────────────────────────────────────────
            if self.world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                self.logger.log_step(
                    step,
                    loss=out["loss"],
                    l1loss=out["l1loss"],
                    ssimloss=out["ssimloss"],
                    sh_degree=out["sh_degree"],
                    pixels=out["pixels"],
                    colors=out["colors"],
                    depthloss=out["depthloss"],
                    tvloss=out["tvloss"],
                )

            # ── Checkpointing ─────────────────────────────────────────────
            if step in save_at or step == max_steps - 1:
                self.logger.save_checkpoint(
                    step,
                    elapsed=time.time() - global_tic,
                    pose_adjust=getattr(self, "pose_adjust", None),
                    app_module=getattr(self, "app_module", None),
                    world_size=self.world_size,
                )
            if (step in ply_at or step == max_steps - 1) and cfg.save_ply:
                self.logger.export_ply(
                    step,
                    out["sh_degree"],
                    app_module=getattr(self, "app_module", None),
                )

            # ── Evaluation ────────────────────────────────────────────────
            if step in eval_at:
                self.evaluator.eval(step)
                self.evaluator.render_traj(step)
                if cfg.compression is not None:
                    self.evaluator.run_compression(step)

            # ── Viewer update ─────────────────────────────────────────────
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                steps_per_sec = 1.0 / max(time.time() - tic, 1e-10)
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    out["num_rays"] * steps_per_sec
                )
                self.viewer.update(step, out["num_rays"])

    # ─────────────────────────────────────────────────────────────────────────
    # Interactive viewer callback
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width, height = (
                render_tab_state.render_width,
                render_tab_state.render_height,
            )
        else:
            width, height = (
                render_tab_state.viewer_width,
                render_tab_state.viewer_height,
            )

        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K = (
            torch.from_numpy(camera_state.get_K((width, height)))
            .float()
            .to(self.device)
        )

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self._rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        mode = render_tab_state.render_mode
        if mode == "rgb":
            return render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()

        if mode in ("depth(accumulated)", "depth(expected)"):
            depth = render_colors[0, ..., :1]
            if render_tab_state.normalize_nearfar:
                near_p, far_p = render_tab_state.near_plane, render_tab_state.far_plane
            else:
                near_p, far_p = depth.min(), depth.max()
            depth_norm = torch.clip((depth - near_p) / (far_p - near_p + 1e-10), 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            return (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )

        if mode == "alpha":
            alpha = render_alphas[0, ..., :1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            return apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
