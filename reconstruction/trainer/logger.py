"""TrainingLogger — checkpoint saving, TensorBoard logging, and PLY export.

Completely decoupled from training logic.  ``splats`` is a live reference to
the model's ``ParameterDict``, so the logger always sees the current state
without needing to be told about it.
"""

import json
import time

from typing import Optional

import numpy as np
import torch

from gsplat import export_splats
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from utils import rgb_to_sh

from .config import Config


class TrainingLogger:
    """Handles all output concerns: TensorBoard scalars/images, checkpoints, and PLY export.

    Designed to be independently replaceable — swap it out or subclass it
    without touching Runner or Evaluator.
    """

    def __init__(
        self,
        cfg: Config,
        splats: torch.nn.ParameterDict,
        stats_dir: str,
        ckpt_dir: str,
        ply_dir: str,
        world_rank: int,
        writer: SummaryWriter,
    ) -> None:
        self.cfg = cfg
        self.splats = splats
        self.stats_dir = stats_dir
        self.ckpt_dir = ckpt_dir
        self.ply_dir = ply_dir
        self.world_rank = world_rank
        self.writer = writer

    # ─────────────────────────────────────────────────────────────────────────

    def log_step(
        self,
        step: int,
        *,
        loss: Tensor,
        l1loss: Tensor,
        ssimloss: Tensor,
        sh_degree: int,
        pixels: Optional[Tensor] = None,
        colors: Optional[Tensor] = None,
        depthloss: Optional[Tensor] = None,
        tvloss: Optional[Tensor] = None,
    ) -> None:
        """Write training scalars (and optionally a side-by-side image) to TensorBoard."""
        mem = torch.cuda.max_memory_allocated() / 1024**3
        self.writer.add_scalar("train/loss", loss.item(), step)
        self.writer.add_scalar("train/l1loss", l1loss.item(), step)
        self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
        self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
        self.writer.add_scalar("train/mem", mem, step)
        if depthloss is not None:
            self.writer.add_scalar("train/depthloss", depthloss.item(), step)
        if tvloss is not None:
            self.writer.add_scalar("train/tvloss", tvloss.item(), step)
        if self.cfg.tb_save_image and pixels is not None and colors is not None:
            canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            canvas = canvas.reshape(-1, *canvas.shape[2:])
            self.writer.add_image("train/render", canvas, step)
        self.writer.flush()

    def save_checkpoint(
        self,
        step: int,
        elapsed: float,
        *,
        pose_adjust=None,
        app_module=None,
        world_size: int = 1,
    ) -> None:
        """Serialise splats (and optional modules) to a ``.pt`` checkpoint."""
        mem = torch.cuda.max_memory_allocated() / 1024**3
        stats = {
            "mem": mem,
            "ellipse_time": elapsed,
            "num_GS": len(self.splats["means"]),
        }
        print("Step:", step, stats)
        with open(
            f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json", "w"
        ) as f:
            json.dump(stats, f)

        ckpt = {"step": step, "splats": self.splats.state_dict()}
        if pose_adjust is not None:
            ckpt["pose_adjust"] = (
                pose_adjust.module.state_dict()
                if world_size > 1
                else pose_adjust.state_dict()
            )
        if app_module is not None:
            ckpt["app_module"] = (
                app_module.module.state_dict()
                if world_size > 1
                else app_module.state_dict()
            )
        torch.save(ckpt, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

    @torch.no_grad()
    def export_ply(self, step: int, sh_degree: int, *, app_module=None) -> None:
        """Export a ``.ply`` point-cloud snapshot of the current splats."""
        cfg = self.cfg
        if cfg.app_opt and app_module is not None:
            rgb = app_module(
                features=self.splats["features"],
                embed_ids=None,
                dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                sh_degree=sh_degree,
            )
            rgb = torch.sigmoid(rgb + self.splats["colors"]).squeeze(0).unsqueeze(1)
            sh0 = rgb_to_sh(rgb)
            shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
        else:
            sh0 = self.splats["sh0"]
            shN = self.splats["shN"]

        export_splats(
            means=self.splats["means"],
            scales=self.splats["scales"],
            quats=self.splats["quats"],
            opacities=self.splats["opacities"],
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
        )
