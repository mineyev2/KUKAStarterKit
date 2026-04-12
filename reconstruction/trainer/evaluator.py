"""Evaluator — image quality evaluation, trajectory rendering, and compression.

Receives a ``rasterize_fn`` callable (Runner._rasterize_splats) so it stays
fully decoupled from Runner internals.  Swap or subclass independently.
"""

import json
import os
import time

from collections import defaultdict
from typing import Callable, Optional

import imageio
import numpy as np
import torch
import tqdm

from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .config import Config


class Evaluator:
    """Computes image-quality metrics and writes evaluation artefacts.

    All heavy rendering goes through ``rasterize_fn`` — the bound method
    ``Runner._rasterize_splats`` — keeping Evaluator unaware of model internals.
    Pass ``splats`` by reference so ``run_compression`` can update it in-place.
    Pass ``color_correct`` (from lib_bilagrid / fused_bilagrid) when using the
    bilateral-grid option; leave it ``None`` otherwise.
    """

    def __init__(
        self,
        cfg: Config,
        rasterize_fn: Callable,
        splats: torch.nn.ParameterDict,
        valset: Dataset,
        parser: Parser,
        scene_scale: float,
        render_dir: str,
        stats_dir: str,
        writer: SummaryWriter,
        world_rank: int,
        device: str,
        compression_method=None,
        color_correct: Optional[Callable] = None,
    ) -> None:
        self.cfg = cfg
        self.rasterize_fn = rasterize_fn
        self.splats = splats
        self.valset = valset
        self.parser = parser
        self.scene_scale = scene_scale
        self.render_dir = render_dir
        self.stats_dir = stats_dir
        self.writer = writer
        self.world_rank = world_rank
        self.device = device
        self.compression_method = compression_method
        self.color_correct = color_correct

        self._build_metrics()

    # ─────────────────────────────────────────────────────────────────────────

    def _build_metrics(self) -> None:
        """Instantiate PSNR, SSIM, and LPIPS on the correct device."""
        cfg = self.cfg
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val") -> None:
        """Evaluate on the validation split; saves metrics + side-by-side renders."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0.0
        metrics = defaultdict(list)

        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_fn(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)
            colors = torch.clamp(colors, 0.0, 1.0)

            if self.world_rank == 0:
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    (canvas * 255).astype(np.uint8),
                )
                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid and self.color_correct is not None:
                    cc = self.color_correct(colors, pixels)
                    cc_p = cc.permute(0, 3, 1, 2)
                    metrics["cc_psnr"].append(self.psnr(cc_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_p, pixels_p))

        if self.world_rank == 0:
            ellipse_time /= len(valloader)
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {"ellipse_time": ellipse_time, "num_GS": len(self.splats["means"])}
            )

            base = (
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, "
                f"LPIPS: {stats['lpips']:.3f} "
            )
            extra = ""
            if cfg.use_bilateral_grid and "cc_psnr" in stats:
                extra = (
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, "
                    f"CC_LPIPS: {stats['cc_lpips']:.3f} "
                )
            print(
                base
                + extra
                + f"Time: {stats['ellipse_time']:.3f}s/image  GS: {stats['num_GS']}"
            )

            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int) -> None:
        """Render a fly-through video along the configured trajectory path."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds
        trim = self.cfg.render_traj_trim
        if trim is not None and len(camtoworlds_all) > 2 * trim:
            camtoworlds_all = camtoworlds_all[trim:-trim]

        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 1)
        elif cfg.render_traj_path == "ellipse":
            height_val = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height_val
            )
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(f"Unsupported render_traj_path: {cfg.render_traj_path}")

        bottom = np.repeat(
            np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
        )
        camtoworlds_all = (
            torch.from_numpy(np.concatenate([camtoworlds_all, bottom], axis=1))
            .float()
            .to(device)
        )

        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_fn(
                camtoworlds=camtoworlds_all[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )
            colors = torch.clamp(renders[..., :3], 0.0, 1.0)
            depths = renders[..., 3:4]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas = (
                torch.cat([colors, depths.repeat(1, 1, 1, 3)], dim=2)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            writer.append_data((canvas * 255).astype(np.uint8))
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int) -> None:
        """Compress splats in-place, then evaluate the decompressed result."""
        if self.compression_method is None:
            return
        print("Running compression...")
        cfg = self.cfg
        compress_dir = f"{cfg.result_dir}/compression/rank{self.world_rank}"
        os.makedirs(compress_dir, exist_ok=True)
        self.compression_method.compress(compress_dir, self.splats)
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")
