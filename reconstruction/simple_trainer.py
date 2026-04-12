"""simple_trainer.py — Hydra entry point for 3D Gaussian Splatting training.

All training logic lives in the ``trainer/`` package:
  trainer/config.py  — Config dataclass
  trainer/model.py   — Gaussian splat initialisation
  trainer/runner.py  — Runner training / evaluation engine

Configuration is loaded from configs/config.yaml (Hydra).

Usage::

    # Default run (DefaultStrategy)
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py \\
        data_dir=data/my_scan result_dir=results/my_scan

    # MCMC preset
    python simple_trainer.py +experiment=mcmc data_dir=data/my_scan

    # Distributed training on 4 GPUs (4x fewer steps)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py steps_scaler=0.25

    # Eval-only from a checkpoint
    python simple_trainer.py ckpt='["results/my_scan/ckpts/ckpt_29999_rank0.pt"]'
"""

import time

import hydra
import torch

from gsplat.distributed import cli
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from trainer import Config, Runner


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config) -> None:
    """Distributed entry point called by gsplat's ``cli`` helper."""
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # Eval-only: load checkpoint(s) and run evaluation
        ckpts = [
            torch.load(f, map_location=runner.device, weights_only=True)
            for f in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1_000_000)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def train(cfg_raw: DictConfig) -> None:
    """Hydra entry point: parse config, validate deps, launch distributed run."""
    # Build the typed Config dataclass from the Hydra DictConfig
    strategy = instantiate(cfg_raw.strategy)
    cfg_dict = OmegaConf.to_container(cfg_raw, resolve=True)
    cfg_dict.pop("strategy")
    cfg_dict["bilateral_grid_shape"] = tuple(cfg_dict["bilateral_grid_shape"])
    cfg = Config(**cfg_dict, strategy=strategy)
    cfg.adjust_steps(cfg.steps_scaler)

    # Validate optional dependencies upfront for a clear error message
    if cfg.compression == "png":
        try:
            import plas  # noqa: F401
            import torchpq  # noqa: F401
        except ImportError:
            raise ImportError(
                "PNG compression requires torchpq and plas. Install via:\n"
                "  pip install torchpq  # see https://github.com/DeMoriarty/TorchPQ\n"
                "  pip install git+https://github.com/fraunhoferhhi/PLAS.git"
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d=true`."

    cli(main, cfg, verbose=True)


if __name__ == "__main__":
    train()
