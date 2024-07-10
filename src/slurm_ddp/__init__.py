"""Generate SLURM sbatch scripts that launch torchrun multi-node DDP."""

from slurm_ddp.launcher import DDPJob, render

__all__ = ["DDPJob", "render"]
__version__ = "0.3.0"
