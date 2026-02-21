# SLURM PyTorch DDP Launcher

Generates a battle-tested sbatch script that runs `torchrun` with c10d
rendezvous for multi-node DistributedDataParallel training. The Python
config is a Pydantic model so misconfiguration fails at parse time, not
in a wedged job after queueing for an hour.

## What's in the box

- `DDPJob` Pydantic model with bounds checks on `nodes`, `gpus_per_node`,
  and SLURM time format.
- `render(job)` emits the sbatch script with:
  - SLURM-aware `MASTER_ADDR` discovery via `scontrol show hostnames`,
  - `torchrun --rdzv_backend=c10d` so any node can serve as the master,
  - `NCCL_DEBUG=INFO` + `NCCL_IB_DISABLE=0` + configurable
    `NCCL_SOCKET_IFNAME` (default `ib0` for InfiniBand clusters),
  - `set -euo pipefail` and a `logs/` mkdir so the job fails loudly.
- `slurm-ddp --config configs/example.yaml` writes a ready-to-submit
  `ddp.sbatch` file.

## Quickstart

```bash
pip install -e ".[dev]"
slurm-ddp --config configs/example.yaml --out ddp.sbatch
sbatch ddp.sbatch
```

Example YAML config — see [configs/example.yaml](configs/example.yaml):

```yaml
job_name: sed2am-ddp
script: train.py
args: ["--epochs=50", "--batch-size=256"]
nodes: 4
gpus_per_node: 4
cpus_per_task: 16
memory: 128G
time: "24:00:00"
partition: gpu-v100
venv: $HOME/envs/drl
env:
  WANDB_PROJECT: drl-vrp
```

## Layout

```
src/slurm_ddp/
├── launcher.py  # DDPJob + render + from_yaml
└── cli.py       # slurm-ddp entrypoint
configs/         # example.yaml
tests/           # render directives, torchrun args, validation errors
```
