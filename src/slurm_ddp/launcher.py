"""DDP launcher: typed config -> validated sbatch script."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class DDPJob(BaseModel):
    """A multi-node DDP job request.

    Renders to an sbatch script that uses `torchrun` with c10d rendezvous
    so every node's local-rank-0 worker connects to the same `MASTER_ADDR`
    derived from the SLURM nodelist.
    """

    job_name: str = "ddp-train"
    script: str  # path to the training script (e.g. train.py)
    args: list[str] = Field(default_factory=list)
    nodes: int = Field(ge=1)
    gpus_per_node: int = Field(ge=1)
    cpus_per_task: int = Field(default=8, ge=1)
    memory: str = "64G"
    time: str = "24:00:00"
    partition: str = "gpu"
    account: str | None = None
    qos: str | None = None
    output_dir: str = "logs"
    venv: str = "$HOME/envs/ddp"
    modules: list[str] = Field(
        default_factory=lambda: ["python/3.11", "cuda/12.2", "nccl/2.18"]
    )
    env: dict[str, str] = Field(default_factory=dict)
    rdzv_port: int = Field(default=29500, ge=1024, le=65535)
    nccl_socket_ifname: str | None = "ib0"

    @field_validator("time")
    @classmethod
    def _check_time(cls, v: str) -> str:
        # SLURM time format: D-HH:MM:SS or HH:MM:SS or MM:SS
        if not v or v.count(":") not in (1, 2):
            raise ValueError("time must look like HH:MM:SS or D-HH:MM:SS")
        return v


def _format_env(env: dict[str, str]) -> str:
    return "\n".join(f"export {k}={v}" for k, v in sorted(env.items()))


def render(job: DDPJob) -> str:
    sbatch_lines: list[str] = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job.job_name}",
        f"#SBATCH --nodes={job.nodes}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gpus-per-node={job.gpus_per_node}",
        f"#SBATCH --cpus-per-task={job.cpus_per_task}",
        f"#SBATCH --mem={job.memory}",
        f"#SBATCH --time={job.time}",
        f"#SBATCH --partition={job.partition}",
        f"#SBATCH --output={job.output_dir}/%x-%j.out",
        f"#SBATCH --error={job.output_dir}/%x-%j.err",
    ]
    if job.account:
        sbatch_lines.insert(2, f"#SBATCH --account={job.account}")
    if job.qos:
        sbatch_lines.insert(2, f"#SBATCH --qos={job.qos}")

    body = [
        "",
        "set -euo pipefail",
        f"mkdir -p {job.output_dir}",
        "",
        *[f"module load {m}" for m in job.modules],
        f"source {job.venv}/bin/activate",
        "",
        'export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)',
        f"export MASTER_PORT={job.rdzv_port}",
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
        "export NCCL_DEBUG=INFO",
        "export NCCL_IB_DISABLE=0",
    ]
    if job.nccl_socket_ifname:
        body.append(f"export NCCL_SOCKET_IFNAME={job.nccl_socket_ifname}")
    if job.env:
        body += ["", _format_env(job.env)]

    body += [
        "",
        "srun \\",
        '    --output=logs/srun-%t.out --error=logs/srun-%t.err \\',
        "    torchrun \\",
        '        --nnodes=$SLURM_JOB_NUM_NODES \\',
        '        --nproc_per_node=$SLURM_GPUS_PER_NODE \\',
        '        --rdzv_id=$SLURM_JOB_ID \\',
        '        --rdzv_backend=c10d \\',
        '        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \\',
        f"        {job.script} {' '.join(job.args)}",
    ]
    return "\n".join(sbatch_lines + body) + "\n"


def from_yaml(text: str) -> DDPJob:
    """Parse a YAML configuration into a validated ``DDPJob``."""
    import yaml
    raw: dict[str, Any] = yaml.safe_load(text) or {}
    return DDPJob.model_validate(raw)
