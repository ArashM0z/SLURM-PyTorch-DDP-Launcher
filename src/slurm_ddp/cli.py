"""CLI: render an sbatch from a YAML config."""
from __future__ import annotations

import argparse
from pathlib import Path

from slurm_ddp.launcher import from_yaml, render


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("ddp.sbatch"))
    args = p.parse_args(argv)
    job = from_yaml(args.config.read_text())
    args.out.write_text(render(job))
    print(f"wrote {args.out}")
    return 0
