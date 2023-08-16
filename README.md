# SLURM PyTorch DDP Launcher

A small Python launcher that auto-discovers SLURM environment variables and wraps a torchrun invocation correctly for multi-node DDP. Handles MASTER_ADDR discovery, rendezvous, and graceful shutdown on SIGTERM.
