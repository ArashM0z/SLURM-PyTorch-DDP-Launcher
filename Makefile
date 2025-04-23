.PHONY: install dev lint test render docker clean
install: ; pip install -e .
dev: ; pip install -e ".[dev]"
lint: ; ruff check src tests
test: ; pytest --cov=slurm_ddp --cov-report=term-missing
render: ; slurm-ddp --config configs/example.yaml --out ddp.sbatch
docker: ; docker build -t slurm-ddp:latest .
clean: ; rm -rf build dist *.egg-info .pytest_cache .ruff_cache ddp.sbatch
