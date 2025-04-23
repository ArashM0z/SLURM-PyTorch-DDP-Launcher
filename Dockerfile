FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY README.md LICENSE ./
RUN pip install --upgrade pip setuptools wheel && pip install -e ".[dev]"
ENTRYPOINT ["slurm-ddp"]
