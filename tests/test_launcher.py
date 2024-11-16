import pytest

from slurm_ddp.launcher import DDPJob, from_yaml, render


def _job(**overrides):
    base = dict(
        job_name="t", script="train.py", nodes=2, gpus_per_node=4,
        time="01:00:00",
    )
    base.update(overrides)
    return DDPJob(**base)


def test_render_contains_required_sbatch_directives():
    body = render(_job())
    for d in ("--job-name=t", "--nodes=2", "--gpus-per-node=4",
              "--time=01:00:00", "--cpus-per-task=8"):
        assert d in body, f"missing {d}"


def test_render_uses_torchrun_with_c10d_rendezvous():
    body = render(_job())
    assert "torchrun" in body
    assert "--rdzv_backend=c10d" in body
    assert "--rdzv_endpoint=" in body


def test_optional_account_is_emitted_when_set():
    body = render(_job(account="rrg-mining"))
    assert "--account=rrg-mining" in body


def test_env_block_emitted_in_sorted_order():
    body = render(_job(env={"WANDB_PROJECT": "x", "AAA": "1"}))
    aaa = body.index("export AAA=1")
    wandb = body.index("export WANDB_PROJECT=x")
    assert aaa < wandb


def test_invalid_time_format_raises():
    with pytest.raises(Exception):
        DDPJob(script="t.py", nodes=1, gpus_per_node=1, time="forever")


def test_invalid_nodes_value_raises():
    with pytest.raises(Exception):
        DDPJob(script="t.py", nodes=0, gpus_per_node=1, time="01:00:00")


def test_from_yaml_parses_full_config():
    yaml = """
script: train.py
nodes: 2
gpus_per_node: 2
time: 02:00:00
env: { FOO: bar }
"""
    job = from_yaml(yaml)
    assert job.nodes == 2
    assert job.env["FOO"] == "bar"
