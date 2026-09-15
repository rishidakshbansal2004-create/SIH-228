from mirad.security.attack_lab import run_attack_lab
from mirad.security.cli import run_demo


def test_offline_demo_runs():
    result = run_demo()
    assert result["status"] == "ok"
    assert result["artifact_verification"]["valid"] is True
    assert result["replay_check"]["valid"] is False


def test_attack_lab_executes():
    scenarios = run_attack_lab()
    assert len(scenarios) >= 30
    assert all(item["passed"] for item in scenarios)
