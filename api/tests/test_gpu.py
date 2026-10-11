"""GPU detection, device validation and CUDA params. No GPU needed: nvidia-smi and the probe are faked."""
import subprocess

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app import gpu
from app.modeling.registry import MODELS, catalog, device_params
from app.modeling.train import train
from app.routers import runs
from app.schemas import RunRequest

USER = {"id": 1, "name": "Admin", "email": None, "role": "lead"}

SMI_OUT = "NVIDIA GeForce RTX 5060 Ti, 16311, 580.95.05\n"


@pytest.fixture(autouse=True)
def fresh_detection():
    gpu._cached, gpu._probed = None, False
    yield
    gpu._cached, gpu._probed = None, False


def fake_smi(monkeypatch, out=SMI_OUT, exc=None):
    monkeypatch.setattr(gpu.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def run(*a, **k):
        if exc:
            raise exc
        return subprocess.CompletedProcess(a[0], 0, stdout=out, stderr="")

    monkeypatch.setattr(gpu.subprocess, "run", run)


def test_no_nvidia_smi_means_no_gpu(monkeypatch):
    monkeypatch.setattr(gpu.shutil, "which", lambda name: None)
    assert gpu.detect() is None
    assert gpu.gpu_available() is False
    assert runs.system()["gpu"] is None


def test_warm_up_detects_in_the_background_once(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu, "_nvidia_smi", lambda: calls.append(1))
    gpu.warm_up().join(timeout=10)
    assert gpu._probed and calls == [1]
    assert gpu.detect() is None and calls == [1]  # served from the cache


def test_failing_nvidia_smi_means_no_gpu(monkeypatch):
    fake_smi(monkeypatch, exc=subprocess.CalledProcessError(9, "nvidia-smi"))
    assert gpu.detect() is None
    fake_smi(monkeypatch, exc=subprocess.TimeoutExpired("nvidia-smi", 5))
    assert gpu.detect(refresh=True) is None


def test_detection_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(gpu, "_nvidia_smi", boom)
    assert gpu.detect() is None


def test_gpu_present_but_cpu_build(monkeypatch):
    fake_smi(monkeypatch)
    monkeypatch.setattr(gpu, "xgboost_cuda_build", lambda: False)
    info = gpu.detect()
    assert info["name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert info["memory_mb"] == 16311 and info["driver"] == "580.95.05" and info["count"] == 1
    assert info["available"] is False and "CPU build" in info["reason"]


def test_gpu_usable(monkeypatch):
    fake_smi(monkeypatch)
    monkeypatch.setattr(gpu, "_probe_xgboost", lambda: None)
    assert gpu.detect() == {"name": "NVIDIA GeForce RTX 5060 Ti", "memory_mb": 16311, "driver": "580.95.05",
                            "count": 1, "available": True, "reason": None}
    assert gpu.gpu_available()


def test_detection_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu, "_nvidia_smi", lambda: calls.append(1))
    gpu.detect(), gpu.detect()
    assert len(calls) == 1
    gpu.detect(refresh=True)
    assert len(calls) == 2


def test_probe_reports_silent_cpu_fallback(monkeypatch):
    monkeypatch.setattr(gpu, "xgboost_cuda_build", lambda: True)
    monkeypatch.setattr(gpu.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="cpu\n", stderr=""))
    assert "fell back to cpu" in gpu._probe_xgboost()
    monkeypatch.setattr(gpu.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="cuda:0\n", stderr=""))
    assert gpu._probe_xgboost() is None
    monkeypatch.setattr(gpu.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, stdout="", stderr="x\nCUDA error 35"))
    assert gpu._probe_xgboost().endswith("CUDA error 35")


def test_cpu_device_adds_no_params():
    for spec in MODELS.values():
        assert device_params(spec, "cpu") == {}
        assert "device" not in spec.default_params


def test_cuda_params_for_xgboost_only():
    assert device_params(MODELS["xgboost"], "cuda") == {"device": "cuda", "tree_method": "hist"}
    assert MODELS["xgboost"].devices == ["cpu", "cuda"]
    for key in ("lightgbm", "random_forest", "logistic_regression"):
        assert MODELS[key].devices == ["cpu"]
        with pytest.raises(ValueError, match="can't train on 'cuda'"):
            device_params(MODELS[key], "cuda")
    with pytest.raises(ValueError):
        device_params(MODELS["xgboost"], "tpu")
    assert {m["key"]: m["devices"] for m in catalog()}["xgboost"] == ["cpu", "cuda"]


def test_xgboost_cuda_estimator_params():
    spec = MODELS["xgboost"]
    est = spec.build({**spec.default_params, **device_params(spec, "cuda")}, 0)
    assert est.get_params()["device"] == "cuda" and est.get_params()["tree_method"] == "hist"


def test_xgboost_is_installed():
    assert MODELS["xgboost"].available


def test_run_request_device():
    assert RunRequest().device == "cpu"
    assert RunRequest(device="cuda").device == "cuda"
    with pytest.raises(ValidationError):
        RunRequest(device="gpu")


def test_train_rejects_unsupported_device_before_loading(tmp_path):
    with pytest.raises(ValueError, match="can't train on 'cuda'"):
        train(tmp_path / "missing", "lightgbm", device="cuda")


@pytest.fixture
def fake_export(monkeypatch):
    exp = {"dataset_id": 1, "model_family": "tree", "output_dir": "/nowhere"}
    manifest = {"targets": ["a"], "contains_nan": True}
    monkeypatch.setattr(runs, "_export_manifest", lambda export_id, user: (exp, manifest))


def test_start_run_cuda_without_gpu_is_409(monkeypatch, fake_export):
    monkeypatch.setattr(gpu.shutil, "which", lambda name: None)
    with pytest.raises(HTTPException) as e:
        runs.start_run(1, RunRequest(model="xgboost", device="cuda"), USER)
    assert e.value.status_code == 409 and "No GPU" in e.value.detail


def test_start_run_cuda_with_unusable_gpu_explains(monkeypatch, fake_export):
    fake_smi(monkeypatch)
    monkeypatch.setattr(gpu, "xgboost_cuda_build", lambda: False)
    with pytest.raises(HTTPException) as e:
        runs.start_run(1, RunRequest(model="xgboost", device="cuda"), USER)
    assert e.value.status_code == 409 and "CPU build" in e.value.detail


def test_start_run_cuda_for_cpu_only_model_is_422(fake_export):
    with pytest.raises(HTTPException) as e:
        runs.start_run(1, RunRequest(model="lightgbm", device="cuda"), USER)
    assert e.value.status_code == 422
