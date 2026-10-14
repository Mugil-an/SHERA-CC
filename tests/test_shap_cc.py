"""
tests/test_shap_cc.py
Unit and integration test suite for SHAP-CC / Shera-Real.
Verifies data generation, ML inference, SHAP explanations, and FastAPI endpoints.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, build_feature_vector, PodWorkloadRequest, NodeState, FEATURE_COLUMNS
from data.dataset_builder import build_dataset

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_feature_vector_builder():
    pod = PodWorkloadRequest(
        pod_name="test-pod",
        workload_type="cpu",
        cpu_req_cores=1.0,
        mem_req_mb=512.0,
        net_mbps=25.0,
        duration_sec=45.0,
    )
    node = NodeState(
        node_name="node-1",
        node_cpu_util_pct=40.0,
        node_mem_util_pct=30.0,
        node_active_pods=5,
        node_allocatable_cpu=12.0,
        node_allocatable_mem_mb=8192.0,
    )
    df = build_feature_vector(pod, node)
    assert len(df) == 1
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing column {col}"
    assert df["workload_type_cpu"].iloc[0] == 1
    assert df["post_alloc_cpu_pct"].iloc[0] > 40.0

def test_models_exist_and_load():
    assert os.path.exists("model/placement_model.joblib")
    assert os.path.exists("model/energy_model.joblib")
    assert os.path.exists("model/contention_model.joblib")
    assert os.path.exists("model/explainer.joblib")

    model = joblib.load("model/placement_model.joblib")
    explainer = joblib.load("model/explainer.joblib")
    assert model is not None
    assert explainer is not None

def test_inference_and_shap_consistency():
    model = joblib.load("model/placement_model.joblib")
    explainer = joblib.load("model/explainer.joblib")

    pod = PodWorkloadRequest(cpu_req_cores=2.0, mem_req_mb=1024.0)
    node = NodeState(node_name="worker", node_cpu_util_pct=50.0, node_mem_util_pct=40.0)
    X = build_feature_vector(pod, node)

    score = float(model.predict(X)[0])
    assert 0.0 <= score <= 100.0

    shap_res = explainer(X)
    shap_sum = float(np.sum(shap_res.values[0]))
    base_val = float(explainer.expected_value)
    # SHAP property: base_value + sum(shap_values) == model_output
    assert np.isclose(score, base_val + shap_sum, atol=1e-2)

def test_fastapi_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True

def test_fastapi_predict_score(client):
    payload = {
        "pod": {
            "pod_name": "eval-pod",
            "workload_type": "cpu",
            "cpu_req_cores": 1.0,
            "mem_req_mb": 512.0,
            "duration_sec": 30.0,
        },
        "candidate_nodes": [
            {
                "node_name": "worker-1",
                "node_cpu_util_pct": 75.0,
                "node_mem_util_pct": 50.0,
                "node_active_pods": 8,
            },
            {
                "node_name": "worker-2",
                "node_cpu_util_pct": 20.0,
                "node_mem_util_pct": 25.0,
                "node_active_pods": 2,
            },
        ],
    }
    res = client.post("/predict/score", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["selected_node"] == "worker-2"
    assert len(data["evaluations"]) == 2
    assert "human_explanation" in data["evaluations"][0]

def test_extender_prioritize(client):
    payload = {
        "NodeNames": ["kind-worker", "kind-worker2"],
        "Pod": {
            "metadata": {"name": "test-pod", "namespace": "workloads"},
            "spec": {
                "containers": [
                    {
                        "name": "c",
                        "resources": {"requests": {"cpu": "500m", "memory": "256Mi"}},
                    }
                ]
            },
        },
    }
    res = client.post("/scheduler/prioritize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 2
    assert all("Host" in item and "Score" in item for item in data)

def test_dataset_builder_generation():
    df = build_dataset(num_samples=100)
    assert len(df) == 100
    assert "placement_score" in df.columns
    assert "energy_proxy_joules" in df.columns
    assert "contention_risk" in df.columns
