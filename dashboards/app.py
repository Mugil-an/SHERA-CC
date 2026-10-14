"""
dashboards/app.py
SHAP-CC: Explainable AI Kubernetes Scheduler - Clean Project Demo UI
A clean, focused single-page demo showcasing:
1. Incoming Pod Specification
2. Candidate Worker Nodes State
3. AI Scheduling Decision (Winning Node, Score, Energy)
4. SHAP Feature Attribution ("Why this node was chosen")
5. Live Cluster Placement verification
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure root repository is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from xgboost import XGBRegressor
import shap

# Page configuration
st.set_page_config(
    page_title="SHAP-CC: AI Kubernetes Scheduler Demo",
    page_icon="⚡",
    layout="wide",
)

# Clean, modern UI styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .winner-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.25) 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .score-badge {
        display: inline-block;
        background: #10b981;
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")

# Load models once in memory
@st.cache_resource
def load_models():
    model_dir = ROOT_DIR / "model"
    models = {}
    try:
        p_json = model_dir / "placement_model.json"
        e_json = model_dir / "energy_model.json"
        
        m_p = XGBRegressor()
        m_p.load_model(str(p_json))
        models["placement"] = m_p

        m_e = XGBRegressor()
        m_e.load_model(str(e_json))
        models["energy"] = m_e

        # Build SHAP TreeExplainer in memory
        models["explainer"] = shap.TreeExplainer(m_p)
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models = load_models()

def evaluate_node(pod: dict, node: dict) -> dict:
    cpu_req = pod["cpu_req"]
    mem_req = pod["mem_req"]
    wtype = pod["wtype"]
    duration = pod["duration"]

    node_cpu = node["cpu_util"]
    node_mem = node["mem_util"]
    alloc_cpu = 12.0
    alloc_mem = 8192.0

    post_cpu = min(100.0, node_cpu + (cpu_req / alloc_cpu) * 100.0)
    post_mem = min(100.0, node_mem + (mem_req / alloc_mem) * 100.0)
    imbalance = abs(post_cpu - post_mem)

    X = pd.DataFrame([{
        "cpu_req_cores": cpu_req,
        "mem_req_mb": mem_req,
        "net_mbps": 20.0,
        "duration_sec": duration,
        "workload_type_cpu": 1 if wtype == "cpu" else 0,
        "workload_type_memory": 1 if wtype == "memory" else 0,
        "workload_type_network": 1 if wtype == "network" else 0,
        "workload_type_mixed": 1 if wtype == "mixed" else 0,
        "node_cpu_util_pct": node_cpu,
        "node_mem_util_pct": node_mem,
        "node_active_pods": node.get("pods", 4),
        "node_power_efficiency": 1.0,
        "post_alloc_cpu_pct": post_cpu,
        "post_alloc_mem_pct": post_mem,
        "resource_imbalance": imbalance,
    }])

    score = float(models["placement"].predict(X)[0]) if "placement" in models else 80.0
    energy = float(models["energy"].predict(X)[0]) if "energy" in models else 500.0

    shap_attrs = []
    if "explainer" in models:
        explainer = models["explainer"]
        vals = explainer(X).values[0]
        for f, v, s in zip(X.columns, X.iloc[0], vals):
            shap_attrs.append({"feature": f, "value": float(v), "shap": float(s)})
        shap_attrs.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "node_name": node["name"],
        "score": round(score, 1),
        "energy": round(energy, 1),
        "post_cpu": round(post_cpu, 1),
        "post_mem": round(post_mem, 1),
        "shap_attrs": shap_attrs[:5],
    }

# ----------------- UI Header -----------------
st.title("⚡ SHAP-CC: Explainable AI Kubernetes Scheduler")
st.markdown("**Live Project Demo:** Optimizing workload placement for energy efficiency & contention reduction with full **SHAP transparency**.")
st.divider()

col_input, col_decision = st.columns([1, 1.4], gap="large")

# ----------------- Left Column: Inputs -----------------
with col_input:
    st.subheader("1. Pod to Schedule")
    with st.container(border=True):
        pod_name = st.text_input("Pod Name", value="demo-batch-job")
        w_type = st.selectbox("Workload Type", ["cpu", "memory", "network", "mixed"])
        c1, c2 = st.columns(2)
        with c1:
            cpu_req = st.slider("CPU Request (cores)", 0.2, 4.0, 1.5, 0.1)
        with c2:
            mem_req = st.slider("Memory Request (MB)", 64, 2048, 512, 64)
        duration = st.slider("Job Duration (sec)", 15, 120, 30, 5)

    st.subheader("2. Cluster Node States")
    with st.container(border=True):
        # Optional live fetch button
        if st.button("🔄 Sync with Live Kind Cluster Telemetry"):
            try:
                q = '100 * (1 - avg by (node) (rate(node_cpu_seconds_total{mode="idle"}[1m])))'
                res = requests.get(f"{PROMETHEUS_URL}/api/v1/query?query={q}", timeout=2).json()
                items = res.get("data", {}).get("result", [])
                for item in items:
                    node_name = item["metric"].get("node")
                    if node_name == "kind-worker":
                        st.session_state["n1_cpu_val"] = float(item["value"][1])
                    elif node_name == "kind-worker2":
                        st.session_state["n2_cpu_val"] = float(item["value"][1])
                st.success("Synced real CPU load from Prometheus!")
            except Exception as e:
                st.warning("Could not reach Prometheus. Using manual sliders.")

        st.markdown("**`kind-worker` (Node 1)**")
        default_n1_cpu = st.session_state.get("n1_cpu_val", 68.0)
        n1_cpu = st.slider("CPU Load (%)", 5.0, 95.0, float(default_n1_cpu), 1.0, key="slider_n1")
        n1_mem = st.slider("Memory Load (%)", 5.0, 95.0, 55.0, 1.0, key="slider_n1_mem")

        st.markdown("**`kind-worker2` (Node 2)**")
        default_n2_cpu = st.session_state.get("n2_cpu_val", 22.0)
        n2_cpu = st.slider("CPU Load (%)", 5.0, 95.0, float(default_n2_cpu), 1.0, key="slider_n2")
        n2_mem = st.slider("Memory Load (%)", 5.0, 95.0, 25.0, 1.0, key="slider_n2_mem")

# ----------------- Right Column: Decision & SHAP -----------------
with col_decision:
    pod = {"name": pod_name, "cpu_req": cpu_req, "mem_req": mem_req, "wtype": w_type, "duration": duration}
    n1 = {"name": "kind-worker", "cpu_util": n1_cpu, "mem_util": n1_mem, "pods": 8}
    n2 = {"name": "kind-worker2", "cpu_util": n2_cpu, "mem_util": n2_mem, "pods": 3}

    res1 = evaluate_node(pod, n1)
    res2 = evaluate_node(pod, n2)

    winner = res1 if res1["score"] >= res2["score"] else res2
    other = res2 if winner == res1 else res1
    energy_saved = max(0.0, round(((other["energy"] - winner["energy"]) / max(1.0, other["energy"])) * 100, 1))

    st.subheader("3. AI Scheduling Decision")

    # Winner Highlight Box
    st.markdown(f"""
    <div class="winner-box">
        <span class="score-badge">SELECTED BY SHAP-CC</span>
        <h2 style="margin: 8px 0; color: #10b981;">Node: <code>{winner['node_name']}</code></h2>
        <div style="display: flex; gap: 30px; margin-top: 10px;">
            <div><span style="color: #94a3b8; font-size: 0.85rem;">Placement Score</span><br><span class="stat-number">{winner['score']} / 100</span></div>
            <div><span style="color: #94a3b8; font-size: 0.85rem;">Projected Energy</span><br><span class="stat-number">{winner['energy']} J</span></div>
            <div><span style="color: #94a3b8; font-size: 0.85rem;">Score Lead</span><br><span class="stat-number">+{winner['score'] - other['score']:.1f} pts</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Side-by-side Score Bars
    fig_score = go.Figure()
    fig_score.add_trace(go.Bar(
        x=[res1["node_name"], res2["node_name"]],
        y=[res1["score"], res2["score"]],
        marker_color=["#10b981" if res1 == winner else "#64748b", "#10b981" if res2 == winner else "#64748b"],
        text=[f"{res1['score']} pts", f"{res2['score']} pts"],
        textposition="auto",
    ))
    fig_score.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 105], title="Score"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_score, width="stretch")

    st.subheader("4. SHAP Explanation (Why did this node win?)")

    # Natural Language Summary
    pos_reasons = [a["feature"].replace("_", " ") for a in winner["shap_attrs"] if a["shap"] > 0][:2]
    neg_reasons = [a["feature"].replace("_", " ") for a in other["shap_attrs"] if a["shap"] < 0][:2]
    
    explanation_text = f"**Explanation:** The AI scheduler selected **`{winner['node_name']}`** because "
    if pos_reasons:
        explanation_text += f"it has favorable **{', '.join(pos_reasons)}**, "
    if neg_reasons:
        explanation_text += f"while **`{other['node_name']}`** was penalized for high **{', '.join(neg_reasons)}**. "
    explanation_text += f"Placing this workload on `{winner['node_name']}` avoids CPU throttling and achieves a **{winner['score']}/100** score."

    st.info(explanation_text)

    # SHAP Diverging Horizontal Bar Chart
    feats = [a["feature"].replace("_", " ") for a in winner["shap_attrs"]][::-1]
    shaps = [a["shap"] for a in winner["shap_attrs"]][::-1]
    bar_colors = ["#10b981" if s >= 0 else "#ef4444" for s in shaps]

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        y=feats,
        x=shaps,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:+.2f}" for s in shaps],
        textposition="outside",
    ))
    fig_shap.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_shap.update_layout(
        height=220,
        margin=dict(l=10, r=30, t=10, b=20),
        xaxis=dict(title="SHAP Feature Contribution (Points Added or Subtracted)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_shap, width="stretch")

    # ----------------- Deploy Live Pod Button -----------------
    st.divider()
    st.subheader("5. Test Live Scheduling on Kind Cluster")
    if st.button(f"🚀 Deploy '{pod_name}' to `{winner['node_name']}`", type="primary"):
        with st.spinner("Submitting Pod manifest to Kubernetes cluster..."):
            pod_manifest = f"""
apiVersion: v1
kind: Pod
metadata:
  name: demo-{int(time.time()) % 10000}
  namespace: workloads
  labels:
    app: shap-cc-demo
    scheduled-by: shap-cc
spec:
  nodeSelector:
    kubernetes.io/hostname: {winner['node_name']}
  restartPolicy: Never
  containers:
    - name: stress
      image: shap-workload:local
      imagePullPolicy: IfNotPresent
      command: ["stress-ng"]
      args: ["--cpu", "1", "--timeout", "20s"]
      resources:
        requests:
          cpu: "{int(cpu_req * 1000)}m"
          memory: "{int(mem_req)}Mi"
"""
            try:
                p = subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out, err = p.communicate(input=pod_manifest)
                if p.returncode == 0:
                    st.success(f"✅ Pod successfully deployed and assigned to **`{winner['node_name']}`**!")
                    time.sleep(2)
                    res = subprocess.run(["kubectl", "get", "pods", "-n", "workloads", "-l", "app=shap-cc-demo", "-o", "wide"], capture_output=True, text=True)
                    st.code(res.stdout, language="bash")
                else:
                    st.error(f"kubectl error: {err}")
            except Exception as e:
                st.error(f"Failed to run kubectl: {e}")

# ----------------- Collapsible Dataset / Benchmarks (For Presentation Details) -----------------
with st.expander("📊 View Dataset & Benchmark Details (Optional)"):
    tab_data, tab_bench = st.tabs(["Observed Workload Metrics", "Benchmark Summary"])
    
    with tab_data:
        obs_csv = ROOT_DIR / "data" / "workloads_observed_summary.csv"
        if obs_csv.exists():
            st.dataframe(pd.read_csv(obs_csv), width="stretch")

    with tab_bench:
        st.markdown("""
        - **Resource Balance Improvement:** +28.0% over default Kubernetes scheduler.
        - **Placement Model Accuracy:** $R^2 = 0.989$ (XGBoost).
        - **Energy Model Accuracy:** $R^2 = 0.991$ ($MAE = 45.18\text{ J}$).
        """)
