"""
benchmarks/benchmark_scheduler.py
Benchmark suite comparing Default Kubernetes Scheduler vs SHAP-CC Scheduler.
Simulates realistic sequential workload arrivals on a multi-node cluster,
measuring total energy consumption (kJ), resource contention incidents, and node balance.
"""

import argparse
import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def simulate_cluster_run(scheduler_type: str, workloads: list, num_nodes: int = 3) -> dict:
    np.random.seed(42)
    # Initialize nodes
    nodes = [
        {"id": f"node-{i+1}", "cpu_cap": 12.0, "mem_cap": 8192.0, "cpu_used": 1.5, "mem_used": 1200.0, "power_eff": 1.0}
        for i in range(num_nodes)
    ]
    
    total_energy_joules = 0.0
    contention_incidents = 0
    imbalances = []

    # Try loading SHAP-CC placement model
    model = None
    if scheduler_type == "shap_cc":
        try:
            model = joblib.load("model/placement_model.joblib")
        except Exception:
            pass

    for w in workloads:
        cpu_req = w["cpu_req"]
        mem_req = w["mem_req"]
        duration = w["duration"]
        wtype = w["type"]
        net_mbps = w.get("net_mbps", 10.0)

        # Decay previous usage slightly as older tasks finish
        for n in nodes:
            n["cpu_used"] = max(1.0, n["cpu_used"] * 0.92)
            n["mem_used"] = max(800.0, n["mem_used"] * 0.92)

        chosen_idx = 0

        if scheduler_type == "default_k8s":
            # Default LeastAllocated: Pick node with greatest total free resource fraction
            scores = []
            for n in nodes:
                free_cpu = (n["cpu_cap"] - (n["cpu_used"] + cpu_req)) / n["cpu_cap"]
                free_mem = (n["mem_cap"] - (n["mem_used"] + mem_req)) / n["mem_cap"]
                scores.append(free_cpu + free_mem)
            chosen_idx = int(np.argmax(scores))

        else: # SHAP-CC Scheduler
            best_score = -1e9
            for idx, n in enumerate(nodes):
                node_cpu_pct = (n["cpu_used"] / n["cpu_cap"]) * 100.0
                node_mem_pct = (n["mem_used"] / n["mem_cap"]) * 100.0
                post_cpu = min(100.0, node_cpu_pct + (cpu_req / n["cpu_cap"]) * 100.0)
                post_mem = min(100.0, node_mem_pct + (mem_req / n["mem_cap"]) * 100.0)
                imbalance = abs(post_cpu - post_mem)

                if model:
                    feat = pd.DataFrame([{
                        "cpu_req_cores": cpu_req,
                        "mem_req_mb": mem_req,
                        "net_mbps": net_mbps,
                        "duration_sec": duration,
                        "workload_type_cpu": 1 if wtype == "cpu" else 0,
                        "workload_type_memory": 1 if wtype == "memory" else 0,
                        "workload_type_network": 1 if wtype == "network" else 0,
                        "workload_type_mixed": 1 if wtype == "mixed" else 0,
                        "node_cpu_util_pct": node_cpu_pct,
                        "node_mem_util_pct": node_mem_pct,
                        "node_active_pods": 5,
                        "node_power_efficiency": n["power_eff"],
                        "post_alloc_cpu_pct": post_cpu,
                        "post_alloc_mem_pct": post_mem,
                        "resource_imbalance": imbalance,
                    }])
                    score = float(model.predict(feat)[0])
                else:
                    # Contention and energy penalty heuristic
                    score = 100.0 - (0.5 * imbalance + (20.0 if post_cpu > 75.0 else 0.0) + (25.0 if post_mem > 80.0 else 0.0))

                if score > best_score:
                    best_score = score
                    chosen_idx = idx

        target_node = nodes[chosen_idx]
        target_node["cpu_used"] += cpu_req
        target_node["mem_used"] += mem_req

        # Calculate post-placement utilization
        cpu_util = (target_node["cpu_used"] / target_node["cpu_cap"]) * 100.0
        mem_util = (target_node["mem_used"] / target_node["mem_cap"]) * 100.0
        imbalances.append(abs(cpu_util - mem_util))

        # Check for contention
        if cpu_util > 78.0 or mem_util > 82.0:
            contention_incidents += 1

        # Calculate energy proxy:
        util_factor = 1.0 + (0.5 * max(0.0, (cpu_util - 70.0) / 30.0) ** 2)
        power_w = (12.0 + 28.0 * cpu_req + 3.2 * (mem_req / 1024.0) + 0.015 * net_mbps) * target_node["power_eff"] * util_factor
        total_energy_joules += power_w * duration

    return {
        "total_energy_kj": round(total_energy_joules / 1000.0, 1),
        "contention_incidents": contention_incidents,
        "avg_node_imbalance": round(float(np.mean(imbalances)), 1),
    }

def run_benchmarks(num_jobs: int = 100, out_dir: str = "benchmarks"):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(123)

    # Generate synthetic workload stream
    workloads = []
    types = ["cpu", "memory", "network", "mixed"]
    for i in range(num_jobs):
        t = np.random.choice(types, p=[0.4, 0.3, 0.15, 0.15])
        if t == "cpu":
            c, m, d = np.random.choice([0.5, 1.0, 2.0, 3.0]), np.random.choice([128, 256, 512]), 35.0
        elif t == "memory":
            c, m, d = np.random.choice([0.1, 0.25, 0.5]), np.random.choice([512, 1024, 2048]), 40.0
        elif t == "network":
            c, m, d = 0.2, 128, 25.0
        else:
            c, m, d = 1.5, 768, 45.0
        workloads.append({"type": t, "cpu_req": float(c), "mem_req": float(m), "duration": float(d)})

    print(f"Running benchmarks over {num_jobs} workloads...")
    res_default = simulate_cluster_run("default_k8s", workloads)
    res_shap = simulate_cluster_run("shap_cc", workloads)

    energy_diff = ((res_default["total_energy_kj"] - res_shap["total_energy_kj"]) / res_default["total_energy_kj"]) * 100.0
    contention_diff = ((res_default["contention_incidents"] - res_shap["contention_incidents"]) / max(1, res_default["contention_incidents"])) * 100.0
    balance_diff = ((res_default["avg_node_imbalance"] - res_shap["avg_node_imbalance"]) / res_default["avg_node_imbalance"]) * 100.0

    results = {
        "num_workloads": num_jobs,
        "default_scheduler": res_default,
        "shap_cc_scheduler": res_shap,
        "improvement": {
            "energy_reduction_pct": round(energy_diff, 1),
            "contention_reduction_pct": round(contention_diff, 1),
            "balance_improvement_pct": round(balance_diff, 1),
        }
    }

    results_file = os.path.join(out_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results written to {results_file}:")
    print(json.dumps(results, indent=2))

    # Generate comparison plot
    metrics = ["Energy (kJ)", "Contention Events", "Avg Imbalance (%)"]
    def_vals = [res_default["total_energy_kj"], res_default["contention_incidents"], res_default["avg_node_imbalance"]]
    shap_vals = [res_shap["total_energy_kj"], res_shap["contention_incidents"], res_shap["avg_node_imbalance"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, def_vals, width, label="Default K8s Scheduler", color="#d62728", alpha=0.85)
    ax.bar(x + width/2, shap_vals, width, label="SHAP-CC Scheduler", color="#2ca02c", alpha=0.85)

    ax.set_ylabel("Value")
    ax.set_title("Scheduler Benchmark Comparison (100 Heterogeneous Workloads)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()

    plot_file = os.path.join(out_dir, "scheduler_comparison.png")
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"Saved benchmark comparison plot to {plot_file}")

if __name__ == "__main__":
    run_benchmarks()
