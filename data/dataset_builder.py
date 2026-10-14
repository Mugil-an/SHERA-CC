"""
data/dataset_builder.py
Constructs a large, high-fidelity dataset for training SHAP-CC scheduling models.
Blends empirical observed Prometheus metrics from live cluster runs with realistic
combinatorial parameter sweeps of cluster node states and workload profiles.
"""

import argparse
import os
import numpy as np
import pandas as pd

def build_dataset(
    observed_csv: str = "data/workloads_observed_summary.csv",
    num_samples: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Builds a comprehensive scheduling dataset of (Pod, Node) decision pairs.
    Features include pod demands, node state before scheduling, projected post-allocation load,
    resource imbalance, and labels (energy proxy, contention risk, placement score).
    """
    np.random.seed(random_seed)

    # Check if we have empirical data to seed distributions
    empirical_stats = {}
    if os.path.exists(observed_csv):
        try:
            df_obs = pd.read_csv(observed_csv)
            for wtype in ["cpu", "memory", "network", "mixed"]:
                sub = df_obs[df_obs["workload_type"] == wtype]
                if not sub.empty:
                    empirical_stats[wtype] = {
                        "cpu_avg": sub["cpu_avg_cores"].mean(),
                        "cpu_peak": sub["cpu_peak_cores"].mean(),
                        "mem_avg": sub["mem_avg_mb"].mean(),
                        "mem_peak": sub["mem_peak_mb"].mean(),
                        "energy_avg": sub["energy_proxy_joules"].mean(),
                    }
            print(f"Loaded empirical metrics from {observed_csv} for types: {list(empirical_stats.keys())}")
        except Exception as e:
            print(f"Notice: Could not parse observed csv: {e}. Using baseline calibration.")

    workload_types = ["cpu", "memory", "network", "mixed"]
    rows = []

    for i in range(num_samples):
        # 1. Pod Specification
        wtype = np.random.choice(workload_types, p=[0.40, 0.30, 0.15, 0.15])
        duration_sec = float(np.random.choice([20, 30, 45, 60, 90, 120, 180, 300], p=[0.15, 0.25, 0.20, 0.15, 0.10, 0.08, 0.05, 0.02]))

        if wtype == "cpu":
            cpu_req = float(np.random.choice([0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]))
            mem_req = float(np.random.choice([64, 128, 256, 512]))
            expected_cpu_ratio = np.random.uniform(0.85, 1.05)
            expected_mem_ratio = np.random.uniform(0.3, 0.7)
            net_mbps = float(np.random.exponential(5.0))
        elif wtype == "memory":
            cpu_req = float(np.random.choice([0.1, 0.2, 0.3, 0.5]))
            mem_req = float(np.random.choice([128, 256, 512, 1024, 2048]))
            expected_cpu_ratio = np.random.uniform(0.2, 0.5)
            expected_mem_ratio = np.random.uniform(0.75, 1.0)
            net_mbps = float(np.random.exponential(10.0))
        elif wtype == "network":
            cpu_req = float(np.random.choice([0.1, 0.25, 0.5]))
            mem_req = float(np.random.choice([64, 128, 256]))
            expected_cpu_ratio = np.random.uniform(0.3, 0.7)
            expected_mem_ratio = np.random.uniform(0.3, 0.6)
            net_mbps = float(np.random.choice([50.0, 100.0, 250.0, 500.0]))
        else: # mixed
            cpu_req = float(np.random.choice([0.5, 1.0, 2.0]))
            mem_req = float(np.random.choice([256, 512, 1024]))
            expected_cpu_ratio = np.random.uniform(0.75, 1.0)
            expected_mem_ratio = np.random.uniform(0.70, 0.95)
            net_mbps = float(np.random.exponential(20.0))

        # Adjust based on empirical observations if available
        if wtype in empirical_stats:
            emp = empirical_stats[wtype]
            expected_cpu_cores = max(0.05, emp["cpu_avg"] * np.random.uniform(0.8, 1.2))
            expected_mem_mb = max(32.0, emp["mem_avg"] * np.random.uniform(0.8, 1.2))
        else:
            expected_cpu_cores = cpu_req * expected_cpu_ratio
            expected_mem_mb = mem_req * expected_mem_ratio

        # 2. Candidate Node State (realistic multi-node cluster)
        node_id = np.random.choice(["node-worker-01", "node-worker-02", "node-worker-03", "node-worker-04"])
        node_allocatable_cpu = 12.0 # 12 cores
        node_allocatable_mem = 8192.0 # 8 GB in MB

        # Simulate varied current node utilization
        node_cpu_util_pct = float(np.clip(np.random.beta(2.5, 4.0) * 100.0, 5.0, 95.0))
        node_mem_util_pct = float(np.clip(np.random.beta(2.5, 3.5) * 100.0, 10.0, 92.0))
        node_active_pods = int(np.random.poisson(lam=8))
        
        # Heterogeneous node power efficiency coefficient (1.0 = base, 0.85 = high-eff, 1.15 = older node)
        node_power_efficiency = float(np.random.choice([0.90, 1.0, 1.10]))

        # 3. Post-allocation Impact
        cpu_load_increment = (expected_cpu_cores / node_allocatable_cpu) * 100.0
        mem_load_increment = (expected_mem_mb / node_allocatable_mem) * 100.0
        
        post_alloc_cpu_pct = min(100.0, node_cpu_util_pct + cpu_load_increment)
        post_alloc_mem_pct = min(100.0, node_mem_util_pct + mem_load_increment)
        
        # Kubernetes BalancedAllocation index: imbalance penalty
        resource_imbalance = abs(post_alloc_cpu_pct - post_alloc_mem_pct)

        # 4. Energy Proxy Calculation (RAPL / power proxy model)
        # Power(W) = Base_overhead + CPU_power * cores + Mem_power * GB + Net_power
        # Non-linear dynamic power scaling at high utilization:
        util_factor = 1.0 + (0.5 * max(0.0, (post_alloc_cpu_pct - 70.0) / 30.0) ** 2)
        power_watts = (12.0 + (28.0 * expected_cpu_cores) + (3.2 * (expected_mem_mb / 1024.0)) + (0.015 * net_mbps)) * node_power_efficiency * util_factor
        energy_joules = power_watts * duration_sec

        # 5. Contention & Throttling Risk
        # Steep non-linear risk as node approaches saturation
        cpu_pressure = max(0.0, (post_alloc_cpu_pct - 75.0) / 25.0) ** 2
        mem_pressure = max(0.0, (post_alloc_mem_pct - 80.0) / 20.0) ** 2
        contention_risk = float(np.clip(0.05 + 0.65 * cpu_pressure + 0.60 * mem_pressure + 0.05 * (node_active_pods / 20.0), 0.0, 1.0))

        # 6. Holistic Placement Score (0 to 100, higher is better)
        # High placement score = low energy, low contention risk, balanced resource usage
        norm_energy_penalty = min(35.0, (energy_joules / (power_watts * 60.0 + 1e-5)) * 2.5)
        contention_penalty = contention_risk * 50.0
        imbalance_penalty = (resource_imbalance / 100.0) * 15.0

        placement_score = float(np.clip(100.0 - (norm_energy_penalty + contention_penalty + imbalance_penalty), 0.0, 100.0))

        rows.append({
            "sample_id": f"s-{i:05d}",
            "workload_type": wtype,
            "workload_type_cpu": 1 if wtype == "cpu" else 0,
            "workload_type_memory": 1 if wtype == "memory" else 0,
            "workload_type_network": 1 if wtype == "network" else 0,
            "workload_type_mixed": 1 if wtype == "mixed" else 0,
            "duration_sec": duration_sec,
            "cpu_req_cores": cpu_req,
            "mem_req_mb": mem_req,
            "net_mbps": round(net_mbps, 2),
            "expected_cpu_cores": round(expected_cpu_cores, 3),
            "expected_mem_mb": round(expected_mem_mb, 2),
            "node_cpu_util_pct": round(node_cpu_util_pct, 2),
            "node_mem_util_pct": round(node_mem_util_pct, 2),
            "node_active_pods": node_active_pods,
            "node_power_efficiency": node_power_efficiency,
            "post_alloc_cpu_pct": round(post_alloc_cpu_pct, 2),
            "post_alloc_mem_pct": round(post_alloc_mem_pct, 2),
            "resource_imbalance": round(resource_imbalance, 2),
            "power_watts": round(power_watts, 2),
            "energy_proxy_joules": round(energy_joules, 2),
            "contention_risk": round(contention_risk, 4),
            "placement_score": round(placement_score, 2),
        })

    df = pd.DataFrame(rows)
    return df

def main():
    parser = argparse.ArgumentParser(description="Build training dataset for SHAP-CC")
    parser.add_argument("--samples", type=int, default=5000, help="Number of dataset samples")
    parser.add_argument("--observed-csv", type=str, default="data/workloads_observed_summary.csv")
    parser.add_argument("--output", type=str, default="data/workload_features_dataset.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = build_dataset(observed_csv=args.observed_csv, num_samples=args.samples)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} samples saved to {args.output}")
    print("\nDataset Summary Statistics:")
    print(df[["cpu_req_cores", "mem_req_mb", "post_alloc_cpu_pct", "post_alloc_mem_pct", "energy_proxy_joules", "contention_risk", "placement_score"]].describe().round(2))

if __name__ == "__main__":
    main()
