"""
data/collector.py
Pulls fine-grained per-job and per-node metrics from the Prometheus HTTP API
for all workloads executed in the 'workloads' namespace.
Calculates time-series data and extracts aggregated summary statistics.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import requests

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")

METRIC_QUERIES = {
    "cpu_cores": 'sum by (pod, node) (rate(container_cpu_usage_seconds_total{namespace="workloads", container!="POD", container!=""}[30s]))',
    "memory_bytes": 'sum by (pod, node) (container_memory_working_set_bytes{namespace="workloads", container!="POD", container!=""})',
    "network_rx_bytes_sec": 'sum by (pod) (rate(container_network_receive_bytes_total{namespace="workloads", pod!=""}[30s]))',
    "network_tx_bytes_sec": 'sum by (pod) (rate(container_network_transmit_bytes_total{namespace="workloads", pod!=""}[30s]))',
    "cpu_throttled_sec": 'sum by (pod) (rate(container_cpu_cfs_throttled_seconds_total{namespace="workloads", container!="POD", container!=""}[30s]))',
}

NODE_QUERIES = {
    "node_cpu_util": '1 - avg by (node) (rate(node_cpu_seconds_total{mode="idle"}[1m]))',
    "node_mem_util": '1 - (sum by (node) (node_memory_MemAvailable_bytes) / sum by (node) (node_memory_MemTotal_bytes))',
}

def query_prometheus_instant(promql: str) -> List[Dict[str, Any]]:
    try:
        res = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=15,
        )
        res.raise_for_status()
        data = res.json()
        if data.get("status") == "success":
            return data.get("data", {}).get("result", [])
    except Exception as e:
        print(f"Warning: query failed for '{promql}': {e}")
    return []

def query_prometheus_range(
    promql: str,
    start: float,
    end: float,
    step: str = "5s",
) -> List[Dict[str, Any]]:
    try:
        res = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={"query": promql, "start": start, "end": end, "step": step},
            timeout=30,
        )
        res.raise_for_status()
        data = res.json()
        if data.get("status") == "success":
            return data.get("data", {}).get("result", [])
    except Exception as e:
        print(f"Warning: query_range failed for '{promql}': {e}")
    return []

def get_pod_metadata() -> Dict[str, Dict[str, Any]]:
    """Retrieves pod-to-node mapping and resource specs from kube-state-metrics."""
    metadata: Dict[str, Dict[str, Any]] = {}
    
    # Pod Info (nodes)
    info_res = query_prometheus_instant('kube_pod_info{namespace="workloads"}')
    for item in info_res:
        metric = item.get("metric", {})
        pod = metric.get("pod")
        node = metric.get("node")
        if pod:
            metadata[pod] = {"node": node, "pod": pod}

    # CPU Requests
    cpu_req = query_prometheus_instant('kube_pod_container_resource_requests{namespace="workloads", resource="cpu"}')
    for item in cpu_req:
        pod = item["metric"].get("pod")
        if pod in metadata:
            metadata[pod]["cpu_req_cores"] = float(item["value"][1])

    # Memory Requests
    mem_req = query_prometheus_instant('kube_pod_container_resource_requests{namespace="workloads", resource="memory"}')
    for item in mem_req:
        pod = item["metric"].get("pod")
        if pod in metadata:
            metadata[pod]["mem_req_bytes"] = float(item["value"][1])

    # CPU Limits
    cpu_lim = query_prometheus_instant('kube_pod_container_resource_limits{namespace="workloads", resource="cpu"}')
    for item in cpu_lim:
        pod = item["metric"].get("pod")
        if pod in metadata:
            metadata[pod]["cpu_lim_cores"] = float(item["value"][1])

    # Memory Limits
    mem_lim = query_prometheus_instant('kube_pod_container_resource_limits{namespace="workloads", resource="memory"}')
    for item in mem_lim:
        pod = item["metric"].get("pod")
        if pod in metadata:
            metadata[pod]["mem_lim_bytes"] = float(item["value"][1])

    return metadata

def collect_metrics(lookback_hours: float = 3.0, step: str = "5s") -> pd.DataFrame:
    """Collects range metrics from Prometheus and builds a unified DataFrame."""
    now = time.time()
    start_time = now - lookback_hours * 3600

    print(f"Collecting Prometheus metrics between {pd.to_datetime(start_time, unit='s')} and {pd.to_datetime(now, unit='s')}...")
    
    pod_meta = get_pod_metadata()
    print(f"Found metadata for {len(pod_meta)} pods.")

    series_dfs = []
    for metric_name, query in METRIC_QUERIES.items():
        results = query_prometheus_range(query, start_time, now, step=step)
        rows = []
        for series in results:
            pod = series["metric"].get("pod", "unknown")
            node = series["metric"].get("node", pod_meta.get(pod, {}).get("node", "unknown"))
            for ts, val in series.get("values", []):
                rows.append({
                    "timestamp": pd.to_datetime(float(ts), unit="s", utc=True),
                    "pod": pod,
                    "node": node,
                    metric_name: float(val),
                })
        if rows:
            df_metric = pd.DataFrame(rows)
            series_dfs.append((metric_name, df_metric))

    if not series_dfs:
        print("No time-series data found in Prometheus for specified window.")
        return pd.DataFrame()

    # Merge on timestamp and pod
    merged = series_dfs[0][1]
    for metric_name, df_m in series_dfs[1:]:
        merged = pd.merge(
            merged,
            df_m.drop(columns=["node"], errors="ignore"),
            on=["timestamp", "pod"],
            how="outer",
        )

    # Fill node metadata
    for pod, meta in pod_meta.items():
        if "node" in meta and meta["node"]:
            merged.loc[merged["pod"] == pod, "node"] = meta["node"]
            merged.loc[merged["pod"] == pod, "cpu_req_cores"] = meta.get("cpu_req_cores", 0.1)
            merged.loc[merged["pod"] == pod, "mem_req_bytes"] = meta.get("mem_req_bytes", 64 * 1024 * 1024)
            merged.loc[merged["pod"] == pod, "cpu_lim_cores"] = meta.get("cpu_lim_cores", 0.5)
            merged.loc[merged["pod"] == pod, "mem_lim_bytes"] = meta.get("mem_lim_bytes", 128 * 1024 * 1024)

    # Classify workload
    def classify_workload(pod_name: str) -> str:
        p = str(pod_name).lower()
        if "cpu" in p:
            return "cpu"
        elif "mem" in p:
            return "memory"
        elif "iperf" in p or "net" in p:
            return "network"
        elif "mix" in p:
            return "mixed"
        return "other"

    merged["workload_type"] = merged["pod"].apply(classify_workload)
    merged = merged.fillna(0.0)
    merged = merged.sort_values(["timestamp", "pod"]).reset_index(drop=True)
    return merged

def compute_pod_summary(df_ts: pd.DataFrame) -> pd.DataFrame:
    """Aggregates time-series into per-pod features and proxy energy consumption."""
    if df_ts.empty:
        return pd.DataFrame()

    pod_summaries = []
    grouped = df_ts.groupby("pod")

    for pod, group in grouped:
        if len(group) < 2:
            continue

        w_type = group["workload_type"].iloc[0]
        node = group["node"].iloc[0]
        cpu_req = group["cpu_req_cores"].iloc[0] if "cpu_req_cores" in group else 0.1
        mem_req = group["mem_req_bytes"].iloc[0] if "mem_req_bytes" in group else 64 * 1024 * 1024
        
        # Duration calculation
        duration_sec = (group["timestamp"].max() - group["timestamp"].min()).total_seconds()
        if duration_sec <= 0:
            duration_sec = 30.0

        cpu_mean = group["cpu_cores"].mean()
        cpu_peak = group["cpu_cores"].max()
        cpu_p95 = group["cpu_cores"].quantile(0.95)
        
        mem_mean = group["memory_bytes"].mean()
        mem_peak = group["memory_bytes"].max()

        net_rx_mean = group["network_rx_bytes_sec"].mean()
        net_tx_mean = group["network_tx_bytes_sec"].mean()
        net_peak = max(group["network_rx_bytes_sec"].max(), group["network_tx_bytes_sec"].max())
        
        throttled_sec_total = group["cpu_throttled_sec"].sum() if "cpu_throttled_sec" in group else 0.0

        # Energy Proxy (in Joules):
        # E = Duration * (P_base + P_cpu * CPU_cores + P_mem * (Mem_GB))
        # Where P_base ~ 10W per container overhead, P_cpu ~ 25W/core, P_mem ~ 3W/GB
        mem_gb = mem_mean / (1024 ** 3)
        power_watts = 10.0 + (25.0 * cpu_mean) + (3.0 * mem_gb)
        energy_joules = power_watts * duration_sec

        # Contention score (0.0 to 1.0): based on throttling, high limit proximity, and multi-tenancy
        contention_score = min(1.0, (throttled_sec_total / max(1.0, duration_sec)) * 2.0 + (0.3 if cpu_peak > cpu_req * 1.5 else 0.0))

        pod_summaries.append({
            "pod": pod,
            "workload_type": w_type,
            "node": node,
            "duration_sec": duration_sec,
            "cpu_req_cores": cpu_req,
            "cpu_peak_cores": cpu_peak,
            "cpu_avg_cores": cpu_mean,
            "cpu_p95_cores": cpu_p95,
            "mem_req_mb": mem_req / (1024 * 1024),
            "mem_peak_mb": mem_peak / (1024 * 1024),
            "mem_avg_mb": mem_mean / (1024 * 1024),
            "net_peak_mbps": (net_peak * 8) / 1e6,
            "net_avg_mbps": ((net_rx_mean + net_tx_mean) * 8) / 1e6,
            "throttled_sec": throttled_sec_total,
            "energy_proxy_joules": energy_joules,
            "contention_score": round(contention_score, 4),
        })

    return pd.DataFrame(pod_summaries)

def main():
    parser = argparse.ArgumentParser(description="Extract metrics from Prometheus")
    parser.add_argument("--hours", type=float, default=2.0, help="Lookback window in hours")
    parser.add_argument("--step", type=str, default="5s", help="Query resolution step")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df_raw = collect_metrics(lookback_hours=args.hours, step=args.step)

    if not df_raw.empty:
        raw_path = os.path.join(args.out_dir, "prometheus_metrics_timeseries.csv")
        df_raw.to_csv(raw_path, index=False)
        print(f"Exported raw time-series ({len(df_raw)} points) to {raw_path}")

        df_summary = compute_pod_summary(df_raw)
        summary_path = os.path.join(args.out_dir, "workloads_observed_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"Exported per-pod summary ({len(df_summary)} workloads) to {summary_path}")
        print(df_summary.head(10).to_string(index=False))
    else:
        print("No metrics collected.")

if __name__ == "__main__":
    main()
