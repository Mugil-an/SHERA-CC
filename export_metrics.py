
import time
import requests
import pandas as pd

PROMETHEUS = "http://localhost:9090"

QUERIES = {
    "cpu_cores": """
        sum by (pod) (
          rate(container_cpu_usage_seconds_total{
            namespace="workloads",
            container!="",
            image!=""
          }[1m])
        )
    """,
    "memory_bytes": """
        sum by (pod) (
          container_memory_working_set_bytes{
            namespace="workloads",
            container!="",
            image!=""
          }
        )
    """,
    "network_rx_bytes_sec": """
        sum by (pod) (
          rate(container_network_receive_bytes_total{
            namespace="workloads",
            pod!=""
          }[1m])
        )
    """,
    "network_tx_bytes_sec": """
        sum by (pod) (
          rate(container_network_transmit_bytes_total{
            namespace="workloads",
            pod!=""
          }[1m])
        )
    """,
}

def query_range(name, promql, start, end, step="15s"):
    response = requests.get(
        f"{PROMETHEUS}/api/v1/query_range",
        params={
            "query": promql,
            "start": start,
            "end": end,
            "step": step,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    if payload["status"] != "success":
        raise RuntimeError(payload)

    rows = []
    for series in payload["data"]["result"]:
        pod = series["metric"].get("pod", "unknown")

        for timestamp, value in series["values"]:
            rows.append({
                "timestamp": pd.to_datetime(
                    float(timestamp), unit="s", utc=True
                ),
                "pod": pod,
                name: float(value),
            })

    return pd.DataFrame(rows)

def main():
    end = time.time()
    start = end - 2 * 60 * 60
    merged = None

    for name, promql in QUERIES.items():
        df = query_range(name, promql, start, end)
        if df.empty:
            print(f"No samples returned for {name}")
            continue

        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df,
                on=["timestamp", "pod"],
                how="outer",
            )

    if merged is None or merged.empty:
        raise SystemExit(
            "No metrics found. Check Prometheus port-forward, "
            "workload namespace, and query results."
        )

    merged["workload_type"] = merged["pod"].map(
        lambda pod:
            "cpu" if pod.startswith("cpu-test-") else
            "memory" if pod.startswith("memory-test-") else
            "network" if pod.startswith("iperf-client-") else
            "other"
    )

    merged = merged.sort_values(["timestamp", "pod"])
    merged.to_csv("workload_metrics.csv", index=False)

    print(f"Exported {len(merged)} rows")
    print(merged.tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
