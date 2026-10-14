"""
workloads/generator.py
Automated parametric workload generator and runner for SHAP-CC / Shera-Real.
Generates diverse CPU, Memory, Network, and Mixed stress jobs and runs them
on the Kubernetes cluster to populate Prometheus with real metrics.
"""

import argparse
import json
import os
import subprocess
import time
from typing import Dict, List, Any
import yaml

WORKLOADS_NAMESPACE = "workloads"
WORKLOAD_IMAGE = "shap-workload:local"

def generate_job_manifest(
    job_name: str,
    workload_type: str,
    stress_args: List[str],
    cpu_request: str,
    cpu_limit: str,
    mem_request: str,
    mem_limit: str,
    node_target: str = None,
    timeout_sec: int = 40,
) -> Dict[str, Any]:
    """Creates a Kubernetes Job specification dictionary."""
    container_spec = {
        "name": "workload-task",
        "image": WORKLOAD_IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "resources": {
            "requests": {"cpu": cpu_request, "memory": mem_request},
            "limits": {"cpu": cpu_limit, "memory": mem_limit},
        },
    }

    if workload_type == "network":
        container_spec["command"] = ["iperf3"]
        container_spec["args"] = stress_args
    else:
        container_spec["command"] = ["stress-ng"]
        container_spec["args"] = stress_args + [
            "--timeout", f"{timeout_sec}s",
            "--metrics-brief",
        ]

    pod_spec: Dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container_spec],
    }

    if node_target:
        pod_spec["nodeSelector"] = {"kubernetes.io/hostname": node_target}

    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": WORKLOADS_NAMESPACE,
            "labels": {
                "app.kubernetes.io/part-of": "shap-workloads",
                "workload_type": workload_type,
                "job_name": job_name,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": timeout_sec + 60,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/part-of": "shap-workloads",
                        "workload_type": workload_type,
                        "job_name": job_name,
                    }
                },
                "spec": pod_spec,
            },
        },
    }
    return job_manifest

def get_workload_matrix() -> List[Dict[str, Any]]:
    """Defines a rich combinatorial parameter grid of workloads."""
    workloads = []
    
    # 1. CPU intensive variations
    cpu_configs = [
        {"cpu": "1", "load": "50", "req_cpu": "200m", "lim_cpu": "500m", "req_mem": "64Mi", "lim_mem": "128Mi"},
        {"cpu": "1", "load": "100", "req_cpu": "500m", "lim_cpu": "1000m", "req_mem": "64Mi", "lim_mem": "128Mi"},
        {"cpu": "2", "load": "80", "req_cpu": "500m", "lim_cpu": "1500m", "req_mem": "128Mi", "lim_mem": "256Mi"},
        {"cpu": "2", "load": "100", "req_cpu": "1000m", "lim_cpu": "2000m", "req_mem": "128Mi", "lim_mem": "256Mi"},
        {"cpu": "3", "load": "90", "req_cpu": "1000m", "lim_cpu": "2500m", "req_mem": "128Mi", "lim_mem": "256Mi"},
        {"cpu": "4", "load": "100", "req_cpu": "1500m", "lim_cpu": "3500m", "req_mem": "256Mi", "lim_mem": "512Mi"},
    ]
    for i, cfg in enumerate(cpu_configs):
        workloads.append({
            "type": "cpu",
            "subtype": f"cpu-{cfg['cpu']}c-{cfg['load']}pct",
            "args": ["--cpu", cfg["cpu"], "--cpu-load", cfg["load"]],
            "req_cpu": cfg["req_cpu"], "lim_cpu": cfg["lim_cpu"],
            "req_mem": cfg["req_mem"], "lim_mem": cfg["lim_mem"],
            "duration": 30,
        })

    # 2. Memory intensive variations
    mem_configs = [
        {"vm": "1", "bytes": "96M", "req_cpu": "100m", "lim_cpu": "300m", "req_mem": "96Mi", "lim_mem": "192Mi"},
        {"vm": "1", "bytes": "192M", "req_cpu": "150m", "lim_cpu": "400m", "req_mem": "192Mi", "lim_mem": "288Mi"},
        {"vm": "2", "bytes": "128M", "req_cpu": "200m", "lim_cpu": "500m", "req_mem": "256Mi", "lim_mem": "384Mi"},
        {"vm": "2", "bytes": "256M", "req_cpu": "250m", "lim_cpu": "600m", "req_mem": "512Mi", "lim_mem": "768Mi"},
        {"vm": "1", "bytes": "384M", "req_cpu": "200m", "lim_cpu": "500m", "req_mem": "384Mi", "lim_mem": "512Mi"},
    ]
    for i, cfg in enumerate(mem_configs):
        workloads.append({
            "type": "memory",
            "subtype": f"mem-{cfg['vm']}v-{cfg['bytes']}",
            "args": ["--vm", cfg["vm"], "--vm-bytes", cfg["bytes"], "--vm-hang", "0"],
            "req_cpu": cfg["req_cpu"], "lim_cpu": cfg["lim_cpu"],
            "req_mem": cfg["req_mem"], "lim_mem": cfg["lim_mem"],
            "duration": 30,
        })

    # 3. Network intensive variations (iperf3 to iperf-server)
    net_configs = [
        {"streams": "1", "time": "25", "req_cpu": "100m", "lim_cpu": "300m", "req_mem": "64Mi", "lim_mem": "128Mi"},
        {"streams": "2", "time": "25", "req_cpu": "150m", "lim_cpu": "400m", "req_mem": "64Mi", "lim_mem": "128Mi"},
        {"streams": "4", "time": "25", "req_cpu": "200m", "lim_cpu": "600m", "req_mem": "96Mi", "lim_mem": "192Mi"},
    ]
    for i, cfg in enumerate(net_configs):
        workloads.append({
            "type": "network",
            "subtype": f"net-p{cfg['streams']}",
            "args": ["-c", "iperf-server", "-t", cfg["time"], "-P", cfg["streams"]],
            "req_cpu": cfg["req_cpu"], "lim_cpu": cfg["lim_cpu"],
            "req_mem": cfg["req_mem"], "lim_mem": cfg["lim_mem"],
            "duration": int(cfg["time"]),
        })

    # 4. Mixed stress variations (CPU + Memory / CPU + IO)
    mixed_configs = [
        {"cpu": "1", "vm": "1", "bytes": "128M", "req_cpu": "300m", "lim_cpu": "800m", "req_mem": "160Mi", "lim_mem": "256Mi"},
        {"cpu": "2", "vm": "1", "bytes": "256M", "req_cpu": "600m", "lim_cpu": "1500m", "req_mem": "300Mi", "lim_mem": "450Mi"},
        {"cpu": "1", "io": "2", "req_cpu": "250m", "lim_cpu": "600m", "req_mem": "96Mi", "lim_mem": "192Mi"},
    ]
    for i, cfg in enumerate(mixed_configs):
        if "vm" in cfg:
            args = ["--cpu", cfg["cpu"], "--vm", cfg["vm"], "--vm-bytes", cfg["bytes"]]
            subtype = f"mixed-c{cfg['cpu']}-m{cfg['bytes']}"
        else:
            args = ["--cpu", cfg["cpu"], "--io", cfg["io"]]
            subtype = f"mixed-c{cfg['cpu']}-io{cfg['io']}"
        workloads.append({
            "type": "mixed",
            "subtype": subtype,
            "args": args,
            "req_cpu": cfg["req_cpu"], "lim_cpu": cfg["lim_cpu"],
            "req_mem": cfg["req_mem"], "lim_mem": cfg["lim_mem"],
            "duration": 30,
        })

    return workloads

def run_command(cmd: List[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed: {res.stderr}")
    return res.stdout

def run_workload_wave(
    batch_index: int,
    workloads: List[Dict[str, Any]],
    workers: List[str],
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Runs a batch of workloads, balancing across workers and automatic placement."""
    executed = []
    timestamp_start = time.time()
    
    print(f"\n==================== Launching Wave {batch_index} ({len(workloads)} jobs) ====================")
    for idx, w in enumerate(workloads):
        job_id = f"batch{batch_index:02d}-{w['type'][:3]}-{idx:02d}"
        # Distribute across workers or allow default scheduling
        if idx % 3 == 0:
            target_node = workers[0] if workers else None
        elif idx % 3 == 1:
            target_node = workers[1] if len(workers) > 1 else None
        else:
            target_node = None  # let k8s scheduler decide
        
        manifest = generate_job_manifest(
            job_name=job_id,
            workload_type=w["type"],
            stress_args=w["args"],
            cpu_request=w["req_cpu"],
            cpu_limit=w["lim_cpu"],
            mem_request=w["req_mem"],
            mem_limit=w["lim_mem"],
            node_target=target_node,
            timeout_sec=w["duration"],
        )
        
        manifest_yaml = yaml.dump(manifest)
        if dry_run:
            print(f"[DRY RUN] Would submit Job: {job_id} (Target: {target_node or 'any'})")
        else:
            p = subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, text=True)
            p.communicate(input=manifest_yaml)
            print(f"-> Submitted Job: {job_id} | Node: {target_node or 'auto'} | Type: {w['subtype']}")

        executed.append({
            "job_id": job_id,
            "workload_type": w["type"],
            "subtype": w["subtype"],
            "target_node": target_node,
            "req_cpu": w["req_cpu"],
            "lim_cpu": w["lim_cpu"],
            "req_mem": w["req_mem"],
            "lim_mem": w["lim_mem"],
            "duration": w["duration"],
            "start_time": timestamp_start,
        })

    if not dry_run:
        # Wait for all jobs in this wave to complete
        print(f"Waiting for wave {batch_index} jobs to finish...")
        wait_deadline = time.time() + 90
        while time.time() < wait_deadline:
            out = run_command(["kubectl", "get", "jobs", "-n", WORKLOADS_NAMESPACE, "--no-headers"])
            lines = [l for l in out.strip().splitlines() if f"batch{batch_index:02d}" in l]
            incomplete = [l for l in lines if "1/1" not in l and "Complete" not in l]
            if not incomplete:
                print(f"All jobs in Wave {batch_index} completed successfully!")
                break
            time.sleep(5)
            
    return executed

def main():
    parser = argparse.ArgumentParser(description="Kubernetes Workload Generator for SHAP-CC")
    parser.add_argument("--batches", type=int, default=2, help="Number of workload batches/waves to run")
    parser.add_argument("--batch-size", type=int, default=6, help="Jobs per batch")
    parser.add_argument("--dry-run", action="store_true", help="Generate manifests without submitting")
    parser.add_argument("--clean", action="store_true", help="Clean up completed jobs in workloads namespace")
    args = parser.parse_args()

    if args.clean:
        print("Cleaning up batch jobs in namespace 'workloads'...")
        run_command(["kubectl", "delete", "jobs", "-n", WORKLOADS_NAMESPACE, "-l", "app.kubernetes.io/part-of=shap-workloads"])
        print("Cleanup done.")
        return

    workers = ["kind-worker", "kind-worker2"]
    matrix = get_workload_matrix()
    
    total_executed = []
    for b in range(1, args.batches + 1):
        # Pick slice from matrix
        start_idx = ((b - 1) * args.batch_size) % len(matrix)
        batch_slice = matrix[start_idx : start_idx + args.batch_size]
        if len(batch_slice) < args.batch_size:
            batch_slice += matrix[: args.batch_size - len(batch_slice)]
            
        executed = run_workload_wave(b, batch_slice, workers, dry_run=args.dry_run)
        total_executed.extend(executed)
        
        if b < args.batches and not args.dry_run:
            print("Cooling down 10s between waves...")
            time.sleep(10)

    # Save execution manifest registry
    os.makedirs("data", exist_ok=True)
    with open("data/workload_execution_registry.json", "w") as f:
        json.dump(total_executed, f, indent=2)
    print(f"\nExecution log written to data/workload_execution_registry.json ({len(total_executed)} jobs)")

if __name__ == "__main__":
    main()
