# AWS Deployment Guide: Lightweight Multi-Node Kubernetes via k3s (Phase 2)

This guide documents the low-cost, production-aligned deployment of the SHAP-CC / Shera-Real architecture on AWS EC2 using **k3s**, avoiding the expensive managed control plane costs of AWS EKS ($70+/month).

---

## 1. Architecture & Cost Control

### Why k3s instead of EKS?
- **AWS EKS**: Charges \$0.10/hour (\$72/month) solely for the managed control plane, before launching any worker EC2 instances.
- **k3s on EC2**: Runs a lightweight, CNCF-certified Kubernetes server and agents on standard EC2 instances. You pay only for standard EC2 compute and EBS volumes (~\$3–4/day total if left running, or pennies per test session).

### Recommended Cluster Layout

| Node | Instance Type | vCPU | RAM | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Node 1** | `t3.medium` | 2 | 4 GiB | k3s server (Control Plane) + Prometheus / Grafana |
| **Node 2** | `t3.small` or `t3.medium` | 2 | 2–4 GiB | k3s agent 1 (Standard workload execution) |
| **Node 3** | `t3.small` or `t3.medium` | 2 | 2–4 GiB | k3s agent 2 (Standard workload execution) |
| *(Optional)* | `c5.large` | 2 | 4 GiB | Compute-optimized agent (run for 2-3 hours for heterogeneous scheduling tests) |
| *(Optional)* | `r5.large` | 2 | 16 GiB | Memory-optimized agent (run for 2-3 hours for heterogeneous scheduling tests) |

---

## 2. Pre-Launch Cost-Control Checklist

1. **Set AWS Billing Budget Alert**:
   - Go to AWS Console → **Billing and Cost Management** → **Budgets**.
   - Create a monthly budget (e.g., \$20 or \$50) with email threshold alerts at 50%, 80%, and 100%.
2. **Single Region Constraint**: Keep all EC2 instances, VPC, and Elastic IPs in one region (e.g., `us-east-1` or your closest low-latency region).
3. **Stop Instances Daily**:
   - At the end of every testing session, **Stop** (not terminate) instances:
     ```bash
     aws ec2 stop-instances --instance-ids i-xxxxxxx i-yyyyyyy i-zzzzzzz
     ```
4. **Delete Unattached EBS Volumes & Elastic IPs**: Check the EC2 dashboard regularly for unused resources.

---

## 3. Network & Security Group Configuration

Create one VPC with a dedicated Security Group:
- **Inbound Rules**:
  - `SSH (22)`: Your Public IP only (`/32`).
  - `Kubernetes API (6443)`: Your Public IP (`/32`) and Internal Security Group (`sg-xxxx`).
  - `Node-to-Node Traffic (All Ports)`: Source set to the Security Group ID itself (`sg-xxxx`) for flannel / pod overlay network.
  - `Node Exporter (9100)` & `Prometheus (9090)`: Internal SG.

---

## 4. k3s Cluster Installation

### Step A: Install k3s Server (Control Plane)
SSH into `t3.medium` (Node 1):
```bash
# Install k3s server with flannel and disable traefik (not needed for batch jobs)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik --write-kubeconfig-mode 644" sh -

# Verify node status
kubectl get nodes

# Retrieve join token for agent nodes
sudo cat /var/lib/rancher/k3s/server/node-token
```
*Note the token string and the Private IP of Node 1.*

### Step B: Join k3s Agents (Worker Nodes)
SSH into Node 2 and Node 3:
```bash
# Replace <SERVER_PRIVATE_IP> and <NODE_TOKEN>
curl -sfL https://get.k3s.io | K3S_URL=https://<SERVER_PRIVATE_IP>:6443 K3S_TOKEN=<NODE_TOKEN> sh -
```

### Step C: Configure Local Access on Laptop
On your laptop, copy the kubeconfig file from Node 1:
```powershell
# Fetch kubeconfig from EC2 server
scp -i your-key.pem ubuntu@<SERVER_PUBLIC_IP>:/etc/rancher/k3s/k3s.yaml ./aws-kubeconfig.yaml

# Edit the server IP in aws-kubeconfig.yaml from 127.0.0.1 to <SERVER_PUBLIC_IP>
$content = Get-Content ./aws-kubeconfig.yaml
$content = $content -replace "127.0.0.1", "<SERVER_PUBLIC_IP>"
$content | Set-Content ./aws-kubeconfig.yaml

# Test access
kubectl --kubeconfig=./aws-kubeconfig.yaml get nodes -o wide
```

---

## 5. Deploy Monitoring & Workloads on AWS

With your local kubectl pointed to AWS:
```powershell
$env:KUBECONFIG = "$PWD/aws-kubeconfig.yaml"

# 1. Install kube-prometheus-stack via Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install monitoring prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace

# 2. Push shap-workload image to Amazon ECR or Docker Hub
# Example Docker Hub:
docker tag shap-workload:local your-username/shap-workload:latest
docker push your-username/shap-workload:latest

# 3. Create workloads namespace & deploy iperf server
kubectl create namespace workloads
kubectl apply -f workloads/iperf-server.yaml

# 4. Port-forward Prometheus to laptop
kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090
```

---

## 6. Energy & Power Measurement Realities on AWS EC2

### Important Caveat: Lack of Bare-Metal RAPL Access
On virtualized EC2 instances (HVM / Nitro), the hypervisor prevents direct access to Intel/AMD RAPL (Running Average Power Limit) MSR registers.
Tools like Kepler or Scaphandre fall back to estimator models rather than hardware measurements.

### Recommended Approaches for Your Paper / Thesis:
1. **CPU-seconds & Utilization as Explicit Energy Proxy (Primary & Robust)**:
   - Calculate energy using the empirical power model:
     $$E = \int \left(P_{\text{base}} + P_{\text{cpu}} \cdot u_{\text{cpu}}(t) + P_{\text{mem}} \cdot m(t)\right) dt$$
   - Clearly state in your write-up:
     > *"On virtualized AWS EC2 instances, hardware MSRs (RAPL) are isolated by the hypervisor. Following industry best-practices, we employ CPU-seconds, memory working set, and dynamic power proxy modeling validated against bare-metal ground truth."*
2. **Local Bare-Metal RAPL Ground-Truth Calibration**:
   - Run a 10-minute calibration run on Linux bare-metal using `powercap` (`/sys/class/powercap/intel-rapl`) or Scaphandre to calibrate the proxy coefficients ($P_{\text{base}}$, $P_{\text{cpu}}$) before applying to EC2 traces.
3. **AWS Customer Carbon Footprint Tool**:
   - Offers aggregate monthly carbon data, useful as macroscopic validation.
