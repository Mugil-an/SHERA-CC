# infra/setup-kind.ps1
# Creates local 3-node kind cluster for SHAP-CC development

Write-Host "Creating kind cluster from kind-config.yaml..." -ForegroundColor Cyan
kind create cluster --config infra/kind-config.yaml

Write-Host "Verifying nodes..." -ForegroundColor Cyan
kubectl get nodes -o wide

Write-Host "Building workload container image..." -ForegroundColor Cyan
docker build -t shap-workload:local workloads/

Write-Host "Loading workload image into kind cluster nodes..." -ForegroundColor Cyan
kind load docker-image shap-workload:local --name kind

Write-Host "Creating workloads namespace..." -ForegroundColor Cyan
kubectl create namespace workloads

Write-Host "Deploying iperf server..." -ForegroundColor Cyan
kubectl apply -f workloads/iperf-server.yaml

Write-Host "Kind setup completed successfully!" -ForegroundColor Green
