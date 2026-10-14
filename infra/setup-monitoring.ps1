# infra/setup-monitoring.ps1
# Deploys kube-prometheus-stack onto the cluster

Write-Host "Adding prometheus-community Helm repo..." -ForegroundColor Cyan
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

Write-Host "Installing kube-prometheus-stack into namespace 'monitoring'..." -ForegroundColor Cyan
helm install monitoring prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace

Write-Host "Waiting for Prometheus and Grafana pods to reach Ready state..." -ForegroundColor Cyan
kubectl wait --namespace monitoring --for=condition=ready pod --selector=app.kubernetes.io/name=prometheus --timeout=300s

Write-Host "Port-forward commands for accessing Grafana & Prometheus:" -ForegroundColor Green
Write-Host "Grafana:    kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80"
Write-Host "Prometheus: kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090"
