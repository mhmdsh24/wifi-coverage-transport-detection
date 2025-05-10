# Deploying to Azure Kubernetes Service (AKS)

This guide provides step-by-step instructions to deploy the WiFi Coverage Analysis application to Azure Kubernetes Service (AKS).

## Prerequisites

- Azure CLI installed: [Install Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- kubectl installed: [Install kubectl](https://kubernetes.io/docs/tasks/tools/)
- Docker installed: [Install Docker](https://docs.docker.com/get-docker/)
- An Azure subscription

## Deployment Steps

### 1. Update Configuration

Before deploying, update the Azure Container Registry (ACR) name in the following files:

- `azure/build-push-acr.sh`: Update `ACR_NAME` variable
- `azure/create-aks-cluster.sh`: Update `ACR_NAME` variable
- `azure/deploy-to-aks.sh`: Update `ACR_NAME` variable

### 2. Create AKS Cluster and ACR

```bash
# Make the script executable
chmod +x azure/create-aks-cluster.sh

# Run the script
./azure/create-aks-cluster.sh
```

This script will:
- Create a new resource group
- Create a new AKS cluster
- Create a new Azure Container Registry
- Attach the ACR to the AKS cluster

### 3. Build and Push Docker Image to ACR

```bash
# Make the script executable
chmod +x azure/build-push-acr.sh

# Run the script
./azure/build-push-acr.sh
```

This script will:
- Build the Docker image
- Push the image to your Azure Container Registry

### 4. Deploy to AKS

```bash
# Make the script executable
chmod +x azure/deploy-to-aks.sh

# Run the script
./azure/deploy-to-aks.sh
```

This script will:
- Apply the Kubernetes manifests (deployment, service, persistent volumes)
- Display the deployment status

## Accessing the Application

After deployment is complete, you can access the application using the External-IP provided by the LoadBalancer service.

```bash
kubectl get services
```

Look for the `wifi-coverage-analysis` service and use its External-IP to access the application in your browser.

## Managing the Deployment

### Scaling the Application

To scale the application, update the number of replicas:

```bash
kubectl scale deployment wifi-coverage-analysis --replicas=3
```

### Updating the Application

1. Update your code
2. Rebuild and push the Docker image with a new tag
3. Update the deployment to use the new image tag:

```bash
kubectl set image deployment/wifi-coverage-analysis wifi-coverage-analysis=${ACR_NAME}.azurecr.io/wifi-coverage-analysis:new-tag
```

### Deleting the Deployment

To delete the deployment:

```bash
kubectl delete -f kubernetes/service.yaml
kubectl delete -f kubernetes/deployment.yaml
kubectl delete -f kubernetes/persistent-volumes.yaml
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### Check Service Status

```bash
kubectl get services
kubectl describe service wifi-coverage-analysis
``` 