#!/bin/bash

# Set your Azure Container Registry name
ACR_NAME="your-acr-name"  # Replace with your ACR name

# Login to Azure
echo "Logging in to Azure..."
az login

# Get AKS credentials
echo "Getting AKS credentials..."
az aks get-credentials --resource-group wifi-coverage-rg --name wifi-coverage-aks

# Replace ACR_NAME in deployment.yaml
echo "Updating Kubernetes manifests with ACR name..."
sed -i "s/\${ACR_NAME}/${ACR_NAME}/g" kubernetes/deployment.yaml

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f kubernetes/persistent-volumes.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

# Check deployment status
echo "Checking deployment status..."
kubectl get pods
kubectl get services

echo "Deployment completed. To access the application, use the External-IP from the service output above." 