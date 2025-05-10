#!/bin/bash

# Azure resource group and location
RESOURCE_GROUP="wifi-coverage-rg"
LOCATION="eastus"
CLUSTER_NAME="wifi-coverage-aks"
ACR_NAME="your-acr-name"  # Replace with your ACR name

# Login to Azure
echo "Logging in to Azure..."
az login

# Create resource group if it doesn't exist
echo "Creating resource group if it doesn't exist..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create AKS cluster
echo "Creating AKS cluster..."
az aks create \
    --resource-group $RESOURCE_GROUP \
    --name $CLUSTER_NAME \
    --node-count 1 \
    --enable-addons monitoring \
    --generate-ssh-keys

# Get AKS credentials
echo "Getting AKS credentials..."
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME

# Create ACR if it doesn't exist
echo "Creating Azure Container Registry if it doesn't exist..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic

# Attach ACR to AKS
echo "Attaching ACR to AKS..."
az aks update \
    --name $CLUSTER_NAME \
    --resource-group $RESOURCE_GROUP \
    --attach-acr $ACR_NAME

echo "AKS cluster setup complete!" 