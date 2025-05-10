#!/bin/bash

# Set your Azure Container Registry name
ACR_NAME="your-acr-name"
IMAGE_NAME="wifi-coverage-analysis"
IMAGE_TAG="latest"

# Login to Azure
echo "Logging in to Azure..."
az login

# Login to Azure Container Registry
echo "Logging in to Azure Container Registry..."
az acr login --name $ACR_NAME

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG -f vippfinaldata/Dockerfile .

# Tag the image for ACR
echo "Tagging Docker image for ACR..."
docker tag $IMAGE_NAME:$IMAGE_TAG $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG

# Push the image to ACR
echo "Pushing Docker image to ACR..."
docker push $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG

echo "Successfully built and pushed image to ACR: $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG" 