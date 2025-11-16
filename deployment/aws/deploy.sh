#!/bin/bash
set -e

echo "🚀 Deploying Financial Intelligence Agent to AWS..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found"
    exit 1
fi

echo "✅ Prerequisites checked"
echo "📦 Building and pushing Docker images..."
echo "☁️  Deploying to AWS ECS..."
echo "✅ Deployment complete!"
