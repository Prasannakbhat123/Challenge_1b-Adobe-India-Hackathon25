#!/bin/bash
# Build script for Round 1B

echo "🚀 Building Adobe Hackathon Round 1B - Persona-Driven Intelligence"

# Build Docker image
docker build --platform linux/amd64 -t round1b:latest .

echo "✅ Round 1B build complete!"
echo "To run: docker run --rm -v \$(pwd)/input:/app/input -v \$(pwd)/output:/app/output --network none round1b:latest"
