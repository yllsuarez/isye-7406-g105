#!/usr/bin/env bash
# Copy trained artifacts from r-models/model into the backend.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p backend/model
cp ../r-models/model/model.pkl ../r-models/model/metadata.json backend/model/
echo "Copied model artifacts to web-service/backend/model/"
