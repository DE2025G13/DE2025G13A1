#!/bin/bash
set -e

echo "Authenticating to Google Cloud..."
gcloud auth application-default login

echo "Authentication complete. Starting Python script..."
exec python3 component.py "$@"