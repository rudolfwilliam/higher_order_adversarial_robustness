#!/bin/bash
module load python/3.6.0
module load python/3.7.1

# Download virtualenv if not present
pip install --user --upgrade --no-cache-dir virtualenv

# Create a virtual environment
virtualenv .venv --python=python3.7

# Install required libraries
pip install -e .

# Create a new folder for logs
mkdir -p logs