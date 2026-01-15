#!/bin/bash 

# Commented out virtual environment setup for local development

# Create virtual environment
python3 -m venv my_env

# Activate it
source my_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run pytest to execute tests
pytest

# Start the main program
python3 src/main.py
