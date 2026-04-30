#!/bin/bash

# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "Error: .env file not found. Create one with MISTRAL_API_KEY and TAVILY_API_KEY."
  exit 1
fi

# Install dependencies
pip install -r requirements.txt

# Run locally with uvicorn (--reload restarts on code changes)
cd app && uvicorn main:app --port 8080 --reload
