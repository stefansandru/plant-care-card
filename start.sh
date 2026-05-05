#!/bin/bash
cd /
gunicorn -w ${WORKERS:=2} \
  -b :80 -t ${TIMEOUT:=300} \
  -k uvicorn.workers.UvicornWorker \
  app.main:app
