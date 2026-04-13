#!/bin/bash
cd /app/backend && uvicorn main:app --host 0.0.0.0 --port 8002 &
nginx -g 'daemon off;'
