#!/bin/zsh
# Keeps the Streamlit app alive: relaunches it whenever the process dies.
cd "$(dirname "$0")"
while true; do
  .venv/bin/streamlit run app.py --server.headless true --server.port 8501
  echo "$(date) — streamlit exited ($?), restarting in 3s" >> restart.log
  sleep 3
done
