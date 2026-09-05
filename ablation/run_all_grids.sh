#!/usr/bin/env bash
PY="C:/Users/user/anaconda3/python.exe"
export PYTHONUNBUFFERED=1          # live progress instead of block buffering
cd "$(dirname "$0")"
echo "===== ABLATION GRIDS START $(date) ====="
echo "----- [1/3] HYBRID (GRU + CNN) -----"
"$PY" -u grid_hybrid.py --encoder both
echo "----- [2/3] LSTM -----"
"$PY" -u grid_lstm.py
echo "----- [3/3] GRAPHMIXER -----"
"$PY" -u grid_graphmixer.py --epochs 20
echo "===== ABLATION GRIDS DONE $(date) ====="
