#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

/home/msds/c250089/.conda/envs/TestEnv/bin/python -m sglang.launch_server \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --context-length 8192 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 4096 \
  --attention-backend flashinfer \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 4.59 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype float32 \
  --host 0.0.0.0 \
  --port 30000
