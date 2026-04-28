#!/usr/bin/env bash
# start_vllm.sh — Launch vLLM OpenAI-compatible server on AMD MI300x
#
# vLLM 0.17.1 is installed inside the 'rocm' Docker container.
# Port 8000 is already mapped from the container to the host, so the
# server is reachable at http://localhost:8000/v1 from the host machine.
#
# Usage (run from the HOST, not inside the container):
#   bash start_vllm.sh                                       # Qwen2.5-7B default
#   MODEL=Qwen/Qwen2.5-3B-Instruct bash start_vllm.sh       # smaller/faster
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B bash start_vllm.sh
#
# Models to try (all open-source, ≤10B, good spatial reasoning):
#   Qwen/Qwen2.5-7B-Instruct                  ~14 GB — recommended
#   Qwen/Qwen2.5-3B-Instruct                  ~6  GB — faster
#   deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   ~14 GB — reasoning distill
#   mistralai/Mistral-7B-Instruct-v0.3         ~14 GB — alternative
#
# With 192 GB VRAM and a 7B bfloat16 model (~14 GB):
#   max-num-seqs 256 → 256 concurrent requests; vLLM batches them automatically.

set -e

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
MAX_SEQS="${MAX_SEQS:-256}"
MAX_LEN="${MAX_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}"
CONTAINER="${CONTAINER:-rocm}"

echo "================================================================"
echo "  Starting vLLM server INSIDE Docker container: $CONTAINER"
echo "  Model    : $MODEL"
echo "  Port     : $PORT  (host port → container port mapping already set)"
echo "  Max seqs : $MAX_SEQS"
echo "  VRAM     : 192 GB (AMD MI300x)"
echo "  vLLM ver : 0.17.1"
echo "================================================================"
echo "  Server will be reachable at http://localhost:$PORT/v1"
echo "================================================================"

docker exec -it "$CONTAINER" python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90
