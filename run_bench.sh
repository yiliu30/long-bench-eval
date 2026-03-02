#!/bin/bash

MODEL_PATH=${MODEL_PATH:-/storage/yiliu7/Qwen/Qwen3-8B}

# Example 1
python -m long_bench_eval.cli \
    --api-key "${YOUR_API_KEY}" \
    --base-url http://localhost:8000/v1 \
    --model "${MODEL_PATH}" \
    --max-context-length 40000 \
    --num-examples 50

# Example 2
python -m long_bench_eval.cli \
    --model "${MODEL_PATH}" \
    --base-url http://localhost:30000/v1 \
    --api-key dummy \
    --max-context-length 40000

# Example 3
python -m long_bench_eval.cli \
    --model "${MODEL_PATH}" \
    --base-url http://localhost:30000/v1 \
    --api-key dummy \
    --max-context-length 40000 \
    --num-examples 50

# Original sglang command for reference
# python -m sglang.test.run_eval \
#     --model "${MODEL_PATH}" \
#     --eval-name longbench_v2 \
#     --max-context-length 40000 \
#     --num-examples 50 \
#     --port 30000
