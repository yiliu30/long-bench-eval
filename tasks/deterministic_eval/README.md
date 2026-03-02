Task:

Make sure the evaluate generate the deterministic result.

Test cmd:
- MODEL_PATH /storage/yiliu7/Qwen/Qwen3-8B
- max_length 40960
- max_gen_toks 2048
- max_ctx_length=$((max_length - max_gen_toks))
- cmd to start the serve
```bash
vllm serve ${MODEL_PATH}  \
    --port SERVER_PORT \
     --max-model-len $max_length \
      --gpu-memory-utilization 0.8
```
- cmd to run long-bench-eval
```bash

python -m long_bench_eval.cli \
    --api-key dummy \
    --base-url http://localhost:${SERVER_PORT}/v1 \
    --model ${MODEL_PATH} \
    --max-context-length ${max_ctx_length} \
    --num-threads 512
```

- Potential fix:
- send request in fixed order