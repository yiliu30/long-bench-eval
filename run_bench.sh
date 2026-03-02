python -m long_bench_eval.cli \
    --api-key YOUR_API_KEY \
    --base-url http://localhost:8000/v1 \
    --model /storage/yiliu7/Qwen/Qwen3-8B \
    --max-context-length 40000   \
      --num-examples 50      \
      --port 30000
      
      
python -m long_bench_eval.cli  \
    --model /storage/yiliu7/Qwen/Qwen3-8B \
    --base-url http://localhost:30000/v1  \
    --api-key dummy     --max-context-length 40000
      
python -m long_bench_eval.cli \
    --model /storage/yiliu7/Qwen/Qwen3-8B \
    --base-url http://localhost:30000/v1 \
    --api-key dummy \
    --max-context-length 40000   \
      --num-examples 50 
    
# python -m sglang.test.run_eval    \
#     --model /storage/yiliu7/Qwen/Qwen3-8B    \
#     --eval-name longbench_v2   \
#     --max-context-length 40000   \
#       --num-examples 50          --port 30000