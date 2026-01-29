from sglang.test.simple_eval_longbench_v2 import LongBenchV2Eval
from sglang.test.simple_eval_common import ChatCompletionSampler

# Initialize evaluator with HuggingFace dataset
eval_obj = LongBenchV2Eval(
    data_source="THUDM/LongBench-v2",
    num_examples=100,  # Limit for testing
    num_threads=16,
    model="/storage/yiliu7/Qwen/Qwen3-8B/"
)

# Create sampler (pointing to your SGLang server)
sampler = ChatCompletionSampler(
    base_url="http://localhost:30000/v1",
    model="/storage/yiliu7/Qwen/Qwen3-8B/"
)

# Run evaluation
result = eval_obj(sampler)
print(f"Overall Score: {result.score:.3f}")
print(f"Metrics: {result.metrics}")