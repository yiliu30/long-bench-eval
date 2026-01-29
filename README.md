# vibe-long-bench

Standalone extraction of the LongBench-v2 evaluation helper that originally lived under `sglang.test`. This repo keeps the original structure and logic so it can drop into existing evaluation workflows with minimal friction.

## Layout

```
vibe-long-bench/
├── README.md
├── vibe_long_bench/
│   ├── __init__.py
│   ├── simple_eval_common.py
│   └── simple_eval_longbench_v2.py
```

## Usage

1. Install the required Python packages (PyTorch/Transformers/Datasets/OpenAI stack) in your environment:
   ```bash
   pip install transformers datasets openai httpx jinja2 numpy requests tqdm
   ```
2. Import and run the evaluator in your own driver script (you need to provide a `SamplerBase` implementation, such as the included `ChatCompletionSampler`):
   ```python
   from vibe_long_bench.simple_eval_longbench_v2 import LongBenchV2Eval
   from vibe_long_bench.simple_eval_common import ChatCompletionSampler

   sampler = ChatCompletionSampler(model="gpt-4o-mini")
   evaluator = LongBenchV2Eval(model="gpt-4o-mini")
   result = evaluator(sampler)
   ```
3. Inspect `result.metrics` or generate an HTML report with `make_report(result)` from `simple_eval_common`.

## Differences vs. the sglang version

- Module paths now live under `vibe_long_bench.*` instead of `sglang.test.*`.
- Imports were updated accordingly; the evaluation logic and helper utilities remain unchanged.
- Added this README to explain setup and highlight the only functional difference (module namespace).
