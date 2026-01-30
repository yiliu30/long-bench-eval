# long-bench-eval

> **Warning:** This is a vibe-coding project. Expect rough edges, rapid experimentation, and frequent changes without prior notice.

This repository packages the LongBench-v2 evaluation helper that was originally part of `sglang.test`, keeping the same structure so you can reuse the official evaluation flow with minimal effort. The implementation remains faithful to the SGLang helper while standing on the LongBench-v2 benchmark introduced in [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-Context Multitasks](https://arxiv.org/abs/2412.15204).

## Layout

```
long-bench-eval/
├── README.md
├── long_bench_eval/
│   ├── __init__.py
│   ├── simple_eval_common.py
│   └── simple_eval_longbench_v2.py
```

## Installation

Install straight from GitHub (or from PyPI once published):

```bash
pip install git+https://github.com/yiliu30/long-bench-eval
```

## Usage

1. Import and run the evaluator in your own driver script (you need to provide a `SamplerBase` implementation, such as the included `ChatCompletionSampler`):
   ```python
   from long_bench_eval.simple_eval_longbench_v2 import LongBenchV2Eval
   from long_bench_eval.simple_eval_common import ChatCompletionSampler

   sampler = ChatCompletionSampler(model="gpt-4o-mini")
   evaluator = LongBenchV2Eval(model="gpt-4o-mini")
   result = evaluator(sampler)
   ```
2. Inspect `result.metrics` or generate an HTML report with `make_report(result)` from `simple_eval_common`.

## Differences vs. the sglang version

- Module paths now live under `long_bench_eval.*` instead of `sglang.test.*`.
- Imports were updated accordingly; the evaluation logic and helper utilities remain unchanged.
- Added this README to explain setup and highlight the only functional difference (module namespace).
