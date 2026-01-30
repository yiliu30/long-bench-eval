#!/usr/bin/env python3
"""Run a 5-example LongBench-v2 evaluation using DeepSeek via the OpenAI SDK."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from long_bench_eval.simple_eval_common import ChatCompletionSampler
from long_bench_eval.simple_eval_longbench_v2 import LongBenchV2Eval


def load_api_key(key_file: str, api_key: Optional[str], env_var: str) -> str:
    """Resolve the API key from CLI flag, env var, or key.json."""
    if api_key:
        return api_key

    from_env = os.environ.get(env_var)
    if from_env:
        return from_env

    data = json.loads(Path(key_file).read_text())
    token = data.get("token")
    if token:
        return token

    raise SystemExit("No API key found; supply --api-key, set the env var, or populate key.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", default="scripts/key.json", help="Path to JSON file with {'token': '...'}")
    parser.add_argument("--api-key", default=None, help="Explicit DeepSeek API token (overrides file/env)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable to read for the token")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI-compatible base URL")
    parser.add_argument("--sampler-model", default="deepseek-chat", help="Remote model to query via ChatCompletionSampler")
    parser.add_argument("--eval-model", default="gpt2", help="Tokenizer model name for context filtering")
    parser.add_argument("--data-source", default="THUDM/LongBench-v2", help="Dataset identifier or local JSON/CSV path")
    parser.add_argument("--num-examples", type=int, default=5, help="How many examples to evaluate")
    parser.add_argument("--num-threads", type=int, default=1, help="Thread pool size for evaluation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampler temperature")
    parser.add_argument("--system-message", default="You are a helpful assistant.", help="System prompt for the sampler")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max completion tokens for each request")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args.key_file, args.api_key, args.api_key_env)
    os.environ.setdefault(args.api_key_env, api_key)

    sampler = ChatCompletionSampler(
        base_url=args.base_url,
        model=args.sampler_model,
        system_message=args.system_message,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    evaluator = LongBenchV2Eval(
        model=args.eval_model,
        data_source=args.data_source,
        num_examples=args.num_examples,
        num_threads=args.num_threads,
    )

    result = evaluator(sampler)
    print(f"Score: {result.score}")
    if result.metrics:
        print("Metrics:")
        print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
