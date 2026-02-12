"""Command-line interface for running LongBench-v2 evaluations."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from .simple_eval_common import ChatCompletionSampler
from .simple_eval_longbench_v2 import LongBenchV2Eval

DEFAULT_KEY_FILE = "scripts/key.json"


def load_api_key(key_file: Optional[str], api_key: Optional[str], env_var: str) -> str:
    """Resolve API credentials from CLI flag, env var, or JSON file."""
    if api_key:
        return api_key

    from_env = os.environ.get(env_var)
    if from_env:
        return from_env

    if key_file:
        path = Path(key_file)
        if path.is_file():
            data = json.loads(path.read_text())
            token = data.get("token") or data.get("api_key")
            if token:
                return token

    raise SystemExit(
        "No API key found. Pass --api-key, set the environment variable, or provide a JSON file with a 'token' entry."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key-file",
        default=DEFAULT_KEY_FILE,
        help="Path to JSON credentials file (expects {'token': '...'}).",
    )
    parser.add_argument("--api-key", default=None, help="Explicit API token; overrides file/env.")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable to set/use for the API token.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:30000/v1",
        help="OpenAI-compatible inference endpoint base URL.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default="deepseek-chat",
        help="Remote model to query via ChatCompletionSampler.",
    )
    parser.add_argument(
        "--data-source",
        default="THUDM/LongBench-v2",
        help="HF dataset name or local JSON/CSV path for evaluation data.",
    )
    parser.add_argument("--num-examples", type=int, default=None, help="Number of examples to evaluate.")
    parser.add_argument("--num-threads", type=int, default=512, help="Thread pool size for evaluation.")
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Repeat the evaluation this many times (n_repeats).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        metavar="CATEGORY",
        help="Optional subset of LongBench task categories to keep.",
    )
    parser.add_argument(
        "--min-context-length",
        type=int,
        default=None,
        help="Drop examples shorter than this many tokens (requires tokenizer).",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help="Drop examples longer than this many tokens (requires tokenizer).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampler temperature.")
    parser.add_argument(
        "--system-message",
        default=None,
        help="System prompt forwarded to the sampler.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max completion tokens for each request.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    api_key = load_api_key(args.key_file, args.api_key, args.api_key_env)
    os.environ.setdefault(args.api_key_env, api_key)

    sampler = ChatCompletionSampler(
        base_url=args.base_url,
        model=args.model,
        system_message=args.system_message,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    evaluator = LongBenchV2Eval(
        model=args.model,
        data_source=args.data_source,
        num_examples=args.num_examples,
        num_threads=args.num_threads,
        n_repeats=args.n_repeats,
        categories=args.categories,
        min_context_length=args.min_context_length,
        max_context_length=args.max_context_length,
    )

    result = evaluator(sampler)
    print(f"Score: {result.score}")
    if result.metrics:
        print("Metrics:")
        print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual CLI entry
    main()
