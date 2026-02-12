"""Command-line interface for running LongBench-v2 evaluations."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .simple_eval_common import ChatCompletionSampler, make_report
from .simple_eval_longbench_v2 import LongBenchV2Eval

DEFAULT_KEY_FILE = "scripts/key.json"


def _build_sampler_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "base_url": args.base_url,
        "model": args.sampler_model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "system_message": args.system_message,
    }


def _build_eval_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "data_source": args.data_source,
        "tokenizer_model": args.tokenizer_model,
        "num_examples": args.num_examples,
        "num_threads": args.num_threads,
        "n_repeats": args.n_repeats,
        "categories": args.categories,
        "min_context_length": args.min_context_length,
        "max_context_length": args.max_context_length,
    }


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _ensure_default_artifacts(args: argparse.Namespace) -> None:
    """Ensure every run writes into a unique artifact folder when paths are omitted."""

    args.artifacts_run_dir = None
    needs_run_dir = not all([args.dump_jsonl, args.dump_html_dir, args.report_html])
    if not needs_run_dir:
        return

    root = Path(getattr(args, "artifacts_root", "artifacts") or "artifacts")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_slug = _slugify(args.sampler_model or "model")
    run_dir = root / f"{timestamp}_{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.dump_jsonl:
        args.dump_jsonl = str(run_dir / "debug.jsonl")
    if not args.dump_html_dir:
        args.dump_html_dir = str(run_dir / "html")
    if not args.report_html:
        args.report_html = str(run_dir / "report.html")

    args.artifacts_run_dir = run_dir


def dump_jsonl(
    path: str,
    result,
    sampler_config: Dict[str, Any],
    eval_config: Dict[str, Any],
) -> None:
    """Persist per-sample request/response payloads for debugging."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    convos = result.convos or []
    htmls = result.htmls or []

    with output_path.open("w", encoding="utf-8") as fh:
        for idx, (convo, html) in enumerate(zip(convos, htmls)):
            if not convo:
                continue
            record = {
                "index": idx,
                "prompt": convo[0] if convo else None,
                "response": convo[-1] if len(convo) >= 2 else None,
                "conversation": convo,
                "sampler": sampler_config,
                "evaluation": eval_config,
                "html": html,
            }
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")


def dump_example_htmls(path: str, result) -> None:
    """Write each per-example HTML snippet to disk for manual inspection."""

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, html in enumerate(result.htmls or []):
        if not html:
            continue
        (output_dir / f"example_{idx:04d}.html").write_text(html, encoding="utf-8")


def save_report_html(path: str, result) -> None:
    """Render the aggregated HTML report (metrics + examples)."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(make_report(result), encoding="utf-8")


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
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Base directory for storing run artifacts (default: ./artifacts).",
    )
    parser.add_argument(
        "--dump-jsonl",
        default=None,
        help="Optional path to write JSONL debug records (request/response pairs).",
    )
    parser.add_argument(
        "--dump-html-dir",
        default=None,
        help="Directory to save per-example HTML dumps (mirrors simple-evals output).",
    )
    parser.add_argument(
        "--report-html",
        default=None,
        help="Path to write a combined HTML report (metrics + examples).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _ensure_default_artifacts(args)
    api_key = load_api_key(args.key_file, args.api_key, args.api_key_env)
    os.environ.setdefault(args.api_key_env, api_key)

    sampler = ChatCompletionSampler(
        base_url=args.base_url,
        model=args.model,
        system_message=args.system_message,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    sampler_config = _build_sampler_config(args)
    eval_config = _build_eval_config(args)
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

    if args.dump_jsonl:
        dump_jsonl(args.dump_jsonl, result, sampler_config, eval_config)
    if args.dump_html_dir:
        dump_example_htmls(args.dump_html_dir, result)
    if args.report_html:
        save_report_html(args.report_html, result)
    if getattr(args, "artifacts_run_dir", None):
        print(f"Artifacts saved under {args.artifacts_run_dir}")

if __name__ == "__main__":  # pragma: no cover - manual CLI entry
    main()
