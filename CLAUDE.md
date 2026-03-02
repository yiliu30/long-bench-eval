# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a standalone Python package that provides LongBench-v2 evaluation utilities for long-context language models. The codebase was extracted from SGLang's test suite to create an independent evaluation tool while maintaining compatibility with the official LongBench-v2 benchmark.

## Development Commands

### Package Installation and Setup
```bash
# Install from source (development mode)
pip install -e .

# Install from GitHub
pip install git+https://github.com/yiliu30/long-bench-eval
```

### Running Evaluations

**Command Line Interface:**
```bash
# Basic evaluation with local inference server (deterministic by default)
python -m long_bench_eval.cli \
    --model MODEL_NAME \
    --base-url http://localhost:30000/v1 \
    --api-key dummy \
    --max-context-length 40000 \
    --num-examples 50

# Enable parallel processing for faster evaluation
python -m long_bench_eval.cli \
    --model MODEL_NAME \
    --base-url http://localhost:30000/v1 \
    --api-key dummy \
    --max-context-length 40000 \
    --num-examples 50 \
    --non-deterministic

# Using compatibility shim from scripts/
python scripts/run_deepseek_longbench_eval.py --model MODEL_NAME
```

**Programmatic Usage:**
```python
from long_bench_eval.simple_eval_longbench_v2 import LongBenchV2Eval
from long_bench_eval.simple_eval_common import ChatCompletionSampler

sampler = ChatCompletionSampler(model="gpt-4o-mini")
evaluator = LongBenchV2Eval(model="gpt-4o-mini")
result = evaluator(sampler)
```

### Testing
```bash
# Run specific test functions (manual execution)
python longbench_v2/test_longbench_v2_eval.py

# Validation scripts
python longbench_v2/validate_longbench_v2.py
python longbench_v2/validate_longbench_v2_standalone.py
```

## Architecture

### Core Components

**Evaluation Framework:**
- `simple_eval_common.py`: Base evaluation utilities adapted from OpenAI simple-evals
  - `SamplerBase`: Abstract interface for model inference
  - `ChatCompletionSampler`: OpenAI-compatible API client with retry logic
  - `EvalResult`: Standard result container with scores, metrics, and HTML reports
  - `Eval`: Base evaluation class with threading and HTML generation

**LongBench-v2 Specifics:**
- `simple_eval_longbench_v2.py`: LongBench-v2 specific evaluation logic
  - `LongBenchV2Eval`: Main evaluator class
  - `format_longbench_v2_question()`: Official question template formatting
  - `extract_longbench_v2_answer()`: Answer extraction from model responses
  - Task categories: single_document_qa, multi_document_qa, long_in_context_learning, long_dialogue_history, code_repo_understanding, long_structured_data

**CLI Interface:**
- `cli.py`: Comprehensive command-line interface with artifact management
  - Automatic unique artifact directory creation per run
  - Multiple output formats: JSONL debug files, HTML reports, metric summaries
  - API key management from files, environment, or CLI args
  - Context length filtering and tokenization support

### Data Flow

1. **Input**: HuggingFace dataset (default: "THUDM/LongBench-v2") or local JSON/CSV
2. **Preprocessing**: Question formatting using official templates, optional context length filtering
3. **Inference**: Multi-threaded model queries via SamplerBase implementations
4. **Evaluation**: Answer extraction and correctness scoring per task category
5. **Output**: EvalResult with aggregated metrics, per-example HTML, and debug traces

### Key Configuration Points

**API Integration:**
- Base URL: Configurable OpenAI-compatible endpoint
- Authentication: API keys from `scripts/key.json`, environment vars, or CLI
- Retry Logic: Built-in exponential backoff (6 attempts, 126 seconds total)

**Evaluation Parameters:**
- Context length bounds (min/max tokens with tokenizer-based filtering)
- Task category subsets for focused evaluation
- Thread pool size for parallel inference (default: 512)
- Example limits for testing vs full evaluation

### File Structure Context

- `longbench_v2/`: Legacy validation and test files from original development
- `artifacts/`: Default output directory for evaluation results
- `scripts/`: Convenience scripts and API key storage
- `run_bench.sh`: Example command variations (untracked file)

## Important Implementation Details

**Threading Model**: Uses ThreadPool for parallel model queries with configurable concurrency

**Error Handling**: Robust retry logic in ChatCompletionSampler with exponential backoff for API failures

**Output Formats**: Three complementary output types:
- JSONL: Per-sample request/response debugging
- HTML: Individual example visualizations
- Report: Aggregated metrics and sample gallery

**Tokenization**: Optional transformers-based token counting for context length filtering

**Template Fidelity**: Maintains exact compatibility with official LongBench-v2 question formatting to ensure valid benchmark results