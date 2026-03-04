# LongBench-v2 Dataset Summary

## Overview
- **Total Samples**: 503 samples across 6 categories
- **Dataset Source**: `THUDM/LongBench-v2` on HuggingFace
- **Context Range**: 48,765 to 16,182,936 characters per sample
- **Format**: Multiple-choice questions (A, B, C, D)

## Category Breakdown

| Category | Samples | % of Total | Min Chars | Max Chars | Mean Chars | Median Chars | Est. Tokens (Mean)* |
|----------|---------|------------|-----------|-----------|------------|--------------|-------------------|
| **Single-Document QA** | 175 | 34.8% | 51,623 | 3,596,979 | 440,503 | 316,318 | ~1,617 |
| **Multi-Document QA** | 125 | 24.9% | 48,765 | 7,211,576 | 519,872 | 226,001 | ~1,909 |
| **Long In-context Learning** | 81 | 16.1% | 53,953 | 5,250,288 | 813,215 | 496,131 | ~2,986 |
| **Code Repository Understanding** | 50 | 9.9% | 101,348 | 16,182,936 | 3,560,624 | 1,941,023 | ~13,072 |
| **Long-dialogue History** | 39 | 7.8% | 87,702 | 566,401 | 325,397 | 195,974 | ~1,195 |
| **Long Structured Data** | 33 | 6.6% | 59,918 | 9,781,854 | 1,207,555 | 550,914 | ~4,433 |

*Token estimates based on GPT-2 tokenizer (2.72 char-to-token ratio)

## Key Characteristics

### Context Length Distribution
- **All samples > 50K characters** - No short-context examples
- **66.4% exceed 200K characters** - Ultra-long context focus
- **Maximum sample**: 16.18M characters (~59K tokens)
- **Overall mean**: 3,201 tokens, **median**: 1,530 tokens

### Model Requirements
- **Minimum context window**: 20K+ tokens recommended
- **Full coverage**: 60K+ tokens needed
- **Challenging for current LLMs**: Exceeds most model context limits

### Category Insights
1. **Code Repository Understanding**: Most demanding (avg 3.6M chars, 13K tokens)
2. **Single & Multi-Document QA**: Core of dataset (60% combined samples)
3. **Long-dialogue History**: Shortest contexts but still substantial (avg 325K chars)
4. **Long Structured Data & ICL**: Medium difficulty with high variance

## Technical Notes
- **Question format**: Official LongBench-v2 template with context + 4 choices
- **Answer extraction**: Regex patterns for "The correct answer is (A)" format
- **Evaluation**: Binary scoring (1.0 correct, 0.0 incorrect)
- **Categories filter**: Can evaluate subsets via `categories` parameter
- **Context filtering**: Min/max token length filtering supported

## Usage Implications
- **Research focus**: Ultra-long context capabilities beyond typical benchmarks
- **Computational requirements**: High memory and processing needs
- **Evaluation time**: Expect longer inference times due to context length
- **Model compatibility**: Best suited for models with 32K+ context windows

---
*Generated from long-bench-eval repository analysis*