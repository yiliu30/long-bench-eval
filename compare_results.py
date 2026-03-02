#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def load_debug_data(file_path):
    """Load debug data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compare_results(run1_path, run2_path):
    """Compare results between two runs"""
    data1 = load_debug_data(run1_path)
    data2 = load_debug_data(run2_path)

    print(f"=== COMPARISON RESULTS ===")
    print(f"Run 1: {run1_path}")
    print(f"Run 2: {run2_path}")
    print(f"")

    if len(data1) != len(data2):
        print(f"ERROR: Different number of samples! Run1: {len(data1)}, Run2: {len(data2)}")
        return

    print(f"Total samples: {len(data1)}")
    print(f"")

    # Compare each sample
    differences = []
    identical = 0
    different_responses = 0
    different_scores = 0

    for i, (sample1, sample2) in enumerate(zip(data1, data2)):
        # Check if prompts are identical
        prompt1 = sample1.get('prompt', {}).get('content', '')
        prompt2 = sample2.get('prompt', {}).get('content', '')

        # Check responses
        response1 = sample1.get('response', '')
        response2 = sample2.get('response', '')

        # Check evaluations/scores
        eval1 = sample1.get('evaluation', {})
        eval2 = sample2.get('evaluation', {})

        score1 = eval1.get('score', 0)
        score2 = eval2.get('score', 0)

        if prompt1 != prompt2:
            print(f"ERROR: Prompts differ at index {i}")

        response_same = response1 == response2
        score_same = score1 == score2

        if response_same and score_same:
            identical += 1
        else:
            differences.append({
                'index': i,
                'response_same': response_same,
                'score_same': score_same,
                'score1': score1,
                'score2': score2,
                'response1_len': len(response1),
                'response2_len': len(response2),
                'response1_preview': response1[:200] + '...' if len(response1) > 200 else response1,
                'response2_preview': response2[:200] + '...' if len(response2) > 200 else response2
            })

            if not response_same:
                different_responses += 1
            if not score_same:
                different_scores += 1

    print(f"=== SUMMARY ===")
    print(f"Identical results: {identical}/{len(data1)} ({identical/len(data1)*100:.1f}%)")
    print(f"Different responses: {different_responses}/{len(data1)} ({different_responses/len(data1)*100:.1f}%)")
    print(f"Different scores: {different_scores}/{len(data1)} ({different_scores/len(data1)*100:.1f}%)")
    print(f"")

    if differences:
        print(f"=== DETAILED DIFFERENCES ===")
        for i, diff in enumerate(differences[:10]):  # Show first 10 differences
            print(f"\nDifference {i+1} (Index {diff['index']}):")
            print(f"  Score1: {diff['score1']}, Score2: {diff['score2']}")
            print(f"  Response1 length: {diff['response1_len']}")
            print(f"  Response2 length: {diff['response2_len']}")
            print(f"  Response1 preview: {repr(diff['response1_preview'])}")
            print(f"  Response2 preview: {repr(diff['response2_preview'])}")
            print(f"  Same response: {diff['response_same']}")

        if len(differences) > 10:
            print(f"\n... and {len(differences) - 10} more differences")

    # Calculate score distribution
    scores1 = [sample.get('evaluation', {}).get('score', 0) for sample in data1]
    scores2 = [sample.get('evaluation', {}).get('score', 0) for sample in data2]

    total_score1 = sum(scores1)
    total_score2 = sum(scores2)

    print(f"\n=== SCORE ANALYSIS ===")
    print(f"Total score Run1: {total_score1}/{len(data1)} = {total_score1/len(data1):.3f}")
    print(f"Total score Run2: {total_score2}/{len(data2)} = {total_score2/len(data2):.3f}")
    print(f"Score difference: {abs(total_score1 - total_score2)} points")

if __name__ == "__main__":
    run1 = "artifacts/20260302-130338__storage_yiliu7_Qwen_Qwen3-8B/debug.jsonl"
    run2 = "artifacts/20260302-130450__storage_yiliu7_Qwen_Qwen3-8B/debug.jsonl"

    compare_results(run1, run2)