#!/usr/bin/env python3
import re

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"

def extract_longbench_v2_answer(response: str):
    """Extract answer from model response using official LongBench-v2 method."""
    response = response.replace("*", "")

    # First try: "The correct answer is (A)"
    match = re.search(r"The correct answer is \(([A-D])\)", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Second try: "The correct answer is A"
    match = re.search(r"The correct answer is ([A-D])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: Standard SGLang multichoice pattern
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
    if match:
        return match.group(1).upper()

    # Generic fallback when model says "answer is A"
    match = re.search(r"answer\s+is\s*\(?([A-D])\)?", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None

# Test responses from our HTML files
run1_example5 = """I think the correct answer is B, as it's a direct statement from the text. However, I"""

run2_example5 = """Wait, the question is about which conclusion is the"""

run2_example6 = """Hmm. Maybe the incorrect statement is option B. Wait, the first article's grass blades are modeled with Bezier curves and physical forces, but the second article's shadowing technique uses color changes based on height. The first article's method might not use color changes for undulation"""

print("=== ANSWER EXTRACTION TEST ===")
print(f"Run 1 Example 5: '{extract_longbench_v2_answer(run1_example5)}'")
print(f"Run 2 Example 5: '{extract_longbench_v2_answer(run2_example5)}'")
print(f"Run 2 Example 6: '{extract_longbench_v2_answer(run2_example6)}'")

# Let's also test what would trigger a successful extraction
test_phrases = [
    "The correct answer is B",
    "The correct answer is (B)",
    "I think the correct answer is B",
    "answer is B",
    "Answer: B"
]

print("\n=== EXTRACTION PATTERNS TEST ===")
for phrase in test_phrases:
    result = extract_longbench_v2_answer(phrase)
    print(f"'{phrase}' -> '{result}'")