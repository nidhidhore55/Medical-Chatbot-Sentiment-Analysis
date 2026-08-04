"""
evaluate_sentiment.py

Measures sentiment-detection accuracy against a small labeled test set,
covering the task's evaluation criteria: "Accuracy of sentiment detection,
appropriateness of responses to different sentiments."

Replace TEST_SET with real examples pulled from your own chatbot logs /
MedQuAD-based conversation samples for a more meaningful number - this
starter set is just to prove the pipeline works end-to-end.
"""

from sentiment_analyzer import SentimentAnalyzer

TEST_SET = [
    ("Thank you, that fixed it perfectly!", "positive"),
    ("I love how quickly this got resolved.", "positive"),
    ("This is unacceptable, I've been waiting for hours.", "negative"),
    ("Your product broke again, I'm so frustrated.", "negative"),
    ("Can you walk me through the return process?", "neutral"),
    ("What are your business hours?", "neutral"),
    ("I'm really happy with the support I received today.", "positive"),
    ("Nothing about this experience has been good.", "negative"),
    ("Please cancel my subscription.", "neutral"),
    ("This is the worst customer service I've ever dealt with.", "negative"),
]


def run_evaluation():
    analyzer = SentimentAnalyzer()
    print(f"Backend: {analyzer.backend}\n")

    correct = 0
    per_class = {"positive": [0, 0], "negative": [0, 0], "neutral": [0, 0]}  # [correct, total]

    for text, expected in TEST_SET:
        result = analyzer.analyze(text)
        is_correct = result.label == expected
        correct += int(is_correct)
        per_class[expected][1] += 1
        per_class[expected][0] += int(is_correct)

        status = "PASS" if is_correct else "FAIL"
        print(f"[{status}] expected={expected:8s} got={result.label:8s} "
              f"(conf={result.score:.2f})  \"{text}\"")

    accuracy = correct / len(TEST_SET)
    print(f"\nOverall accuracy: {accuracy:.1%} ({correct}/{len(TEST_SET)})")
    print("\nPer-class accuracy:")
    for label, (c, t) in per_class.items():
        pct = c / t if t else 0
        print(f"  {label:8s}: {pct:.1%} ({c}/{t})")


if __name__ == "__main__":
    run_evaluation()
