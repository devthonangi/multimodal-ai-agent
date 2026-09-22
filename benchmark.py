import argparse
import json
from time import perf_counter

from inference.multimodal_agent import MultimodalAgent


def token_f1(expected, actual):
    expected_tokens = set(expected.lower().split())
    actual_tokens = set(actual.lower().split())
    overlap = len(expected_tokens & actual_tokens)
    if not overlap:
        return 0.0
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser(description="Benchmark one image question")
    parser.add_argument("image")
    parser.add_argument("question")
    parser.add_argument("--expected", help="Optional reference answer")
    args = parser.parse_args()

    agent = MultimodalAgent()
    started = perf_counter()
    answer = agent.process_query(args.image, args.question)
    result = {
        "image": args.image,
        "question": args.question,
        "answer": answer,
        "latency_seconds": round(perf_counter() - started, 3),
        **agent.status(),
    }
    if args.expected:
        result["token_f1"] = round(token_f1(args.expected, answer), 3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
