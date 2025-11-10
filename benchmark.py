"""
benchmark.py — Evaluate Multimodal Agent performance across models.
"""

import time
import csv
from nltk.translate.bleu_score import sentence_bleu
from inference.multimodal_agent import MultimodalAgent
import nltk

nltk.download('punkt', quiet=True)

TEST_CASES = [
    {
        "image": "samples/dog.jpg",
        "query": "What animal is in the image?",
        "expected": "a dog"
    },
    {
        "image": "samples/bridge.jpg",
        "query": "What structure is shown?",
        "expected": "a bridge"
    },
    {
        "image": "samples/people.jpg",
        "query": "What are the people doing?",
        "expected": "talking"
    }
]

def benchmark_model(agent, cases=TEST_CASES, csv_out="benchmark_results.csv"):
    print("[BENCHMARK] Running multimodal evaluation...")
    results = []

    for case in cases:
        img, query, expected = case["image"], case["query"], case["expected"]

        t0 = time.time()
        output = agent.process_query(img, query)
        latency = time.time() - t0
        bleu = sentence_bleu([expected.split()], output.split())

        results.append({
            "image": img,
            "query": query,
            "expected": expected,
            "output": output,
            "latency_sec": round(latency, 2),
            "bleu_score": round(bleu, 3)
        })
        print(f" {img} | BLEU: {bleu:.3f} | Latency: {latency:.2f}s")

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n Benchmark completed — results saved to `{csv_out}`")

if __name__ == "__main__":
    agent = MultimodalAgent()
    benchmark_model(agent)
