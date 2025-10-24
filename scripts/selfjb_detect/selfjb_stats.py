"""
Print summary statistics
"""
import json
import pathlib
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="Path to input file containing model outputs")
args = parser.parse_args()

def print_summary(results):
    """Print summary statistics"""

    all_annotations = []
    sentence_counts = []
    for result in results:
        selfjb_annos = result.get("selfjb_annos", {})
        if isinstance(selfjb_annos, dict) and "answer" in selfjb_annos:
            annotations = selfjb_annos["answer"]
            if isinstance(annotations, list):
                all_annotations.extend(annotations)
                # Count sentences in this CoT
                if "cot_sentences" in result:
                    sentence_count = len(result["cot_sentences"].split("\n"))
                    sentence_counts.append(sentence_count)

    if all_annotations:
        print(f"--------------------------------")
        print(f"Self-jailbreaking Analysis:")
        print(f"Total self-jailbreaking sentences found: {len(all_annotations)}")

        instances_with_selfjb = 0
        for r in results:
            if r.get("selfjb_annos") is not None and r.get("selfjb_annos", {}).get("answer", []):
                instances_with_selfjb += 1
        print(f"Instances with self-jailbreaking: {instances_with_selfjb}")

        if sentence_counts:
            total_sentences = sum(sentence_counts)
            print(f"Total CoT sentences analyzed: {total_sentences}")
            print(f"Self-jailbreaking rate (i.e., self-jailbreaking instances / all instances): {instances_with_selfjb/len(results):.2%}")
            print(f"Per-sentence self-jailbreaking rate (i.e., self-jailbreaking CoT / total CoT sentences): {len(all_annotations)/total_sentences:.2%}")

        # Show distribution of self-jailbreaking sentence indices
        if all_annotations:
            annotation_counts = Counter(all_annotations)
            print(f"Most common self-jailbreaking sentence positions:")
            for idx, count in annotation_counts.most_common(10):
                print(f"  Position {idx}: {count} times")

        print(f"--------------------------------")


if __name__ == "__main__":
    input_fp = pathlib.Path(args.input_file)
    results = []
    with open(input_fp, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    print_summary(results)