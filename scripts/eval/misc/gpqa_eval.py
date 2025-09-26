import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str, required=True, help='Path to the input JSONL file')
args = parser.parse_args()

FP = args.input_file

correct = count = 0
with open(FP) as rf:
    for line in rf:
        line = json.loads(line)
        cot, answer = line["full_conversation"].split("<|im_start|>answer", 1)

        pattern = r'\\boxed\{([A-Z])\}'
        matches = re.findall(pattern, answer)
        if matches:
            pred = matches[-1]
        else:
            pred = None

        correct += int(pred == line['answer'])
        count += 1

print("Accuracy:", f"{correct / count * 100:.2f}%")
print("Count:", count)