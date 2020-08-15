"""This script prepares the QASC dataset"""

import os
import jsonlines


def export(input_file, output_file):
    """Reads input_file and transforms to a jsonlines containing objects
    with the following structure:
        {
            "question": "...",
            "answer": "...",
            "distractor": "..."
        }
    """
    instances = []

    print(f"Reading {input_file}")
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            # answerKey is sometimes missing
            if not "answerKey" in obj:
                continue
            # Select the right answer
            choices = obj["question"]["choices"]
            answer = next(c for c in choices if c["label"] == obj["answerKey"])
            # Remove it from the choices
            choices.remove(answer)
            instances.append(
                {"question": obj["question"]["stem"],
                 "answer": answer["text"],
                 "distractor": choices[0]["text"]})

    print(f"Writing {output_file}")
    with jsonlines.open(output_file, mode='w') as writer:
        for instance in instances:
            writer.write(instance)


# Process Files
# =============

output_dir = "processed"

# Make sure processed directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate files
export(input_file="QASC/dev.jsonl",
       output_file=f"{output_dir}/qasc_dev.jsonl")

export(input_file="QASC/test.jsonl",
       output_file=f"{output_dir}/qasc_test.jsonl")

export(input_file="QASC/train.jsonl",
       output_file=f"{output_dir}/qasc_train.jsonl")
