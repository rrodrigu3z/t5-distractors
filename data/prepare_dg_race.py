"""This script prepares the Distractor-Generation-RACE dataset"""

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
    file_ids = []

    print(f"Reading {input_file}")
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            if obj["id"]["file_id"] in file_ids:
                continue

            file_ids.append(obj["id"]["file_id"])
            instances.append({"question": " ".join(obj["question"]),
                              "answer": " ".join(obj["answer_text"]),
                              "distractor": " ".join(obj["distractor"])})

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
export(input_file="DG_RACE/race_dev_updated.json",
       output_file=f"{output_dir}/dg_race_dev.jsonl")

export(input_file="DG_RACE/race_test_updated.json",
       output_file=f"{output_dir}/dg_race_test.jsonl")

export(input_file="DG_RACE/race_train_updated.json",
       output_file=f"{output_dir}/dg_race_train.jsonl")
