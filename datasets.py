import os
import jsonlines
from torch.utils.data import Dataset


class DistractorDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=256):
        self.inputs = []
        self.targets = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        path = os.path.join(data_dir, type_path + '.jsonl')
        self._build(path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        # might need to squeeze
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "target_ids": target_ids,
                "target_mask": target_mask}

    def _build(self, path):
        with jsonlines.open(path) as reader:
            for obj in reader:
                c = obj["context"]
                q = obj["question"]
                a = obj["answer"]
                tokenized_input = self._tokenize(
                    f"generate distractor: {q}  answer: {a}  context: {c} </s>")
                tokenized_target = self._tokenize(
                    f"{obj['distractor']} </s>")

                self.inputs.append(tokenized_input)
                self.targets.append(tokenized_target)


    def _tokenize(self, text):
        return self.tokenizer.batch_encode_plus([text],
                                                max_length=self.max_len,
                                                pad_to_max_length=True,
                                                return_tensors="pt")
