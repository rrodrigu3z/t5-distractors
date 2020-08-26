import argparse
import os
import logging
import random

import nltk
nltk.download('punkt')

import numpy as np
import torch
import pytorch_lightning as pl

from transformers import T5Tokenizer

from datasets import DistractorDataset
from fine_tuner import T5FineTuner, LoggingCallback

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

BASE_MODEL = "t5-small"
MAX_SEQ_LENGTH = 512
logger = logging.getLogger(__name__)

args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path=BASE_MODEL,
    tokenizer_name_or_path=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=6,
    eval_batch_size=6,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)

data_path = os.path.join("data", "processed", "dg_race")
dataset = DistractorDataset(tokenizer, data_path, "test", MAX_SEQ_LENGTH)
print("Val dataset: ",len(dataset))

data = dataset[42]
print(tokenizer.decode(data["source_ids"]))
print(tokenizer.decode(data["target_ids"]))

if not os.path.exists("t5_distractor"):
    os.makedirs('t5_distractor')

args_dict.update({"data_dir": data_path,
                  "output_dir": "t5_distractor",
                  "num_train_epochs": 2,
                  "max_seq_length": MAX_SEQ_LENGTH})

args = argparse.Namespace(**args_dict)
print(args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir,
    prefix="checkpoint",
    monitor="val_loss",
    mode="min",
    save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision=(16 if args.fp_16 else 32),
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)


def get_dataset(tokenizer, type_path, args):
  return DistractorDataset(tokenizer=tokenizer,
                           data_dir=args.data_dir,
                           type_path=type_path,
                           max_len=args.max_seq_length)



print("Initialize model")
model = T5FineTuner(args)
model.set_dataset_loader(get_dataset)

trainer = pl.Trainer(**train_params)

print("Training model")
trainer.fit(model)

print("Training finished")

print ("Saving model")
model.model.save_pretrained("t5_distractor")

print ("Model saved")
