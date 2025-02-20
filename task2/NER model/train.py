import json
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

#load data
with open('ner_dataset.json') as f:
    data = json.load(f)

random.shuffle(data)

#80% train, 20% eval
split_index = int(0.8 * len(data))
train_data = data[:split_index]
eval_data = data[split_index:]

#create dataset
datasets = DatasetDict({
    "train": Dataset.from_list(train_data),
    "eval": Dataset.from_list(eval_data)
})

#load bert
MODEL_NAME = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, padding='max_length', is_split_into_words=True, max_length=20)
    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["ner_tags"][word_idx])
        else:
            labels.append(labels[-1])
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

#tokenization
tokenized_datasets = datasets.map(tokenize_and_align_labels)

#model init
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
)

#training
trainer.train()

#saving
torch.save(model.state_dict(), 'ner_model.pth')
torch.save(tokenizer, 'ner_tokenizer.pth')

print("model saved")