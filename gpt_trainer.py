from time import time
import argparse

from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, Trainer, TrainingArguments
import torch

from model import Transformer


DEFAULT_CONFIG = {

    # Transformer parameters
    "embedding_size": 768,
    "context_length": 512,
    "num_layers": 12,
    "dropout": 0.1,
    "num_heads": 12,

    # trainer parameters
    "output_dir": "./model_output",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": './logs',
    "logging_steps": 500,
    "save_steps": 10000,
    "gradient_accumulation_steps": 1,
    
    # Data parameters
    "split_ratio": 0.04,
    "dataset_percent": None,  # full dataset
}


def load_and_split_dataset(tokenizer, seq_len, split_ratio=0.1, dataset_percent=None):
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len, padding='max_length')

    split = f'train[:{dataset_percent}%]' if dataset_percent is not None else 'train'
    dataset = load_dataset("openwebtext", split=split)    
    # Tokenize the dataset. For full dataset, takes ~30 mins the first time
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, batch_size=100000, remove_columns=['text'])
    
    print(f"Full tokenized dataset: {tokenized_dataset}")    
    print("\nSplitting dataset into train/eval ...") # for full dataset, takes ~5 mins the first time
    
    train_test_split = tokenized_dataset.train_test_split(test_size=split_ratio, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    return train_dataset, eval_dataset


# Custom Data Collator
class CustomDataCollatorForLanguageModeling:
    def __call__(self, examples):
        input_ids = torch.stack([torch.tensor(ex["input_ids"]) for ex in examples])
        labels = input_ids.clone()
        labels[:, :-1] = labels[:, 1:]
        labels[:, -1] = -100  # Ignore the computation loss for the last position
        return {"input_ids": input_ids, "labels": labels}


def run(config):

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load and split the dataset
    st = time()
    train_dataset, eval_dataset = load_and_split_dataset(
        tokenizer, config['context_length'], config["split_ratio"], config["dataset_percent"])
    print(f"Train: {train_dataset}")
    print(f"Eval: {eval_dataset}")
    print(f"Datasets created/loaded in {time()-st:.1f} seconds")

    # Initialize the custom data collator
    data_collator = CustomDataCollatorForLanguageModeling()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Transformer(
        config['embedding_size'],
        tokenizer.vocab_size,
        config['context_length'],
        config['num_layers'],
        config['dropout'],
        config['num_heads'],
        device
    )
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_dir=config["logging_dir"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    model.save_pretrained(config["output_dir"])


def parse_args(default_config):
    parser = argparse.ArgumentParser(description='Train a Transformer model from scratch')
    for key, value in default_config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args(DEFAULT_CONFIG)

    # Override defaults with any command-line arguments
    config = {key: getattr(args, key) for key in DEFAULT_CONFIG}
    run(config)
