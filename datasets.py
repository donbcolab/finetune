from datasets import load_dataset, load_from_disk
import pandas as pd
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DefaultDataCollator
from joblib import Memory
import torch

# Create a cache directory if it doesn't exist
cache_dir = ".cache"
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)

def load_and_prepare_dataset(dataset_name, config, source="hub"):
    """
    Loads and prepares the specified dataset from the Hugging Face Hub or a local CSV file.

    Args:
        dataset_name (str): The name of the dataset to load or the path to the CSV file.
        config (dict): The configuration dictionary.
        source (str, optional): The source of the dataset. 
            Can be 'hub' for Hugging Face Hub or 'file' for a local CSV file. 
            Defaults to 'hub'.

    Returns:
        datasets.Dataset: The loaded and prepared dataset.
    """
    if source == "hub":
        dataset_config = config["datasets"][dataset_name]
        dataset_path = dataset_config.get("path")
        dataset_columns = dataset_config.get("columns")
        preparation_config = dataset_config.get("preparation")

        # Load the dataset
        dataset = load_dataset(dataset_path)

        # Rename columns if mapping provided
        if dataset_columns:
            dataset = dataset.rename_columns(dataset_columns)

        # Apply dataset-specific preparation
        if preparation_config:
            prep_type = preparation_config["type"]
            if prep_type == "text_classification":
                prefix = preparation_config["prompt_prefix"]
                length = preparation_config["prompt_length"]

                def add_prompt_completion(example):
                    example["prompt"] = prefix + example["text"][:length]
                    example["completion"] = example["text"][length:]
                    return example

                dataset = dataset.map(add_prompt_completion)
            elif prep_type == "summarization":
                # No specific preparation needed for summarization in this example
                # You can add custom logic here if required for other summarization datasets
                pass
    elif source == "file":
        dataset = load_dataset("csv", data_files=dataset_name)
        dataset = dataset["train"]
        # Assuming your CSV has "prompt" and "completion" columns
        # Rename columns if needed based on your CSV structure

    return dataset


@memory.cache
def tokenize_and_collate(dataset, model_name, tokenization, collation, config, **kwargs):
    """
    Tokenizes and collates the dataset based on the selected strategies.

    Args:
        dataset (datasets.Dataset): The loaded dataset.
        model_name (str): The name of the model.
        tokenization (str): The selected tokenization strategy.
        collation (str): The selected collation strategy.
        config (dict): The configuration dictionary.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the tokenized dataset and the data collator.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Apply tokenization
    if tokenization == "default":
        def tokenize_function(examples):
            # Tokenize based on task type (text-classification or summarization)
            task_type = config["datasets"][kwargs["dataset_name"]]["preparation"]["type"]
            if task_type == "text_classification":
                return tokenizer(examples["text"], padding="max_length", truncation=True)
            elif task_type == "summarization":
                # Tokenize prompt and completion separately for summarization
                inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True)
                targets = tokenizer(examples['completion'], padding="max_length", truncation=True)
                # Add 'labels' field as per transformers.Trainer expectation
                inputs["labels"] = targets["input_ids"]
                return inputs
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

    elif tokenization == "custom":
        def tokenize_function(examples):
            # Get max length values from config
            max_input_length = config["tokenization_strategies"]["custom"][
                "max_input_length"
            ]
            max_output_length = config["tokenization_strategies"]["custom"][
                "max_output_length"
            ]

            # Tokenize the prompts and completions separately
            inputs = tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                max_length=max_input_length,
            )
            outputs = tokenizer(
                examples["completion"],
                padding="max_length",
                truncation=True,
                max_length=max_output_length,
            )

            # Concatenate inputs and outputs
            input_ids = inputs["input_ids"] + outputs["input_ids"]
            attention_mask = inputs["attention_mask"] + outputs["attention_mask"]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": outputs["input_ids"],
            }

    else:
        raise ValueError(f"Unsupported tokenization strategy: {tokenization}")

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    # Apply collation
    if collation == "default":
        data_collator = DefaultDataCollator()
    elif collation == "seq2seq":
        data_collator = DataCollatorForSeq2Seq(tokenizer)
    elif collation == "custom":
        def data_collator(features):
            # Custom collation logic based on the notebook
            input_ids = [feature["input_ids"] for feature in features]
            attention_mask = [feature["attention_mask"] for feature in features]
            labels = [feature["labels"] for feature in features]

            # Pad to the maximum length in the batch
            max_length = max(len(ids) for ids in input_ids)
            padded_input_ids = [
                ids + [tokenizer.pad_token_id] * (max_length - len(ids))
                for ids in input_ids
            ]
            padded_attention_mask = [
                mask + [0] * (max_length - len(mask)) for mask in attention_mask
            ]
            padded_labels = [
                labs + [-100] * (max_length - len(labs)) for labs in labels
            ]

            return {
                "input_ids": torch.tensor(padded_input_ids),
                "attention_mask": torch.tensor(padded_attention_mask),
                "labels": torch.tensor(padded_labels),
            }

    else:
        raise ValueError(f"Unsupported collation strategy: {collation}")

    return tokenized_dataset, data_collator