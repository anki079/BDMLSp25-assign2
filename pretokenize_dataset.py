import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from itertools import islice

def main():
    parser = argparse.ArgumentParser(description="Pretokenize dataset for LLM training")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="num of tokens per chunk")
    # parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data_chunks")
    args = parser.parse_args()
    

    tokenized_data_dir = "./tokenized_data_chunks"
    print(f"Running in FULL MODE: Will save to {tokenized_data_dir}")
    
    if os.path.exists(tokenized_data_dir):
        print(f"Tokenized dataset already exists at {tokenized_data_dir}")
        user_input = input("Do you want to overwrite it? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting without overwriting.")
            return
        print(f"Will overwrite {tokenized_data_dir}")
    
    os.makedirs(tokenized_data_dir, exist_ok=True)

    chunk_size = args.chunk_size
  
    data_files = {"train": "./train.txt", "test": "./test.txt"}
    raw_datasets = load_dataset("text", data_files=data_files)

    print("Loading tokenizer...")
    model_dir = "./llama-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    print("First map: tokenizing in variable length format")
    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    #defining a group/chunk function
    def group_texts(examples):
        """
        Concatenate all tokens from the batch, then split into chunks of chunk_size.
        Return multiple 'input_ids' examples from each batch.
        """
        all_input_ids = []
        all_attn_masks = []

        for i in range(len(examples["input_ids"])):
            all_input_ids.extend(examples["input_ids"][i])
            # if attention masks from the tokenizer step are present:
            if "attention_mask" in examples:
                all_attn_masks.extend(examples["attention_mask"][i])

        # chunk in steps of chunk_size
        total_length = len(all_input_ids)
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        # drop the small remainder if not divisible
        if total_length >= chunk_size:
            total_length = (total_length // chunk_size) * chunk_size

        for i in range(0, total_length, chunk_size):
            chunk_input_ids = all_input_ids[i : i + chunk_size]
            chunk_attn_mask = all_attn_masks[i : i + chunk_size] if all_attn_masks else [1] * chunk_size
            # for causal LM, labels = input_ids
            result["input_ids"].append(chunk_input_ids)
            result["attention_mask"].append(chunk_attn_mask)
            result["labels"].append(chunk_input_ids)  # same as input_ids

        return result

    # group texts into fixed-size chunks
    tokenized_and_chunked = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000, 
        remove_columns=["special_tokens_mask"]
    )

    # chunk_size-length examples
    print(f"Train examples: {len(tokenized_and_chunked['train'])}")
    print(f"Test examples: {len(tokenized_and_chunked['test'])}")

    if os.path.exists(tokenized_data_dir):
        print(f"Warning: {tokenized_data_dir} already exists, it will be overwritten.")
    tokenized_and_chunked.save_to_disk(tokenized_data_dir)

if __name__ == "__main__":
    main()
