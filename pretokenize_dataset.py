import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from itertools import islice

def main():
    parser = argparse.ArgumentParser(description="Pretokenize dataset for LLM training")
    parser.add_argument("--max_examples", type=int, default=None, 
                        help="max num of examples to process for testing")
    parser.add_argument("--max_length", type=int, default=None)
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
    
    # def tokenize_function(examples):
    #     outputs = tokenizer(
    #         examples["text"], 
    #         truncation=True, 
    #         padding="max_length", 
    #         max_length=max_length, 
    #         return_tensors="pt"
    #     )
    #     outputs["labels"] = outputs["input_ids"].clone()
    #     return outputs

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    print("First map: tokenizing in variable length format")
    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    # 5) Define a group/chunk function
    def group_texts(examples):
        """
        Concatenate all tokens from the batch, then split into chunks of chunk_size.
        Return multiple 'input_ids' examples from each batch.
        """
        # Concatenate
        all_input_ids = []
        all_attn_masks = []

        for i in range(len(examples["input_ids"])):
            all_input_ids.extend(examples["input_ids"][i])
            # If you have attention masks from the tokenizer step:
            if "attention_mask" in examples:
                all_attn_masks.extend(examples["attention_mask"][i])

        # Now chunk them in steps of chunk_size
        total_length = len(all_input_ids)
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        # Drop the small remainder if not divisible
        # (Alternatively, you could keep leftover if you'd like)
        if total_length >= chunk_size:
            total_length = (total_length // chunk_size) * chunk_size

        for i in range(0, total_length, chunk_size):
            chunk_input_ids = all_input_ids[i : i + chunk_size]
            chunk_attn_mask = all_attn_masks[i : i + chunk_size] if all_attn_masks else [1] * chunk_size
            # For causal LM, labels = input_ids
            result["input_ids"].append(chunk_input_ids)
            result["attention_mask"].append(chunk_attn_mask)
            result["labels"].append(chunk_input_ids)  # same as input_ids

        return result

    # 6) Second map: group texts into fixed-size chunks
    tokenized_and_chunked = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,  # or something that merges enough examples to get a big chunk
    )

    # Now we have chunk_size-length examples
    print(f"Train examples: {len(tokenized_and_chunked['train'])}")
    print(f"Test examples: {len(tokenized_and_chunked['test'])}")

    # 7) Save to disk
    if os.path.exists(tokenized_data_dir):
        print(f"Warning: {tokenized_data_dir} already exists, it will be overwritten.")
    tokenized_and_chunked.save_to_disk(tokenized_data_dir)

if __name__ == "__main__":
    main()

    # print("Tokenizing dataset...")
    # # Use appropriate num_proc based on dataset size
    # # num_proc = 1 if streaming else 4
    # num_proc=1

    # tokenized_datasets = dataset["train"].map(
    #     tokenize_function, 
    #     batched=True, 
    #     num_proc=num_proc, 
    #     remove_columns=["text"]
    # )
    
    # Create a DatasetDict with both splits
#     from datasets import DatasetDict
#     tokenized_datasets = DatasetDict({
#         "train": tokenized_datasets,
#         "test": dataset["test"].map(
#             tokenize_function, 
#             batched=True, 
#             num_proc=num_proc, 
#             remove_columns=["text"]
#         )
#     })
    
#     print("Number of tokenized train examples:", len(tokenized_datasets["train"]))
#     print("Number of tokenized test examples:", len(tokenized_datasets["test"]))

#     print(f"Saving tokenized dataset to {tokenized_data_dir}...")
#     tokenized_datasets.save_to_disk(tokenized_data_dir)
    
#     # Create a metadata file to track what was saved
#     with open(os.path.join(tokenized_data_dir, "metadata.txt"), "w") as f:
#         f.write(f"Test mode: {args.test_mode}\n")
#         f.write(f"Max examples: {args.max_examples}\n")
#         f.write(f"Train size: {len(tokenized_datasets['train'])}\n")
#         f.write(f"Test size: {len(tokenized_datasets['test'])}\n")
#         f.write(f"Max sequence length: {max_length}\n")
    
#     print("Pretokenization done!")

# if __name__ == "__main__":
#     main()

