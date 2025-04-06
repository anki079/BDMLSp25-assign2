import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from itertools import islice

def main():
    parser = argparse.ArgumentParser(description="Pretokenize dataset for LLM training")
    parser.add_argument("--max_examples", type=int, default=None, 
                        help="Maximum number of examples to process for testing")
    parser.add_argument("--test_mode", action="store_true", 
                        help="If set, save to test directory instead of main directory")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    
    # Use streaming if max_examples is set
    streaming = args.max_examples is not None
    
    data_files = {"train": "./train.txt", "test": "./test.txt"}
    max_length = args.max_length
    model_dir = "./llama-hf"
    
    # Determine output directory based on test mode
    if args.test_mode:
        tokenized_data_dir = "./tokenized_data_test"
        print(f"Running in TEST MODE: Will save to {tokenized_data_dir}")
    else:
        tokenized_data_dir = "./tokenized_data"
        print(f"Running in FULL MODE: Will save to {tokenized_data_dir}")
    
    if os.path.exists(tokenized_data_dir):
        print(f"Tokenized dataset already exists at {tokenized_data_dir}")
        user_input = input("Do you want to overwrite it? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting without overwriting.")
            return
        print(f"Will overwrite {tokenized_data_dir}")
    
    os.makedirs(tokenized_data_dir, exist_ok=True)
    
    print(f"Loading {'subset of ' if streaming else ''}raw dataset...")
    
    # Load dataset with streaming option if max_examples is set
    dataset = load_dataset("text", data_files=data_files, streaming=streaming)
    
    # If streaming, take only the number of examples needed
    if streaming:
        # For train set
        train_subset = list(islice(dataset["train"], args.max_examples))
        # For test set - use 10% or at least 100 examples
        test_examples = max(min(args.max_examples // 10, 100), 1)
        test_subset = list(islice(dataset["test"], test_examples))
        
        # Convert back to regular dataset
        from datasets import Dataset
        dataset = {
            "train": Dataset.from_dict({k: [example[k] for example in train_subset] 
                                      for k in train_subset[0].keys()}),
            "test": Dataset.from_dict({k: [example[k] for example in test_subset] 
                                     for k in test_subset[0].keys()})
        }
        print(f"Took subset of {len(dataset['train'])} examples from train set")
        print(f"Took subset of {len(dataset['test'])} examples from test set")
    
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length, 
            return_tensors="pt"
        )
        outputs["labels"] = outputs["input_ids"].clone()
        return outputs

    print("Tokenizing dataset...")
    # Use appropriate num_proc based on dataset size
    num_proc = 1 if streaming else 4
    
    tokenized_datasets = dataset["train"].map(
        tokenize_function, 
        batched=True, 
        num_proc=num_proc, 
        remove_columns=["text"]
    )
    
    # Create a DatasetDict with both splits
    from datasets import DatasetDict
    tokenized_datasets = DatasetDict({
        "train": tokenized_datasets,
        "test": dataset["test"].map(
            tokenize_function, 
            batched=True, 
            num_proc=num_proc, 
            remove_columns=["text"]
        )
    })
    
    print("Number of tokenized train examples:", len(tokenized_datasets["train"]))
    print("Number of tokenized test examples:", len(tokenized_datasets["test"]))

    print(f"Saving tokenized dataset to {tokenized_data_dir}...")
    tokenized_datasets.save_to_disk(tokenized_data_dir)
    
    # Create a metadata file to track what was saved
    with open(os.path.join(tokenized_data_dir, "metadata.txt"), "w") as f:
        f.write(f"Test mode: {args.test_mode}\n")
        f.write(f"Max examples: {args.max_examples}\n")
        f.write(f"Train size: {len(tokenized_datasets['train'])}\n")
        f.write(f"Test size: {len(tokenized_datasets['test'])}\n")
        f.write(f"Max sequence length: {max_length}\n")
    
    print("Pretokenization done!")

if __name__ == "__main__":
    main()