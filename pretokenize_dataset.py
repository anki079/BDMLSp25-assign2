import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

def main():

    parser = argparse.ArgumentParser(description="Pretokenize dataset for distributed LlaMa fine-tuning")
    parser.add_argument("--max_examples", type=int, default=None, 
                        help="Maximum number of examples to process for testing")
    parser.add_argument("--test_mode", action="store_true", 
                        help="If set, save to test directory instead of main directory")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    data_files = {"train": "./train.txt", "test": "./test.txt"}
    # tokenized_data_dir = "./tokenized_data"
    model_dir = "./llama-hf"
    # max_length = 128

    if args.test_mode:
        tokenized_data_dir = "tokenized_data_test"
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
    
    print("Loading raw dataset...")
    dataset = load_dataset("text", data_files=data_files)
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

    # Subsample if max_examples is provided
    if args.max_examples:
        print(f"Taking subset of {args.max_examples} examples from train set")
        dataset["train"] = dataset["train"].select(range(min(args.max_examples, len(dataset["train"]))))
        
        # For test set, use 10% of max_examples or all available examples
        test_examples = max(args.max_examples // 10, 100)  # At least 100 test examples
        test_examples = min(test_examples, len(dataset["test"]))  # But not more than available
        print(f"Taking subset of {test_examples} examples from test set")
        dataset["test"] = dataset["test"].select(range(test_examples))
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        outputs["labels"] = outputs["input_ids"].clone()
        return outputs

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    print("Number of tokenized train examples:", len(tokenized_datasets["train"]))

    print(f"Saving tokenized dataset to {tokenized_data_dir}...")
    tokenized_datasets.save_to_disk(tokenized_data_dir)

    # metadata file to track what was saved
    with open(os.path.join(tokenized_data_dir, "metadata.txt"), "w") as f:
        f.write(f"Test mode: {args.test_mode}\n")
        f.write(f"Max examples: {args.max_examples}\n")
        f.write(f"Train size: {len(tokenized_datasets['train'])}\n")
        f.write(f"Test size: {len(tokenized_datasets['test'])}\n")
        f.write(f"Max sequence length: {args.max_length}\n")

    print("Pretokenization done!")

if __name__ == "__main__":
    main()