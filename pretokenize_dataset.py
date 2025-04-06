import os
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

def main():
    data_files = {"train": "assign-2/train.txt", "test": "assign-2/test.txt"}
    tokenized_data_dir = "./tokenized_data"
    model_dir = "./llama-hf"
    max_length = 128
    
    if os.path.exists(tokenized_data_dir):
        print(f"Tokenized dataset already exists at {tokenized_data_dir}")
        return
    
    print("Loading raw dataset...")
    dataset = load_dataset("text", data_files=data_files)
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
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
    print("Pretokenization done!")

if __name__ == "__main__":
    main()