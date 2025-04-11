import os
import math
import torch
import argparse
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaMA model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="Path to the model checkpoint directory")
    parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data_chunks",
                        help="Path to tokenized dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading model from checkpoint: {args.checkpoint_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("Loading tokenizer...")
    model_dir = "./llama-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading test dataset from: {args.tokenized_data_dir}")
    tokenized_datasets = load_from_disk(args.tokenized_data_dir)
    test_dataset = tokenized_datasets["test"]
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator
    )
    
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    total_steps = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            total_steps += 1
            
            if step % 10 == 0:
                print(f"Step {step}/{len(test_dataloader)}, Current loss: {loss.item():.4f}")
    
    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    
    print(f"\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    
    results_file = os.path.join(args.checkpoint_dir, "eval_results_data_parallel.txt")
    with open(results_file, "w") as f:
        f.write("Evaluation Results:\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Perplexity: {perplexity:.2f}\n")
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()