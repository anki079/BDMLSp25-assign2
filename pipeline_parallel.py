'''
Pipeline Parallel LLaMA Fine-Tuning using DeepSpeed
- Splits the LLaMA model across 2 GPUs using DeepSpeed's pipeline parallelism
- Uses the same hyperparameters as data parallelism for fair comparison
- Optimizations: Gradient checkpointing, bf16 precision
'''

import os
import math
import time
import torch
import argparse
import deepspeed
import bitsandbytes as bnb
import torch.distributed as dist

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)

def main():
    parser = argparse.ArgumentParser(description="Pipeline Parallel Fine-Tuning with DeepSpeed")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data_chunks")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank from torch.distributed.launch")
    args = parser.parse_args()

    # DeepSpeed initializes the distributed environment
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    
    print(f"Process Info: RANK={local_rank}, PID={os.getpid()}")
    
    output_dir = "./checkpoints-llama-pipeline-parallel-deepspeed"
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Checkpoints + results will be saved to: {output_dir}")

    model_dir = "./llama-hf"
    print(f"[RANK {local_rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[RANK {local_rank}] Loading tokenized datasets from: {args.tokenized_data_dir}")
    tokenized_datasets = load_from_disk(args.tokenized_data_dir)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    
    num_train_examples = len(train_dataset)
    print(f"[RANK {local_rank}] Num Train Examples = {num_train_examples}, Num Test Examples = {len(test_dataset)}")

    print(f"[RANK {local_rank}] Loading base LLaMA model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16
    )
    
    print(f"[RANK {local_rank}] Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    # deepspeed config for pipeline parallelism
    ds_config = {
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps * 2,  # 2 GPUs
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 0
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": int(num_train_examples / (args.batch_size * 2 * args.gradient_accumulation_steps) * args.epochs * args.warmup_ratio),
                "total_num_steps": int(num_train_examples / (args.batch_size * 2 * args.gradient_accumulation_steps) * args.epochs)
            }
        },
        "gradient_clipping": 1.0,
        "pipeline": {
            "stages": 2,
            "activation_checkpoint_interval": 1
        }
    }
    
    # deepspeed engine for pipeline parallelism
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        training_data=train_dataset
    )
    
    print(f"[RANK {local_rank}] Starting pipeline-parallel training for {args.epochs} epoch(s)...")
    total_train_start = time.time()
    
    for epoch in range(args.epochs):
        if local_rank == 0:
            print(f"Starting epoch {epoch}...")
        epoch_start = time.time()
        
        model_engine.train_batch()
        
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        
        if local_rank == 0:
            print(f"Epoch {epoch} finished. Time per epoch={epoch_time:.2f}s")
            
            epoch_ckpt_dir = os.path.join(output_dir, f"epoch-{epoch}")
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            print(f"Saving model checkpoint after epoch {epoch} to {epoch_ckpt_dir}...")
            model_engine.save_checkpoint(epoch_ckpt_dir)
    
    total_train_end = time.time()
    total_train_time = total_train_end - total_train_start
    time_per_epoch = total_train_time / args.epochs
    
    if local_rank == 0:
        print(f"Total training time={total_train_time:.2f}s => ~{time_per_epoch:.2f}s per epoch")
    
    if local_rank == 0:
        print("Evaluating model on test set for perplexity...")
        model_engine.eval()
        
        total_loss_eval = 0.0
        total_eval_steps = 0
        
        with torch.no_grad():
            for i in range(0, len(test_dataset), args.batch_size):
                batch = test_dataset[i:i+args.batch_size]
                input_ids = batch["input_ids"].to(model_engine.device)
                labels = input_ids.clone()
                
                outputs = model_engine(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                total_loss_eval += loss.item()
                total_eval_steps += 1
        
        avg_eval_loss = total_loss_eval / total_eval_steps if total_eval_steps > 0 else float("inf")
        perplexity = math.exp(avg_eval_loss) if avg_eval_loss < 20 else float("inf")
        
        print(f"Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        # Write results
        results_path = os.path.join(output_dir, "eval_results_pipeline_parallel_deepspeed.txt")
        print(f"Writing results to {results_path}")
        with open(results_path, "w") as f:
            f.write(f"Time per epoch: {time_per_epoch:.2f}\n")
            f.write(f"Eval Loss: {avg_eval_loss:.4f}\n")
            f.write(f"Perplexity: {perplexity:.2f}\n")
    
    print("Pipeline parallel fine-tuning complete!")

if __name__ == "__main__":
    main()