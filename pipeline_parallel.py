import os
import math
import time
import torch
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser(description="Pipeline Parallel Fine-Tuning with HuggingFace + DeepSpeed")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data_chunks")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # Set up distributed training environment
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    is_main_process = (local_rank in [-1, 0])
    
    print(f"[RANK {local_rank}] Process info: PID={os.getpid()}")
    
    # Output directory
    output_dir = "./checkpoints-llama-pipeline-parallel"
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and datasets
    model_dir = "./llama-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[RANK {local_rank}] Loading datasets from {args.tokenized_data_dir}...")
    tokenized_datasets = load_from_disk(args.tokenized_data_dir)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    
    # Load model
    print(f"[RANK {local_rank}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16
    )
    
    # Enable gradient checkpointing
    print(f"[RANK {local_rank}] Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ds_config = {
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 0
        },
        "pipeline": {
            "enabled": True,
            "num_stages": 2,
            "pipe_chunk_size": 2
        },
        "optimizer": {
            "type": "adamw_torch"
        },
        "fp16": {
            "enabled": False
        }

    }
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        bf16_full_eval=True,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        learning_rate=2e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        group_by_length=True,
        report_to="none",
        # Use DeepSpeed with pipeline parallelism config
        deepspeed=ds_config
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )
    
    # Train
    print(f"[RANK {local_rank}] Starting training with pipeline parallelism...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    # Calculate time per epoch
    time_per_epoch = (end_time - start_time) / args.epochs
    
    # Evaluate
    print(f"[RANK {local_rank}] Evaluating model...")
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss)
    
    if is_main_process:
        print(f"Time per epoch: {time_per_epoch:.2f} seconds")
        print(f"Eval Loss: {eval_loss}, Perplexity: {perplexity:.2f}")
        
        # Save results
        with open(os.path.join(output_dir, "eval_results_pipeline_parallel.txt"), "w") as f:
            f.write(f"Time per epoch: {time_per_epoch:.2f}\n")
            f.write(f"Eval Loss: {eval_loss}\n")
            f.write(f"Perplexity: {perplexity:.2f}\n")
    
    print("Pipeline parallel fine-tuning complete!")

if __name__ == "__main__":
    main()