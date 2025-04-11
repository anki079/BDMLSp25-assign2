"""
- Pipeline Parallel LLaMA fine-tuning on climate data
- Approach uses DeepSpeed to implement pipeline parallelism
- Memory optimizations used: lora, 4bit, gradient checkpointing, bf16
- Using the same chunked dataset and hyperparams as the data parallel code
"""

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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def main():
    parser = argparse.ArgumentParser(description="Pipeline Parallel Fine-Tuning with HuggingFace + DeepSpeed")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8) # effective batch size = 64 for PP with 8 x 8
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data_chunks")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = (local_rank in [-1, 0])
    
    print(f"[RANK {local_rank}] Process info: PID={os.getpid()}, world_size={world_size}")
    
    output_dir = "./checkpoints-llama-pipeline-parallel"
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    model_dir = "./llama-hf"
    print(f"[RANK {local_rank}] Loading tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[RANK {local_rank}] Loading datasets from {args.tokenized_data_dir}...")
    tokenized_datasets = load_from_disk(args.tokenized_data_dir)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    
    print(f"[RANK {local_rank}] Train size = {len(train_dataset)}, Test size = {len(test_dataset)}")
    
    print(f"[RANK {local_rank}] Loading 4-bit quantization config...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"[RANK {local_rank}] Loading model from {model_dir} in 4bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config
    )
    
    print(f"[RANK {local_rank}] Preparing model for LoRA + kbit + enabling gradient checkpointing...")
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    # pipeline parallel deepspeed config
    ds_config = {
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 0
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 2
        },
        "pipeline": {
            "enabled": True,
            "pipeline_parallel_size": 2,
            "schedule": "1F1B",
            "chunks": 4
        },
        "fp16": {
            "enabled": False
        },
        "wall_clock_breakdown": False

    }

    if is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")
    
    print(f"[RANK {local_rank}] Creating data collator...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    print(f"[RANK {local_rank}] Setting up TrainingArguments with DeepSpeed")
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
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        learning_rate=2e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        group_by_length=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        deepspeed=ds_config
    )
    
    print(f"[RANK {local_rank}] Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )
    
    print(f"[RANK {local_rank}] Starting training (Deepspeed pipeline parallel)...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    time_per_epoch = (end_time - start_time)/training_args.num_train_epochs
    if is_main_process:
        print(f"Time per epoch: {time_per_epoch:.2f} seconds")
        print("Saving final model checkpoint...")
        trainer.save_model(output_dir)

    print(f"[RANK {local_rank}] Evaluating model for perplexity...")
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss)
    if is_main_process:
        print(f"Eval Loss: {eval_loss}, Perplexity: {perplexity:.2f}")
        
        with open(os.path.join(output_dir, "eval_results_pipeline_parallel.txt"), "w") as f:
            f.write(f"Time per epoch: {time_per_epoch:.2f}\n")
            f.write(f"Eval Loss: {eval_loss}\n")
            f.write(f"Perplexity: {perplexity:.2f}\n")
    
    print("Pipeline parallel fine-tuning complete!")

if __name__ == "__main__":
    main()