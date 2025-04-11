'''
- Data parallel distributed LlaMa-3.2-3B fine-tuning on climate data
- Approach leverages HuggingFace's Trainer class to handle distributed data parallelism automatically 
    when launched with torchrun or torch.distributed.launch
- Memory optimizations used: gradient checkpointing, bf16, 4bit, lora
'''

import os
import math
import torch
import time
import argparse
import bitsandbytes as bnb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser(description="Data Parallel Fine-Tuning")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4) # effective batch size = 64 for DP with 8 x 4 x 2 GPUs
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data_chunks")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for DDP (set by torchrun)")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    print(f"[RANK {local_rank}] Running in {'distributed' if is_distributed else 'standalone'} mode")   

    print(f"PROCESS INFO: RANK={local_rank}, PID={os.getpid()}")

    is_main_process = (local_rank in [-1, 0])

    tokenized_data_dir = args.tokenized_data_dir
    if tokenized_data_dir == "./tokenized_data_test":
        output_dir = "./checkpoints-llama-data-parallel-test"
    else:
        output_dir = "./checkpoints-llama-data-parallel-mem-opt"
    model_dir = "./llama-hf"
    
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    print(f"[RANK {local_rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[RANK {local_rank}] Loading tokenized datasets from:", args.tokenized_data_dir)
    tokenized_datasets = load_from_disk(args.tokenized_data_dir)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    device_map = {"": local_rank}
    print(f"[RANK {local_rank}] Using device: {device_map}")
    
    print(f"[RANK {local_rank}] Loading 4-bit quantization config...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    print(f"[RANK {local_rank}] Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )

    print(f"[RANK {local_rank}] Model loaded to device {device_map}")

    print(f"[RANK {local_rank}] Preparing model for kbit + enabling gradient checkpointing...")
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print(f"[RANK {local_rank}] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    if is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M "
              f"({100*trainable_params/total_params:.4f}%)")

    print(f"[RANK {local_rank}] Creating data collator...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"[RANK {local_rank}] Setting up TrainingArguments...")
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
        # max_steps=5000,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        learning_rate=2e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        group_by_length=True,
        report_to="none",
        ddp_find_unused_parameters=False
    )


    print(f"[RANK {local_rank}] Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    print(f"[RANK {local_rank}] Starting training (data parallel)...")

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    time_per_epoch = (end_time - start_time)/training_args.num_train_epochs
    if is_main_process:
        print(f"Time per epoch: {time_per_epoch:.2f} seconds")

        print("Saving final model checkpoint...")
        trainer.save_model(output_dir)

    print("Evaluating model for perplexity...")
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss)
    if is_main_process:
        print(f"Eval Loss: {eval_loss}, Perplexity: {perplexity:.2f}")

        with open(os.path.join(output_dir, "eval_results_data_parallel.txt"), "w") as f:
            f.write(f"Time per epoch: {time_per_epoch:.2f}\n")
            f.write(f"Eval Loss: {eval_loss}\n")
            f.write(f"Perplexity: {perplexity:.2f}\n")

    print("Data parallel fine-tuning complete!")

if __name__ == "__main__":
    main()
