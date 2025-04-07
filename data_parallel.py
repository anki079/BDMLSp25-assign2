# data_parallel.py
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
    parser = argparse.ArgumentParser(description="Data Parallel Fine-Tuning with 4-bit + LoRA")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP (set by torchrun)")
    args = parser.parse_args()

    # Only rank 0 prints logs
    is_main_process = (args.local_rank in [-1, 0])

    tokenized_data_dir = args.tokenized_data_dir
    if tokenized_data_dir == "./tokenized_data_test":
        output_dir = "./checkpoints-llama-data-parallel-test"
    else:
        output_dir = "./checkpoints-llama-data-parallel"
    model_dir = "./llama-hf"
    os.makedirs(output_dir, exist_ok=True)

    if is_main_process:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        print("Loading tokenized datasets from:", args.tokenized_data_dir)
    tokenized_datasets = load_from_disk(args.tokenized_data_dir)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    local_rank_env = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank_env}
    
    if is_main_process:
        print("Loading 4-bit quantized base model (data parallel)...")

    # 4-bit quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )

    if is_main_process:
        print("Preparing model for k-bit training + enabling gradient checkpointing...")
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if is_main_process:
        print("Applying LoRA adapters...")
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if is_main_process:
        print("Setting up TrainingArguments...")
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
        logging_steps=200,
        save_strategy="epoch",
        save_total_limit=1,
        # max_steps=5000,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        group_by_length=True,
        report_to="none",
        ddp_find_unused_parameters=False
    )

    if is_main_process:
        print("Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    if is_main_process:
        print("Starting training (data parallel, 4-bit + LoRA + gradient checkpointing)...")

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
        print(f"Eval Loss: {eval_loss}, Perplexity: {perplexity:.2f}")

        with open(os.path.join(output_dir, "eval_results_data_parallel.txt"), "w") as f:
            f.write(f"Time per epoch: {time_per_epoch:.2f}\n")
            f.write(f"Eval Loss: {eval_loss}\n")
            f.write(f"Perplexity: {perplexity:.2f}\n")

        print("Data parallel (4-bit + LoRA) fine-tuning complete!")

if __name__ == "__main__":
    main()
