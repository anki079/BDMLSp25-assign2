'''
- Tensor parallel distributed LlaMa-3.2-3B fine-tuning on climate data
- 
- Memory optimizations used: Gradient checkpointing, bf16
'''

import os
import math
import time
import torch
import argparse
import bitsandbytes as bnb
from torch.distributed.tensor.parallel import parallelize_module
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup # lr for scheduler

def main():
    parser = argparse.ArgumentParser(description="Tensor Parallel Fine-Tuning (matching data parallel hyperparams)")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data_chunks")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=100)
    args = parser.parse_args()

    # single process 2 gpus
    print("*** Tensor Parallel Fine-Tuning ***")
    print(f"Process Info: PID={os.getpid()} - single process controlling GPUs [0,1].")
    
    output_dir = "./checkpoints-llama-tensor-parallel"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Checkpoints + results will be saved to: {output_dir}")

    model_dir = "./llama-hf"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_data_dir = "./tokenized_data_chunks"
    print(f"Loading tokenized datasets from: {tokenized_data_dir}")
    tokenized_datasets = load_from_disk(tokenized_data_dir)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    
    #convert to dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size)

    num_train_examples = len(train_dataset)
    num_test_examples = len(test_dataset)
    print(f"Num Train Examples = {num_train_examples}, Num Test Examples = {num_test_examples}")
    
    print("Loading base LLaMA model...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    print("Converting model to bf16 + enabling gradient checkpointing...")
    model = model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # move model to GPU before parallelizing
    model.cuda(0)

    # tensor parallelism
    print("Applying tensor parallel across GPUs [0,1] (parallel_mode='column')...")
    parallelize_module(model, "column", devices=[0, 1])

    # optimizer + lr scheduler setup
    optimizer = bnb.optim.PagedAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # replicating "cosine" schedule
    # total_steps = total training steps = (num_batches_per_epoch * epochs)
    # num_batches_per_epoch = len(train_loader) = (num_train_examples / batch_size)
    # using gradient_accumulation effectively each "accum" cycle is 1 step in HF
    steps_per_epoch = math.ceil(num_train_examples / (args.batch_size * 1.0))  # ignoring shuffle, approximate
    effective_steps_per_epoch = steps_per_epoch / args.gradient_accumulation_steps
    total_steps = int(effective_steps_per_epoch * args.epochs)

    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Using Cosine LR scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # trainign loop
    print(f"Starting training for {args.epochs} epoch(s) with gradient_accumulation={args.gradient_accumulation_steps} ...")
    total_train_start = time.time()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()

        total_loss = 0.0
        total_steps_in_epoch = 0
        accum_counter = 0

        for step, batch in enumerate(train_loader):
            # move batch to GPU0 
            input_ids = batch["input_ids"].cuda(0)
            labels = input_ids.clone()

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps  # scale by grad_accum for correct accumulation
            loss.backward()

            accum_counter += 1
            total_loss += loss.item()
            if accum_counter == args.gradient_accumulation_steps:
                # update
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                total_steps_in_epoch += 1
                accum_counter = 0

                # logging
                if (global_step % args.logging_steps) == 0:
                    avg_loss = (total_loss * args.gradient_accumulation_steps) / (total_steps_in_epoch) 
                    # "total_loss" was scaled so multiply back
                    print(f"Epoch={epoch}, global_step={global_step}, avg_loss={avg_loss:.4f}")

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        # recompute average loss for final epoch stats
        avg_epoch_loss = 0.0
        if total_steps_in_epoch > 0:
            avg_epoch_loss = (total_loss * args.gradient_accumulation_steps) / total_steps_in_epoch

        print(f"Epoch {epoch} complete. Time per epoch={epoch_time:.2f}s, avg_loss={avg_epoch_loss:.4f}")

        # save checkpoint after each epoch
        epoch_ckpt_dir = os.path.join(output_dir, f"epoch-{epoch}")
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        print(f"Saving model checkpoint after epoch {epoch} to {epoch_ckpt_dir}...")
        model.save_pretrained(epoch_ckpt_dir)

    total_train_end = time.time()
    total_train_time = total_train_end - total_train_start
    time_per_epoch = total_train_time / args.epochs
    print(f"Total training time={total_train_time:.2f}s => ~{time_per_epoch:.2f}s per epoch")

    # eval
    print("Evaluating model on test set for perplexity...")
    model.eval()
    total_loss_eval = 0.0
    total_eval_steps = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].cuda(0)
            labels = input_ids.clone()

            outputs = model(input_ids, labels=labels)
            total_loss_eval += outputs.loss.item()
            total_eval_steps += 1

    avg_eval_loss = total_loss_eval / total_eval_steps if total_eval_steps > 0 else float("inf")
    perplexity = math.exp(avg_eval_loss) if avg_eval_loss < 20 else float("inf")

    print(f"Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")

    # write results file
    results_path = os.path.join(output_dir, "eval_results_tensor_parallel.txt")
    print(f"Writing results to {results_path}")
    with open(results_path, "w") as f:
        f.write(f"Time per epoch: {time_per_epoch:.2f}\n")
        f.write(f"Eval Loss: {avg_eval_loss:.4f}\n")
        f.write(f"Perplexity: {perplexity:.2f}\n")

    print("Tensor parallel fine-tuning complete!")

if __name__ == "__main__":
    main()
