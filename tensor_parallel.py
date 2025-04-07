# tensor_parallel.py
import os
import math
import time
import torch
import torch.distributed as dist
from torch.distributed.tensor.parallel import parallelize_module
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW

def main():
    """
    Minimal conceptual example of tensor parallel training with PyTorch’s new APIs.
    This likely requires PyTorch nightly. Actual usage may differ!
    """
    
    ####################################
    # 1. Initialize Distributed Process
    ####################################
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else rank
    world_size = dist.get_world_size()

    # Each process (GPU) uses local_rank as its CUDA device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Only rank 0 will print logs
    is_main_process = (rank == 0)

    ####################################
    # 2. Load Tokenizer + Dataset
    ####################################
    model_dir = "./llama-hf"
    tokenized_data_dir = "./tokenized_data"
    
    if is_main_process:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        print("Loading dataset from disk...")
    tokenized_datasets = load_from_disk(tokenized_data_dir)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    
    # Convert datasets to DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=8)

    ####################################
    # 3. Load Model + Parallelize
    ####################################
    if is_main_process:
        print("Loading base LLaMA model...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    # Move model to local rank
    model.cuda(device)

    # *Experimental* PyTorch Tensor Parallel API
    # parallel_mode="column" or "row" are typical for large linear layers
    parallelize_module(model, parallel_mode="column", devices=[0, 1])
    
    # At this point, large layers (e.g. linear layers) are split across GPU 0 and GPU 1.
    # You should not wrap the model in DistributedDataParallel again — that would combine
    # data parallel + tensor parallel. If your assignment only wants pure tensor parallel,
    # do not add a DDP wrapper.

    ####################################
    # 4. Define Optimizer / Training Loop
    ####################################
    optimizer = AdamW(model.parameters(), lr=5e-4)
    epochs = 1

    if is_main_process:
        print("Starting tensor-parallel training...")

    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        # Basic training loop (no HF Trainer)
        for step, batch in enumerate(train_loader):
            # batch has "input_ids", "attention_mask", etc.
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if is_main_process and step % 100 == 0:
                print(f"Epoch {epoch}, step {step}, loss = {loss.item()}")

        end_time = time.time()
        if is_main_process:
            print(f"Epoch {epoch} finished. Time for epoch = {end_time - start_time:.2f} seconds")

    # Evaluate perplexity on test set
    if is_main_process:
        print("Evaluating model perplexity...")
    model.eval()
    total_loss, total_steps = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            total_loss += outputs.loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)

    if is_main_process:
        print(f"Test Loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}")

    dist.destroy_process_group()
    if is_main_process:
        print("Tensor parallel training complete!")

if __name__ == "__main__":
    main()
