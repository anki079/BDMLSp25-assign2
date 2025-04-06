# import os
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
# from torch.cuda.amp import autocast, GradScaler
# from accelerate import init_empty_weights
# from transformers import BitsAndBytesConfig
# import time
# import argparse


# class TextDataset(Dataset):
#     def __init__(self, file_path, tokenizer, max_length=256, max_lines=None):
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#         with open(file_path, 'r', encoding='utf-8') as f:
#             lines = [line.strip() for line in f if line.strip()]
#             if max_lines:
#                 lines = lines[:max_lines]
#         self.lines = lines
#         print(f"Loaded {len(self.lines)} examples from {file_path}")
    
#     def __len__(self):
#         return len(self.lines)
    
#     def __getitem__(self, idx):
#         text = self.lines[idx]
#         print(f"Tokenizer type: {type(self.tokenizer)}")  # Debug line
#         if not callable(self.tokenizer):
#             print(f"ERROR: self.tokenizer is not callable: {self.tokenizer}")
#         # Fallback or error handling here
    
#         encodings = self.tokenizer(
#             text, 
#             max_length=self.max_length, 
#             padding="max_length", 
#             truncation=True, 
#             return_tensors="pt"
#         )
        
#         item = {key: val.squeeze(0) for key, val in encodings.items()}
#         item['labels'] = item['input_ids'].clone()
#         return item

# def setup_ddp():
#     """
#     Initializes the default process group using the environment variables 
#     that `torchrun` automatically sets: RANK, LOCAL_RANK, WORLD_SIZE, etc.
#     """
#     # We assume torchrun sets MASTER_ADDR, MASTER_PORT, etc.
#     # If you want to override them, you still can:
#     os.environ.setdefault('MASTER_ADDR', 'localhost')
#     os.environ.setdefault('MASTER_PORT', '29500')
    
#     # Initialize process group using env:// which will read RANK/WORLD_SIZE
#     dist.init_process_group(backend="nccl", init_method="env://")

#     # local_rank is which GPU on the node to use
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)

# def cleanup():
#     dist.destroy_process_group()

# def train(model_path, train_file, test_file, epochs, batch_size):
#     """
#     Main training function. We do NOT pass local_rank or rank as arguments.
#     Instead, we read them inside the function from the environment.
#     """
#     # 1. Initialize the process group
#     setup_ddp()

#     # 2. Determine our local rank (which GPU), global rank (which process)
#     local_rank = int(os.environ["LOCAL_RANK"])
#     rank       = dist.get_rank()
#     world_size = dist.get_world_size()
#     device     = torch.device(f"cuda:{local_rank}")

#     print(f"[Process {rank}] local_rank={local_rank} world_size={world_size}")

#     # 3. Load model/tokenizer
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#         print(f"[Rank {rank}] Tokenizer loaded.")

#         config = LlamaConfig.from_pretrained(model_path)
#         print(f"[Rank {rank}] Model config loaded: {config.hidden_size} hidden size, {config.num_hidden_layers} layers")

#         quant_config = BitsAndBytesConfig(
#             load_in_8bit=True,   # or load_in_4bit=True
#             llm_int8_threshold=6.0
#         )

#         model = LlamaForCausalLM.from_pretrained(
#             model_path,
#             quantization_config=quant_config
#         )
#         print(f"[Rank {rank}] Model loaded with 8 bit quantization.")

#         model.gradient_checkpointing_disable()
#         # print(f"[Rank {rank}] Gradient checkpointing enabled.")

#         total_params = sum(p.numel() for p in model.parameters())
#         print(f"[Rank {rank}] Total parameters: {total_params:,}")
#     except Exception as e:
#         print(f"[Rank {rank}] Error loading model/tokenizer: {e}")
#         cleanup()
#         return

#     # 4. Move model to this process's GPU
#     model.to(device)

#     # 5. Wrap with DistributedDataParallel
#     model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

#     # 6. Load datasets with small max_lines for testing
#     try:
#         train_dataset = TextDataset(train_file, tokenizer, max_lines=20)
#         test_dataset = TextDataset(test_file, tokenizer, max_lines=10)
#     except Exception as e:
#         print(f"[Rank {rank}] Error loading datasets: {e}")
#         cleanup()
#         return

#     # 7. Create sampler + DataLoader
#     train_sampler = DistributedSampler(
#         train_dataset,
#         num_replicas=world_size,
#         rank=rank,
#         shuffle=True
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         sampler=train_sampler,
#         pin_memory=True,
#         num_workers=0
#     )

#     print(f"[Rank {rank}] DataLoader ready. Starting training loop...")

#     # 8. Create optimizer
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

#     scaler = GradScaler()  # added for mixed precision training

#     # gradient accumulation steps
#     accumulation_steps = 4

#     # 9. Training loop (short version)
#     total_time = 0
#     for epoch in range(epochs):
#         epoch_start_time = time.time()
#         model.train()
#         train_sampler.set_epoch(epoch)  # ensures different shuffling each epoch
#         total_loss = 0

#         # Reset gradients at the beginning of each epoch
#         optimizer.zero_grad()
        
#         # We'll limit to 2 steps for a quick test
#         for step, batch in enumerate(train_loader):
#             if step >= 2:
#                 break

#             print(f"[Rank {rank}] Step {step}")
#             # Move batch to device
#             batch = {k: v.to(device) for k, v in batch.items()}

#             # Mixed precision forward pass
#             with autocast():
#                 outputs = model(**batch)
#                 # Scale the loss by accumulation steps for gradient accumulation
#                 loss = outputs.loss / accumulation_steps
            
#             # Mixed precision backward pass
#             scaler.scale(loss).backward()
            
#             # Update weights after accumulation_steps or at the end of the epoch
#             if (step + 1) % accumulation_steps == 0 or (step == len(train_loader) - 1):
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()
            
#             # Track the full loss (not the scaled one)
#             total_loss += outputs.loss.item()
            
#             if rank == 0 and step % 10 == 0:
#                 print(f"Epoch {epoch+1}/{epochs} | Batch {step}/{len(train_loader)} | Loss: {outputs.loss.item():.4f}")
        
#         epoch_time = time.time() - epoch_start_time
#         total_time += epoch_time

#         #     outputs = model(**batch)
#         #     loss = outputs.loss
#         #     total_loss += loss.item()
            
#         #     optimizer.zero_grad()
#         #     loss.backward()
#         #     optimizer.step()
            
#         #     if rank == 0 and step % 10 == 0:
#         #         print(f"Epoch {epoch+1}/{epochs} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
#         # epoch_time = time.time() - epoch_start_time
#         # total_time += epoch_time
        
#         if rank == 0:
#             avg_loss = total_loss / (step+1)  # step+1 because of 0-based
#             print(f"[Rank 0] Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

#     # 10. Evaluation on rank 0
#     if rank == 0:
#         model.eval()
#         ppl = compute_perplexity(model, test_dataset, device)
#         print(f"[Rank 0] Perplexity: {ppl:.2f}")
#         print(f"[Rank 0] Average time per epoch: {total_time/epochs:.2f}s")

#     cleanup()
#     if rank == 0:
#         print("[Rank 0] ✅ Test run completed successfully.")

# def compute_perplexity(model, dataset, device):
#     test_loader = DataLoader(dataset, batch_size=4)
#     model.eval()
#     total_loss = 0
#     total_length = 0
    
#     # with torch.no_grad():
#     #     for batch in test_loader:
#     #         batch = {k: v.to(device) for k, v in batch.items()}
#     #         outputs = model(**batch)
#     #         loss = outputs.loss
#     #         total_loss += loss.item() * batch["input_ids"].size(0)
#     #         total_length += batch["input_ids"].size(0)

#     with torch.no_grad():
#         # Use mixed precision for evaluation too
#         with autocast():
#             for batch in test_loader:
#                 batch = {k: v.to(device) for k, v in batch.items()}
#                 outputs = model(**batch)
#                 loss = outputs.loss
#                 total_loss += loss.item() * batch["input_ids"].size(0)
#                 total_length += batch["input_ids"].size(0)
    
#     avg_loss = total_loss / total_length
#     ppl = torch.exp(torch.tensor(avg_loss)).item()
#     return ppl

# def main():
#     parser = argparse.ArgumentParser()
#     # We do NOT rely on --local_rank from the user. torchrun will set LOCAL_RANK env variable
#     parser.add_argument("--epochs", type=int, default=1)
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--train_file", type=str, required=True)
#     parser.add_argument("--test_file", type=str, required=True)

#     args = parser.parse_args()

#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(current_dir, "llama-hf")
#     train_file = os.path.join(current_dir, args.train_file)
#     test_file = os.path.join(current_dir, args.test_file)

#     train(model_path, train_file, test_file, args.epochs, args.batch_size)

# if __name__ == "__main__":
#     main()

#max length = 128
import os
import torch
import math
import time
import argparse
# import torch.distributed as dist
import bitsandbytes as bnb
# from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk

def main():

    parser = argparse.ArgumentParser(description="Distributed LlaMa fine tuning")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data")
    # parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()
    
    # print logs only on the main process
    is_main_process = args.local_rank in [-1, 0]

    model_dir = "./llama-hf"
    output_dir = "./checkpoints-llama-data-parallelism"
    os.makedirs(output_dir, exist_ok=True)
    tokenized_data_dir = args.tokenized_data_dir

    if is_main_process:
        print("Loading tokenizer for data collator...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    if is_main_process:
        print("Loading tokenized datasets...")
    tokenized_datasets = load_from_disk(tokenized_data_dir)

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    if is_main_process:
        print("Loading model...")
    model = AutoModelForCausalLLM.from_pretrained(model_dir)

    if is_main_process:
        print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5000,
        save_strategy="epoch",
        # save_steps=500,
        bf16=True,
        bf16_full_eval=True,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        learning_rate=5e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        group_by_length=True,
        report_to="none",
        save_total_limit=1
    )

    if is_main_process:
        print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    if is_main_process:
        print("Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    time_per_epoch = (end_time - start_time)/training_args.num_train_epochs
    
    if is_main_process:
        print(f"Time per epoch: {time_per_epoch} seconds")

        print("Saving model...")
        trainer.save_model(output_dir)

        print("Evaluating model...")
        eval_results = trainer.evaluate()
        eval_loss = eval_results["eval_loss"]
        perplexity = math.exp(eval_loss)
        print(f"Eval Loss: {eval_loss}, Perplexity: {perplexity}")
        
        with open(os.path.join(output_dir, "eval_results_DP.txt"), "w") as f:
            f.write("***** Distributed Fine Tuning with Data Parallelism Results *****")
            f.write(f"Time per epoch: {time_per_epoch}\n")
            f.write(f"Eval Loss: {eval_loss}\n")
            f.write(f"Perplexity: {perplexity}\n")
        
        print("Distributed fine-tuning using data parallelism complete!")

if __name__ == "__main__":
    main()