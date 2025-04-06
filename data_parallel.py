# import os
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
# import time
# import argparse

# class TextDataset(Dataset):
#     def __init__(self, file_path, tokenizer, max_length=512, max_lines=None):
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
#         encodings = self.tokenizer(text, max_length=self.max_length, padding="max_length", 
#                                   truncation=True, return_tensors="pt")
        
#         item = {key: val.squeeze(0) for key, val in encodings.items()}
        
#         item['labels'] = item['input_ids'].clone()
        
#         return item

# def setup(local_rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '29500'
    
#     # initialize the process group
#     dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    
#     # set device
#     torch.cuda.set_device(local_rank)
    
# def cleanup():
#     dist.destroy_process_group()

# def train(local_rank, world_size, model_path, train_file, test_file, epochs, batch_size):
#     # setup process group
#     setup(local_rank, world_size)
#     rank = dist.get_rank()  # Proper rank after process group initialized
#     device = torch.device(f"cuda:{local_rank}")

#     print(f"[Rank {rank}] Starting training setup...")

#     # if rank == 0:
#     #     print(f"Starting distributed training on {world_size} GPUs")
    
#     # load model and tokenizer
#     try:
#         tokenizer = LlamaTokenizer.from_pretrained(model_path)
#         # if rank == 0:
#         #     print("Tokenizer loaded successfully")
#         print(f"[Rank {rank}] Tokenizer loaded.")

#         config = LlamaConfig.from_pretrained(model_path)
#         # if rank == 0:
#         #     print(f"Model config loaded: {config.hidden_size} hidden size, {config.num_hidden_layers} layers")
#         print(f"[Rank {rank}] Model config loaded: {config.hidden_size} hidden size, {config.num_hidden_layers} layers")    
#         model = LlamaForCausalLM.from_pretrained(model_path)
#         print(f"[Rank {rank}] Model loaded.")
#         total_params = sum(p.numel() for p in model.parameters())
#         print(f"Total parameters: {total_params:,}")
#         # if rank == 0:
#         #     print("Model loaded successfully")
#         #     total_params = sum(p.numel() for p in model.parameters())
#         #     print(f"Total parameters: {total_params:,}")
#     except Exception as e:
#         print(f"[Rank {rank}] Error loading model: {e}")
#         cleanup()
#         return
    
#     # move model to the appropriate device
#     # device = torch.device(f"cuda:{rank}")
#     model = model.to(device)
    
#     # wrap model with DDP
#     model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
#     # load datasets
#     try:
#         train_dataset = TextDataset(train_file, tokenizer, max_lines=20)
#         test_dataset = TextDataset(test_file, tokenizer, max_lines=10)
#     except Exception as e:
#         print(f"[Rank {rank}] Error loading datasets: {e}")
#         cleanup()
#         return
    
#     # create sampler and dataloader
#     train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         sampler=train_sampler,
#         pin_memory=True,
#         num_workers=0 # down from 4
#     )
    
#     print(f"[Rank {rank}] DataLoader ready. Beginning training loop...")

#     # setup optimizer
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

#     # training loop
#     total_time = 0
#     for epoch in range(epochs):
#         epoch_start_time = time.time()
#         model.train()
#         train_sampler.set_epoch(epoch)
#         total_loss = 0
        
#         for step, batch in enumerate(train_loader):
#             if step >= 2:
#                 break #limit batches for testing
#             print(f"[Rank {rank}] Step {step}")
#             # move batch to device
#             batch = {k: v.to(device) for k, v in batch.items()}
            
#             # fwd pass
#             outputs = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 labels=batch["labels"]
#             )
            
#             loss = outputs.loss
#             total_loss += loss.item()
            
#             # bckwd pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             # display progress
#             if rank == 0 and step % 10 == 0:
#                 print(f"Epoch {epoch+1}/{epochs} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
#         epoch_end_time = time.time()
#         epoch_time = epoch_end_time - epoch_start_time
#         total_time += epoch_time
        
#         if rank == 0:  # only print metrics on the main process
#             avg_loss = total_loss / len(train_loader)
#             print(f"Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
#     # eval
#     if rank == 0:
#         model.eval()
#         perplexity = compute_perplexity(model, test_dataset, device)
#         print(f"Perplexity: {perplexity:.2f}")
#         print(f"Average time per epoch: {total_time/epochs:.2f}s")
    
#     cleanup()
#     if rank == 0:
#         print("✅ Test run completed successfully.")


# def compute_perplexity(model, dataset, device):
#     test_loader = DataLoader(dataset, batch_size=4)
#     model.eval()
#     total_loss = 0
#     total_length = 0
    
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 labels=batch["labels"]
#             )
            
#             loss = outputs.loss
#             total_loss += loss.item() * batch["input_ids"].size(0)
#             total_length += batch["input_ids"].size(0)
    
#     # compute perplexity
#     avg_loss = total_loss / total_length
#     perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
#     return perplexity

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", type=int, default=0)
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--batch_size", type=int, default=2)
#     parser.add_argument("--train_file", type=str, required=True)
#     parser.add_argument("--test_file", type=str, required=True)

#     args = parser.parse_args()
    
#     # get the number of available GPUs
#     world_size = torch.cuda.device_count()
    
#     # file paths relative to the current directory
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(current_dir, "llama-hf")
#     train_file = os.path.join(current_dir, args.train_file)
#     test_file = os.path.join(current_dir, args.test_file)
    
#     train(args.local_rank, world_size, model_path, train_file, test_file, args.epochs, args.batch_size)

# if __name__ == "__main__":
#     main()



#########################
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
import time
import argparse

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, max_lines=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            if max_lines:
                lines = lines[:max_lines]
        self.lines = lines
        print(f"Loaded {len(self.lines)} examples from {file_path}")
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        text = self.lines[idx]
        print(f"Tokenizer type: {type(self.tokenizer)}")  # Debug line
        if not callable(self.tokenizer):
            print(f"ERROR: self.tokenizer is not callable: {self.tokenizer}")
        # Fallback or error handling here
    
        encodings = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

def setup_ddp():
    """
    Initializes the default process group using the environment variables 
    that `torchrun` automatically sets: RANK, LOCAL_RANK, WORLD_SIZE, etc.
    """
    # We assume torchrun sets MASTER_ADDR, MASTER_PORT, etc.
    # If you want to override them, you still can:
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')
    
    # Initialize process group using env:// which will read RANK/WORLD_SIZE
    dist.init_process_group(backend="nccl", init_method="env://")

    # local_rank is which GPU on the node to use
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

def train(model_path, train_file, test_file, epochs, batch_size):
    """
    Main training function. We do NOT pass local_rank or rank as arguments.
    Instead, we read them inside the function from the environment.
    """
    # 1. Initialize the process group
    setup_ddp()

    # 2. Determine our local rank (which GPU), global rank (which process)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{local_rank}")

    print(f"[Process {rank}] local_rank={local_rank} world_size={world_size}")

    # 3. Load model/tokenizer
    try:
        tokenizer =AutoTokenizer.from_pretrained(model_path)
        print(f"[Rank {rank}] Tokenizer loaded.")

        config = LlamaConfig.from_pretrained(model_path)
        print(f"[Rank {rank}] Model config loaded: {config.hidden_size} hidden size, {config.num_hidden_layers} layers")

        model = LlamaForCausalLM.from_pretrained(model_path)
        print(f"[Rank {rank}] Model loaded.")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Rank {rank}] Total parameters: {total_params:,}")
    except Exception as e:
        print(f"[Rank {rank}] Error loading model/tokenizer: {e}")
        cleanup()
        return

    # 4. Move model to this process's GPU
    model.to(device)

    # 5. Wrap with DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # 6. Load datasets with small max_lines for testing
    try:
        train_dataset = TextDataset(train_file, tokenizer, max_lines=20)
        test_dataset = TextDataset(test_file, tokenizer, max_lines=10)
    except Exception as e:
        print(f"[Rank {rank}] Error loading datasets: {e}")
        cleanup()
        return

    # 7. Create sampler + DataLoader
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=0
    )

    print(f"[Rank {rank}] DataLoader ready. Starting training loop...")

    # 8. Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # 9. Training loop (short version)
    total_time = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_sampler.set_epoch(epoch)  # ensures different shuffling each epoch
        total_loss = 0
        
        # We'll limit to 2 steps for a quick test
        for step, batch in enumerate(train_loader):
            if step >= 2:
                break

            print(f"[Rank {rank}] Step {step}")
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        
        if rank == 0:
            avg_loss = total_loss / (step+1)  # step+1 because of 0-based
            print(f"[Rank 0] Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    # 10. Evaluation on rank 0
    if rank == 0:
        model.eval()
        ppl = compute_perplexity(model, test_dataset, device)
        print(f"[Rank 0] Perplexity: {ppl:.2f}")
        print(f"[Rank 0] Average time per epoch: {total_time/epochs:.2f}s")

    cleanup()
    if rank == 0:
        print("[Rank 0] ✅ Test run completed successfully.")

def compute_perplexity(model, dataset, device):
    test_loader = DataLoader(dataset, batch_size=4)
    model.eval()
    total_loss = 0
    total_length = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_length += batch["input_ids"].size(0)
    
    avg_loss = total_loss / total_length
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl

def main():
    parser = argparse.ArgumentParser()
    # We do NOT rely on --local_rank from the user. torchrun will set LOCAL_RANK env variable
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)

    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "llama-hf")
    train_file = os.path.join(current_dir, args.train_file)
    test_file = os.path.join(current_dir, args.test_file)

    train(model_path, train_file, test_file, args.epochs, args.batch_size)

if __name__ == "__main__":
    main()
