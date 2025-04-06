import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
import argparse

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        text = self.lines[idx]
        encodings = self.tokenizer(text, max_length=self.max_length, padding="max_length", 
                                  truncation=True, return_tensors="pt")
        
        # Remove the batch dimension added by the tokenizer
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        return item

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, model_path, train_file, test_file, epochs):
    # Setup process group
    setup(rank, world_size)
    
    # Load model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    
    # Move model to the appropriate device
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Load datasets
    train_dataset = TextDataset(train_file, tokenizer)
    test_dataset = TextDataset(test_file, tokenizer)
    
    # Create sampler and dataloader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    total_time = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(input_ids=batch["input_ids"], 
                           attention_mask=batch["attention_mask"], 
                           labels=batch["input_ids"])
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time += epoch_time
        
        if rank == 0:  # Only print metrics on the main process
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")
    
    # Evaluation
    if rank == 0:
        model.eval()
        perplexity = compute_perplexity(model, test_dataset, device)
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Average time per epoch: {total_time/epochs:.2f}s")
    
    cleanup()

def compute_perplexity(model, dataset, device):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=4)
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], 
                           attention_mask=batch["attention_mask"], 
                           labels=batch["input_ids"])
            
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_tokens += batch["input_ids"].size(0)
    
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    model_path = "./llama-hf"
    train_file = "./train.txt"
    test_file = "./test.txt"
    
    train(args.local_rank, world_size, model_path, train_file, test_file, args.epochs)

if __name__ == "__main__":
    main()