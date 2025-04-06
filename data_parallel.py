import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import time
import argparse

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]
            
        print(f"Loaded {len(self.lines)} examples from {file_path}")
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        text = self.lines[idx]
        encodings = self.tokenizer(text, max_length=self.max_length, padding="max_length", 
                                  truncation=True, return_tensors="pt")
        
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        
        item['labels'] = item['input_ids'].clone()
        
        return item

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # set device
    torch.cuda.set_device(rank)
    
def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, model_path, train_file, test_file, epochs, batch_size):
    # setup process group
    setup(rank, world_size)
    
    if rank == 0:
        print(f"Starting distributed training on {world_size} GPUs")
    
    # load model and tokenizer
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        if rank == 0:
            print("Tokenizer loaded successfully")
        
        config = LlamaConfig.from_pretrained(model_path)
        if rank == 0:
            print(f"Model config loaded: {config.hidden_size} hidden size, {config.num_hidden_layers} layers")
            
        model = LlamaForCausalLM.from_pretrained(model_path)
        if rank == 0:
            print("Model loaded successfully")
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")
    except Exception as e:
        print(f"Error loading model: {e}")
        cleanup()
        return
    
    # move model to the appropriate device
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    
    # wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # load datasets
    try:
        train_dataset = TextDataset(train_file, tokenizer)
        test_dataset = TextDataset(test_file, tokenizer)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        cleanup()
        return
    
    # create sampler and dataloader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4
    )
    
    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # training loop
    total_time = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            # move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # fwd pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # bckwd pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # display progress
            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time += epoch_time
        
        if rank == 0:  # only print metrics on the main process
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # eval
    if rank == 0:
        model.eval()
        perplexity = compute_perplexity(model, test_dataset, device)
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Average time per epoch: {total_time/epochs:.2f}s")
    
    cleanup()

def compute_perplexity(model, dataset, device):
    test_loader = DataLoader(dataset, batch_size=4)
    model.eval()
    total_loss = 0
    total_length = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_length += batch["input_ids"].size(0)
    
    # compute perplexity
    avg_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    
    # get the number of available GPUs
    world_size = torch.cuda.device_count()
    
    # file paths relative to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "llama-hf")
    train_file = os.path.join(current_dir, "train.txt")
    test_file = os.path.join(current_dir, "test.txt")
    
    train(args.local_rank, world_size, model_path, train_file, test_file, args.epochs, args.batch_size)

if __name__ == "__main__":
    main()