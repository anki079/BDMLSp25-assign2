import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import time
import argparse

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]
            
        print(f"Loaded {len(self.lines)} examples from {file_path}")
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        text = self.lines[idx]
        encodings = self.tokenizer(text, max_length=self.max_length, padding="max_length", 
                                  truncation=True, return_tensors="pt")
        
        # Remove the batch dimension added by the tokenizer
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        
        # Add labels for causal language modeling
        item['labels'] = item['input_ids'].clone()
        
        return item

def setup(rank, num_gpus):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=num_gpus)
    
    # Set device
    torch.cuda.set_device(rank)
    
def cleanup():
    dist.destroy_process_group()

class TensorParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, rank, num_gpus, bias=True):
        super().__init__()
        self.rank = rank
        self.num_gpus = num_gpus
        
        # Split the output dimension
        self.out_features_per_gpu = out_features // num_gpus
        if rank == num_gpus - 1:  # Last GPU might have more features
            self.out_features_per_gpu = out_features - (num_gpus - 1) * self.out_features_per_gpu
            
        # Create a local linear layer
        self.linear = torch.nn.Linear(in_features, self.out_features_per_gpu, bias=bias)
    
    def forward(self, x):
        # Local computation
        local_output = self.linear(x)
        
        # Gather outputs from all GPUs
        output_list = [torch.zeros_like(local_output) for _ in range(self.num_gpus)]
        dist.all_gather(output_list, local_output)
        
        # Concatenate along the output dimension
        return torch.cat(output_list, dim=-1)

def apply_tensor_parallelism(model, rank, num_gpus):
    """Apply tensor parallelism to selected linear layers of the model."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.out_features > 1024:
            # Replace large linear layers with tensor-parallel versions
            # Get parent module
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            parent = model if parent_name == '' else model.get_submodule(parent_name)
            
            # Create tensor-parallel layer
            tp_layer = TensorParallelLinear(
                module.in_features, 
                module.out_features,
                rank,
                num_gpus,
                bias=(module.bias is not None)
            )
            
            # Replace layer
            setattr(parent, child_name, tp_layer)
            
            if rank == 0:
                print(f"Replaced {name} with tensor parallel version")
    
    return model

def train(rank, num_gpus, model_path, train_file, test_file, epochs, batch_size):
    # Setup process group
    setup(rank, num_gpus)
    
    if rank == 0:
        print(f"Starting tensor parallel training on {num_gpus} GPUs")
    
    # Load model and tokenizer
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
    
    # Apply tensor parallelism
    model = apply_tensor_parallelism(model, rank, num_gpus)
    
    # Move model to the appropriate device
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    
    # Load datasets
    try:
        train_dataset = TextDataset(train_file, tokenizer)
        test_dataset = TextDataset(test_file, tokenizer)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        cleanup()
        return
    
    # Create dataloader - no need for DistributedSampler as each GPU has the full model
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # Training loop
    total_time = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Synchronize gradients across GPUs
            for param in model.parameters():
                if param.requires_grad:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= num_gpus
            
            optimizer.step()
            
            # Print progress
            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time += epoch_time
        
        if rank == 0:  # Only print metrics on the main process
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # Evaluation
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
    
    # Compute perplexity
    avg_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    # Set file paths relative to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "llama-hf")
    train_file = os.path.join(current_dir, "train.txt")
    test_file = os.path.join(current_dir, "test.txt")
    
    train(args.local_rank, num_gpus, model_path, train_file, test_file, args.epochs, args.batch_size)

if __name__ == "__main__":
    main()