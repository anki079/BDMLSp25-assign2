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

class PipelineStage(torch.nn.Module):
    """A module representing a stage in the pipeline."""
    def __init__(self, stage_id, num_stages, model_parts, hidden_size):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.model_parts = model_parts
        self.hidden_size = hidden_size
    
    def forward(self, batch_dict=None, hidden_states=None):
        if self.stage_id == 0:
            # First stage - takes input_ids and returns hidden states
            outputs = self.model_parts(
                input_ids=batch_dict["input_ids"],
                attention_mask=batch_dict["attention_mask"]
            )
            return outputs.hidden_states
            
        elif self.stage_id == self.num_stages - 1:
            # Last stage - takes hidden states and computes logits/loss
            logits = self.model_parts(hidden_states)
            
            if batch_dict is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch_dict["labels"][..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return logits, loss
            
            return logits, None
            
        else:
            # Middle stage - processes hidden states
            return self.model_parts(hidden_states)

def create_pipeline_stages(model, num_gpus):
    """Split a LLaMA model into pipeline stages."""
    if num_gpus != 2:
        raise ValueError("This implementation only supports 2 GPUs")
    
    # Get model components
    embedding_layer = model.model.embed_tokens
    layers = model.model.layers
    num_layers = len(layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head
    hidden_size = model.config.hidden_size
    
    # For 2 GPUs:
    # Stage 0: Embeddings + First half of layers
    # Stage 1: Second half of layers + Norm + LM head
    
    # Create first stage
    first_stage = torch.nn.Sequential()
    first_stage.add_module("embeddings", embedding_layer)
    for i in range(num_layers // 2):
        first_stage.add_module(f"layer_{i}", layers[i])
    
    # Create second stage
    second_stage = torch.nn.Sequential()
    for i in range(num_layers // 2, num_layers):
        second_stage.add_module(f"layer_{i-num_layers//2}", layers[i])
    second_stage.add_module("norm", norm_layer)
    second_stage.add_module("lm_head", lm_head)
    
    # Wrap in pipeline stages
    pipeline_stages = [
        PipelineStage(0, 2, first_stage, hidden_size),
        PipelineStage(1, 2, second_stage, hidden_size)
    ]
    
    return pipeline_stages, hidden_size

def train(rank, num_gpus, model_path, train_file, test_file, epochs, batch_size):
    # Setup process group
    setup(rank, num_gpus)
    
    if rank == 0:
        print(f"Starting pipeline parallel training on {num_gpus} GPUs")
    
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
        
        # Create pipeline stages and get the one for this rank
        pipeline_stages, hidden_size = create_pipeline_stages(model, num_gpus)
        stage = pipeline_stages[rank]
        
        # Free up memory
        del model
        torch.cuda.empty_cache()
        
        if rank == 0:
            print(f"Created pipeline with {len(pipeline_stages)} stages")
    except Exception as e:
        print(f"Error setting up model: {e}")
        cleanup()
        return
    
    # Move stage to the appropriate device
    device = torch.device(f"cuda:{rank}")
    stage = stage.to(device)
    
    # Load datasets
    try:
        train_dataset = TextDataset(train_file, tokenizer)
        test_dataset = TextDataset(test_file, tokenizer)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        cleanup()
        return
    
    # Create dataloader - each rank needs the full dataset
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(stage.parameters(), lr=5e-5, weight_decay=0.01)
    
    # Training loop with pipeline parallelism
    total_time = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        stage.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Pipeline forward pass
            if rank == 0:
                # First stage: process input and send hidden states
                hidden_states = stage(batch_dict=batch)
                
                # Send hidden states to next stage
                dist.send(hidden_states, dst=rank+1)
                
                # Receive gradients from next stage during backward pass
                hidden_grads = torch.zeros_like(hidden_states)
                dist.recv(hidden_grads, src=rank+1)
                
                # Backward pass with received gradients
                optimizer.zero_grad()
                hidden_states.backward(hidden_grads)
                optimizer.step()
                
            else:  # rank == 1, last stage
                # Receive hidden states from previous stage
                hidden_states = torch.zeros(
                    (batch["input_ids"].size(0), batch["input_ids"].size(1), hidden_size),
                    device=device
                )
                dist.recv(hidden_states, src=rank-1)
                
                # Forward pass in last stage
                hidden_states.requires_grad_()
                logits, loss = stage(batch_dict=batch, hidden_states=hidden_states)
                
                if loss is not None:
                    total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Send gradients to previous stage
                dist.send(hidden_states.grad, dst=rank-1)
                
                optimizer.step()
            
            # Print progress
            if rank == 1 and step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time += epoch_time
        
        # Only rank 1 (last stage) has the loss
        if rank == 1:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # Evaluation (simplified - only last stage reports perplexity)
    if rank == 1:
        perplexity = evaluate_pipeline(stage, test_dataset, device, hidden_size, rank)
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Average time per epoch: {total_time/epochs:.2f}s")
    
    cleanup()

def evaluate_pipeline(stage, dataset, device, hidden_size, rank):
    """Simplified pipeline evaluation for the last stage."""
    if rank != 1:
        return 0  # Only last stage calculates perplexity
    
    stage.eval()
    test_loader = DataLoader(dataset, batch_size=4)
    total_loss = 0
    total_length = 0
    
    # This is a simplified evaluation that doesn't involve pipeline
    # In a real implementation, we would need communication between stages
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # For evaluation, we're using a dummy hidden state
            # In a real pipeline, this would come from the previous stage
            hidden_states = torch.zeros(
                (batch["input_ids"].size(0), batch["input_ids"].size(1), hidden_size),
                device=device
            )
            
            _, loss = stage(batch_dict=batch, hidden_states=hidden_states)
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