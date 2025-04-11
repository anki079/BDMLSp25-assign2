# """
# Pipeline Parallel LLaMA Fine-Tuning (similar hyperparams to data/tensor parallel).
# Splits model layers across 2 GPUs using torch.distributed.pipeline.sync.Pipe.

#  - BF16
#  - Gradient checkpointing
#  - 3 epochs, same LR, gradient_accumulation, etc.
#  - 'paged AdamW 8-bit' via bitsandbytes
#  - Cosine LR scheduling
#  - Saves checkpoints each epoch
#  - Evaluates perplexity on a test set
# """

# import os
# import math
# import time
# import torch
# import argparse
# import bitsandbytes as bnb
# import torch.nn as nn

# from torch.distributed.pipeline.sync import Pipe
# from torch.utils.data import DataLoader
# from datasets import load_from_disk
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import get_cosine_schedule_with_warmup  # for LR scheduling

# # We'll define a small "sequential" wrapper to simulate pipeline stages
# class LlamaPipelineModel(nn.Module):
#     """
#     A container that splits a LLaMA model's layers into two pipeline partitions.
#     For simplicity, we do something naive:
#       - Partition 1: The 'body' (transformer backbone)
#       - Partition 2: The 'lm_head'
#     In practice, you might split the Transformer layers more evenly.
#     """
#     def __init__(self, llama_model):
#         super().__init__()
#         # The HF AutoModelForCausalLM includes a .model (backbone) + .lm_head
#         # We'll break them into two pipeline stages:
#         self.partition1 = nn.Sequential(
#             llama_model.model  # everything except final output projection
#         )
#         self.partition2 = nn.Sequential(
#             llama_model.lm_head
#         )

#     def forward(self, x, labels=None):
#         """
#         We won't use forward() directly, since Pipe will handle it in a sequential manner.
#         """
#         raise NotImplementedError("Pipe will call partition1->partition2 automatically")


# def main():
#     parser = argparse.ArgumentParser(description="Pipeline Parallel Fine-Tuning (matching data/tensor hyperparams)")
#     parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size")
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data_chunks")
#     parser.add_argument("--lr", type=float, default=2e-4)
#     parser.add_argument("--warmup_ratio", type=float, default=0.05)
#     parser.add_argument("--weight_decay", type=float, default=0.01)
#     parser.add_argument("--logging_steps", type=int, default=100)
#     args = parser.parse_args()

#     print("*** Pipeline Parallel Fine-Tuning ***")
#     print(f"Process Info: PID={os.getpid()} - single process controlling GPUs [0,1].")

#     # Output dir
#     output_dir = "./checkpoints-llama-pipeline-parallel"
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Checkpoints + results will be saved to: {output_dir}")

#     # 1) Load Tokenizer + Dataset
#     model_dir = "./llama-hf"
#     print("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     print(f"Loading tokenized datasets from: {args.tokenized_data_dir}")
#     tokenized_datasets = load_from_disk(args.tokenized_data_dir)
#     train_dataset = tokenized_datasets["train"]
#     test_dataset  = tokenized_datasets["test"]

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size)

#     num_train_examples = len(train_dataset)
#     print(f"Num Train Examples = {num_train_examples}, Num Test Examples = {len(test_dataset)}")

#     # 2) Load LLaMA Model
#     print("Loading base LLaMA model...")
#     llama_model = AutoModelForCausalLM.from_pretrained(model_dir)

#     # BF16 + grad checkpointing
#     print("Converting model to bf16 + enabling gradient checkpointing...")
#     llama_model = llama_model.to(torch.bfloat16)
#     llama_model.gradient_checkpointing_enable()
#     llama_model.config.use_cache = False

#     # 3) Create Pipeline Model
#     # We'll wrap the model in a simple container that splits it into partition1 and partition2
#     pipeline_model = LlamaPipelineModel(llama_model)

#     # 4) Setup Pipe for pipeline parallel
#     # We'll define a 2-stage pipeline (balance=[1,1]) with devices=[0,1].
#     # In reality, you might want to split layers more precisely, e.g. half on GPU0, half on GPU1
#     from torch.distributed.pipeline.sync import Pipe
#     pipeline = Pipe(
#         pipeline_model,
#         balance=[1,1],           # 2 stages: partition1 is stage 0, partition2 is stage 1
#         devices=[0, 1],         # GPU0 for stage 0, GPU1 for stage 1
#         chunks=8,               # micro-batches per batch for pipeline concurrency
#         checkpoint="never"      # we already do gradient_checkpointing; can also try "always"
#     )

#     # The pipeline is now a single nn.Module that automatically runs data through partitions in sequence.
#     # We'll define a small helper function to do forward with labels:
#     def pipeline_forward(input_ids, labels):
#         """Compute logits from pipeline, returns (logits,) so we can do loss manually."""
#         # Pipe must be called with a single Tensor or tuple of Tensors.
#         # We'll pass input_ids, then do cross-entropy ourselves.
#         output_tuple = pipeline(input_ids)  # returns (logits,) from partition2
#         # The final partition is basically the final LM head output => shape: (batch, seq_len, vocab)
#         logits = output_tuple[0]
#         # We'll compute loss manually. Shift by 1 for a causal LM.
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         loss_fct = torch.nn.CrossEntropyLoss()
#         loss = loss_fct(
#             shift_logits.view(-1, shift_logits.size(-1)),
#             shift_labels.view(-1)
#         )
#         return loss

#     # 5) Define Optimizer, Scheduler
#     optimizer = bnb.optim.PagedAdamW(
#         pipeline.parameters(),
#         lr=args.lr,
#         weight_decay=args.weight_decay
#     )

#     # Approx steps
#     steps_per_epoch = math.ceil(num_train_examples / (args.batch_size * 1.0))
#     effective_steps_per_epoch = steps_per_epoch / args.gradient_accumulation_steps
#     total_steps = int(effective_steps_per_epoch * args.epochs)
#     warmup_steps = int(total_steps * args.warmup_ratio)

#     print(f"Using Cosine LR scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}")
#     lr_scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=total_steps
#     )

#     # 6) Training Loop
#     print(f"Starting pipeline-parallel training for {args.epochs} epoch(s) with batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}...")
#     total_train_start = time.time()

#     global_step = 0
#     for epoch in range(args.epochs):
#         pipeline.train()
#         epoch_start = time.time()

#         total_loss = 0.0
#         total_steps_in_epoch = 0
#         accum_counter = 0

#         for step, batch in enumerate(train_loader):
#             # We'll feed input_ids -> pipeline, compute manual cross-entropy
#             input_ids = batch["input_ids"].to(0)  # move to GPU0
#             labels = input_ids.clone()

#             # forward pass
#             loss = pipeline_forward(input_ids, labels=labels)
#             # scale by grad_accum
#             loss = loss / args.gradient_accumulation_steps

#             loss.backward()
#             accum_counter += 1
#             total_loss += loss.item()

#             # Gradient accumulation
#             if accum_counter == args.gradient_accumulation_steps:
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()

#                 global_step += 1
#                 total_steps_in_epoch += 1
#                 accum_counter = 0

#                 # Logging
#                 if (global_step % args.logging_steps) == 0:
#                     avg_loss = (total_loss * args.gradient_accumulation_steps) / total_steps_in_epoch
#                     print(f"Epoch={epoch}, global_step={global_step}, avg_loss={avg_loss:.4f}")

#         epoch_end = time.time()
#         epoch_time = epoch_end - epoch_start

#         # Final epoch average loss
#         avg_epoch_loss = 0.0
#         if total_steps_in_epoch > 0:
#             avg_epoch_loss = (total_loss * args.gradient_accumulation_steps) / total_steps_in_epoch

#         print(f"Epoch {epoch} finished. Time per epoch={epoch_time:.2f}s, avg_loss={avg_epoch_loss:.4f}")

#         # Save checkpoint after each epoch
#         epoch_ckpt_dir = os.path.join(output_dir, f"epoch-{epoch}")
#         os.makedirs(epoch_ckpt_dir, exist_ok=True)
#         print(f"Saving model checkpoint after epoch {epoch} to {epoch_ckpt_dir}...")
#         # Pipe is a wrapper, but we can still access the underlying partitions if needed.
#         # For demonstration, let's just call 'save_pretrained' on the underlying model.
#         # We'll do a naive approach: pipeline_model.partition1 + partition2 are each on different GPUs.
#         # A thorough approach would gather state to CPU. For assignment demonstration, it's typically enough:
#         pipeline_model.cpu()  # move back to CPU
#         torch.save(pipeline_model.state_dict(), os.path.join(epoch_ckpt_dir, "pipeline_model.bin"))
#         # Then move back to GPUs
#         pipeline.to([0, 1])
#         # end saving

#     total_train_end = time.time()
#     total_train_time = total_train_end - total_train_start
#     time_per_epoch = total_train_time / args.epochs
#     print(f"Total training time={total_train_time:.2f}s => ~{time_per_epoch:.2f}s per epoch")

#     # 7) Evaluation
#     print("Evaluating model on test set for perplexity...")
#     pipeline.eval()
#     total_loss_eval = 0.0
#     total_eval_steps = 0

#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids = batch["input_ids"].to(0)
#             labels = input_ids.clone()

#             loss = pipeline_forward(input_ids, labels=labels)
#             total_loss_eval += loss.item()
#             total_eval_steps += 1

#     avg_eval_loss = total_loss_eval / total_eval_steps if total_eval_steps > 0 else float("inf")
#     perplexity = math.exp(avg_eval_loss) if avg_eval_loss < 20 else float("inf")

#     print(f"Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")

#     # 8) Write final results
#     results_path = os.path.join(output_dir, "eval_results_pipeline_parallel.txt")
#     print(f"Writing results to {results_path}")
#     with open(results_path, "w") as f:
#         f.write(f"Time per epoch: {time_per_epoch:.2f}\n")
#         f.write(f"Eval Loss: {avg_eval_loss:.4f}\n")
#         f.write(f"Perplexity: {perplexity:.2f}\n")

#     print("Pipeline parallel fine-tuning complete!")


# if __name__ == "__main__":
#     main()

# pipeline_parallel_even_split.py
import os
import math
import time
import torch
import argparse
import bitsandbytes as bnb
import torch.nn as nn

from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.distributed.pipeline.sync import Pipe
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)

###############################################################################
# 1) Define partition modules for an even split of LLaMA layers
###############################################################################
class PipelinePartition1(nn.Module):
    """Partition 1: embed_tokens + first half of the layers."""
    def __init__(self, embed_tokens, layers):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList(layers)

    def forward(self, input_ids):
        # Convert token IDs -> embedded hidden states
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class PipelinePartition2(nn.Module):
    """Partition 2: second half of the layers + final norm + lm_head."""
    def __init__(self, layers, norm, lm_head):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


###############################################################################
# 2) Main Script for Pipeline Parallel Fine-Tuning
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Pipeline Parallel Fine-Tuning with an Even Model Split")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tokenized_data_dir", type=str, default="./tokenized_data_chunks")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=100)
    args = parser.parse_args()

    print("*** Pipeline Parallel Fine-Tuning (Even Split) ***")
    print(f"PID={os.getpid()} (single process controlling GPUs [0,1])")

    # Output directory for saving model + logs
    output_dir = "./checkpoints-llama-pipeline-parallel-even"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Checkpoints + results will be saved to {output_dir}")

    ############################################################################
    # 2A) Load Tokenizer + Dataset
    ############################################################################
    model_dir = "./llama-hf"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading tokenized dataset from: {args.tokenized_data_dir}")
    ds = load_from_disk(args.tokenized_data_dir)
    train_dataset = ds["train"]
    test_dataset  = ds["test"]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)

    num_train_examples = len(train_dataset)
    print(f"Num Train Examples = {num_train_examples}, Num Test Examples = {len(test_dataset)}")

    ############################################################################
    # 2B) Load LLaMA Model
    ############################################################################
    print("Loading base LLaMA model...")
    full_model = AutoModelForCausalLM.from_pretrained(model_dir)

    # BF16 + gradient checkpointing
    print("Converting model to bf16 + enabling gradient checkpointing...")
    full_model = full_model.to(torch.bfloat16)
    full_model.gradient_checkpointing_enable()
    full_model.config.use_cache = False

    # The LLaMA structure:
    #  - full_model.model.embed_tokens
    #  - full_model.model.layers (list of N layers)
    #  - full_model.model.norm
    #  - full_model.lm_head
    # We want to split half the layers on Partition1, half on Partition2.
    n_layers = len(full_model.model.layers)
    split_index = n_layers // 2
    print(f"LLaMA has {n_layers} layers; splitting at layer {split_index}")

    partition1 = PipelinePartition1(
        embed_tokens=full_model.model.embed_tokens,
        layers=full_model.model.layers[:split_index]
    )
    partition2 = PipelinePartition2(
        layers=full_model.model.layers[split_index:],
        norm=full_model.model.norm,
        lm_head=full_model.lm_head
    )

    # We'll create a sequential container with these two partitions
    # so that Pipe can see them as [partition1, partition2].
    pipeline_model = nn.Sequential(partition1, partition2)

    ############################################################################
    # 3) Build Pipe
    ############################################################################
    from torch.distributed.pipeline.sync import Pipe
    # We'll define 2 pipeline stages (balance=[1,1]) across devices [0,1].
    # chunks=8 => micro-batching
    pipeline = Pipe(
        pipeline_model,
        balance=[1,1],
        devices=[0,1],
        chunks=8,
        checkpoint="never"  # we already do gradient_checkpointing in the model
    )

    # Now 'pipeline' is effectively our model that splits layers across GPU0, GPU1.

    ############################################################################
    # 4) Define Optimizer + LR Scheduler
    ############################################################################
    optimizer = bnb.optim.PagedAdamW(
        pipeline.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    steps_per_epoch = math.ceil(num_train_examples / float(args.batch_size))
    effective_steps_per_epoch = steps_per_epoch / args.gradient_accumulation_steps
    total_steps = int(effective_steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Using Cosine LR scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    ############################################################################
    # 5) Training Loop
    ############################################################################
    print(f"Starting pipeline-parallel training for {args.epochs} epoch(s).")
    print(f"Batch size={args.batch_size}, gradient_accumulation={args.gradient_accumulation_steps}")

    total_train_start = time.time()
    global_step = 0

    def forward_pass(input_ids, labels):
        """
        We'll feed input_ids through pipeline -> produce logits,
        then manually compute cross-entropy.
        """
        # Pipe returns a tuple, (logits,) in this case
        outputs = pipeline(input_ids)
        logits = outputs[0]
        # We'll do standard causal LM shifting
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        return loss

    for epoch in range(args.epochs):
        pipeline.train()
        epoch_start = time.time()

        total_loss = 0.0
        total_steps_in_epoch = 0
        accum_counter = 0

        for step, batch in enumerate(train_loader):
            # Move input_ids to GPU0
            input_ids = batch["input_ids"].to(0)
            labels = input_ids.clone()

            loss = forward_pass(input_ids, labels)
            # scale by grad_accum
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            accum_counter += 1
            total_loss += loss.item()

            if accum_counter == args.gradient_accumulation_steps:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                total_steps_in_epoch += 1
                accum_counter = 0

                # Logging
                if (global_step % args.logging_steps) == 0:
                    avg_loss = (total_loss * args.gradient_accumulation_steps) / total_steps_in_epoch
                    print(f"Epoch={epoch}, global_step={global_step}, avg_loss={avg_loss:.4f}")

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        avg_epoch_loss = 0.0
        if total_steps_in_epoch > 0:
            avg_epoch_loss = (total_loss * args.gradient_accumulation_steps) / total_steps_in_epoch

        print(f"Epoch {epoch} finished. Time per epoch={epoch_time:.2f}s, avg_loss={avg_epoch_loss:.4f}")

        # Save checkpoint
        epoch_ckpt_dir = os.path.join(output_dir, f"epoch-{epoch}")
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        print(f"Saving model checkpoint after epoch {epoch} to {epoch_ckpt_dir} ...")

        # We'll do a naive approach to saving:
        # Move pipeline to CPU, save the state_dict, then move it back
        pipeline.to("cpu")
        # Pipe is a wrapper around pipeline_model
        # pipeline[0] = partition1, pipeline[1] = partition2
        torch.save(pipeline.state_dict(), os.path.join(epoch_ckpt_dir, "pipeline_state.pt"))
        # Move back to GPUs
        pipeline.to([0,1])

    total_train_time = time.time() - total_train_start
    time_per_epoch = total_train_time / args.epochs
    print(f"Total training time={total_train_time:.2f}s => ~{time_per_epoch:.2f}s per epoch")

    ############################################################################
    # 6) Evaluation
    ############################################################################
    print("Evaluating model on test set for perplexity...")
    pipeline.eval()

    total_loss_eval = 0.0
    total_eval_steps = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(0)
            labels = input_ids.clone()
            loss = forward_pass(input_ids, labels)
            total_loss_eval += loss.item()
            total_eval_steps += 1

    avg_eval_loss = total_loss_eval / total_eval_steps if total_eval_steps > 0 else float("inf")
    perplexity = math.exp(avg_eval_loss) if avg_eval_loss < 20 else float("inf")

    print(f"Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")

    # Write results
    results_path = os.path.join(output_dir, "eval_results_pipeline_parallel_even.txt")
    print(f"Writing results to {results_path}")
    with open(results_path, "w") as f:
        f.write(f"Time per epoch: {time_per_epoch:.2f}\n")
        f.write(f"Eval Loss: {avg_eval_loss:.4f}\n")
        f.write(f"Perplexity: {perplexity:.2f}\n")

    print("Pipeline parallel (even split) fine-tuning complete!")


if __name__ == "__main__":
    main()
