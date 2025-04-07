import torch
import time

def main():
    # make a small tensor on GPU
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device="cuda")
        while True:
            # minimal matrix multiplication
            y = torch.matmul(x, x)
            # sleep to avoid hogging the GPU
            time.sleep(1)
    else:
        print("CUDA not available")

if __name__ == "__main__":
    main()