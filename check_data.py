import torch


print("PyTorch version:", torch.__version__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available.")
    print("Using Apple GPU:", device)

elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.")
    print("Using NVIDIA GPU:", device)

else:
    device = torch.device("cpu")
    print("MPS/CUDA not available.")
    print("Using CPU.")