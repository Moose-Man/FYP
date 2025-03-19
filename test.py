import torch
print("PyTorch Built for CUDA:", torch.version.cuda)
print("Is CUDA Available:", torch.cuda.is_available())
print("Torch Compiled With CUDA?", torch.backends.cudnn.enabled)