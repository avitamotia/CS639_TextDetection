import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("No CUDA GPU found — running on CPU only.")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple MPS (Metal) is available as an alternative.")
