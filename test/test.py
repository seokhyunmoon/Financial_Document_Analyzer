import torch

# 1. Check basic version info
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
print(f"CUDA Version:    {torch.version.cuda}")
print(f"GPU Count:       {torch.cuda.device_count()}")

# 2. Loop through all GPUs to check their names and memory
if torch.cuda.is_available():
    print("-" * 30)
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"   Memory: {props.total_memory / 1024**3:.2f} GB")
        
    # 3. Perform a small tensor calculation on GPU 0 to ensure drivers are working
    print("-" * 30)
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.matmul(x, x)
        print("✓ Tensor operation on GPU successful.")
    except Exception as e:
        print(f"✗ Error during tensor operation: {e}")