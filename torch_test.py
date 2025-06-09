import torch

# 1. 檢查 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 2. 檢查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("CUDA is available.")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")  # 顯示第一個 CUDA 設備的名稱
    print(f"CUDA Device Count: {torch.cuda.device_count()}")  # 顯示可用的 CUDA 設備數量
else:
    print("CUDA is not available.")

# 3. 測試 GPU 是否可用（進行簡單的 tensor 操作）
if cuda_available:
    # 在 GPU 上創建一個 tensor，並進行簡單操作
    device = torch.device("cuda")
    x = torch.randn(3, 3).to(device)  # 在 GPU 上創建一個 tensor
    y = torch.randn(3, 3).to(device)
    z = x + y  # 在 GPU 上進行加法操作
    print("GPU tensor operation successful.")
else:
    # 如果 CUDA 不可用，則在 CPU 上進行操作
    device = torch.device("cpu")
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x + y  # 在 CPU 上進行加法操作
    print("CPU tensor operation successful.")
