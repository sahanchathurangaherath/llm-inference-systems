import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(torch.cuda.get_device_name(0), "| VRAM GB:", round(p.total_memory / 1e9, 1), "| cc:", p.major, p.minor)
    a = torch.randn(2048, 2048, device="cuda"); print("matmul ok:", (a @ a).shape)
