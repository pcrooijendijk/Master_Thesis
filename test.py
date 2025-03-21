import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return number of available GPUs
print(torch.cuda.get_device_name(0))  # Should return GPU name
print(torch.cuda.current_device())  # Should return the current device index
