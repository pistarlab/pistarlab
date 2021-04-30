import torch
info={}
gpu_count = torch.cuda.device_count()
info['version'] = torch.__version__
info['gpu_count'] = gpu_count
info['gpu_list'] = {i:torch.cuda.get_device_name(0) for i in range(gpu_count)}
dev_id = torch.cuda.current_device()
info['current_gpu_device_id'] = dev_id 
info['current_gpu_device_name'] = torch.cuda.get_device_name(dev_id)
info['cuda_available'] = torch.cuda.is_available()
print(info)