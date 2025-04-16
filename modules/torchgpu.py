import torch
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor)
    # torch.set_default_dtype(torch.float32)
    # torch.set_default_device(device)
    
if device.type == 'cpu':
    warnings.warn("Running on CPU")