import torch

pretrained_path = 'trainer/saved_models/vi_filtered/iter_300000.pth'
state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))  # Add map_location
pred_weight_shape = state_dict['module.Prediction.weight'].shape
pred_bias_shape = state_dict['module.Prediction.bias'].shape
print(f"Pretrained Prediction.weight shape: {pred_weight_shape}")
print(f"Pretrained Prediction.bias shape: {pred_bias_shape}")