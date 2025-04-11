import torch

def adjust_state_dict(pretrained_path, num_classes_new, num_classes_old=97):
    # Load pretrained state dict (use map_location to ensure compatibility with CPU/GPU)
    state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    
    # Get the old weights and bias
    old_weight = state_dict['module.Prediction.weight']  # Shape: [97, 256]
    old_bias = state_dict['module.Prediction.bias']      # Shape: [97]
    
    # Initialize new weights and bias
    new_weight = torch.zeros(num_classes_new, old_weight.size(1))  # Shape: [136, 256]
    new_bias = torch.zeros(num_classes_new)                       # Shape: [136]
    
    # Copy the weights for the minimum number of classes
    min_classes = min(num_classes_old, num_classes_new)
    new_weight[:min_classes] = old_weight[:min_classes]
    new_bias[:min_classes] = old_bias[:min_classes]
    
    # Initialize the remaining weights (for new classes) using the same method as the model
    # For example, initialize with a normal distribution (common for linear layers)
    if num_classes_new > num_classes_old:
        torch.nn.init.normal_(new_weight[min_classes:], mean=0.0, std=0.02)
        torch.nn.init.zeros_(new_bias[min_classes:])
    
    # Update the state dict
    state_dict['module.Prediction.weight'] = new_weight
    state_dict['module.Prediction.bias'] = new_bias
    
    # Save the adjusted state dict
    adjusted_path = pretrained_path.replace('.pth', '_adjusted.pth')
    torch.save(state_dict, adjusted_path)
    return adjusted_path

# Usage
pretrained_path = 'trainer/saved_models/vi_filtered/iter_300000.pth'
adjusted_path = adjust_state_dict(pretrained_path, num_classes_new=136)
print(f"Adjusted model saved to: {adjusted_path}")