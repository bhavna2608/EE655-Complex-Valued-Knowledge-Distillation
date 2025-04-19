import matplotlib.pyplot as plt
import numpy as np
import torchvision
import seaborn as sns

def plot_attention_maps(attention_weights, patch_size=8, img_size=224):
    """Visualize attention maps from transformer layers"""
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]
    
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(3*num_heads, 3*num_layers))
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
            grid_size = int(np.sqrt(attn.shape[-1]))
            attn = attn.reshape(grid_size, grid_size)
            
            if num_layers > 1:
                ax = axes[layer_idx, head_idx]
            else:
                ax = axes[head_idx]
            
            sns.heatmap(attn, ax=ax, cmap='viridis', cbar=False)
            ax.set_title(f'L{layer_idx+1} H{head_idx+1}')
            ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_curves(train_metrics, val_metrics, metric_name='top1'):
    """Plot training and validation curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training curve
    train_values = [m[metric_name] for m in train_metrics]
    ax.plot(train_values, label=f'Training {metric_name}')
    
    # Plot validation curve
    val_values = [m[metric_name] for m in val_metrics]
    ax.plot(val_values, label=f'Validation {metric_name}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Training vs Validation {metric_name}')
    ax.legend()
    ax.grid(True)
    
    return fig

def visualize_patches(images, patch_size=8):
    """Visualize image patches with borders"""
    # Add borders between patches
    images = images.clone()
    images[:, :, ::patch_size, :] = 1.0  # Vertical lines
    images[:, :, :, ::patch_size] = 1.0  # Horizontal lines
    
    # Make grid of images
    grid = torchvision.utils.make_grid(images, nrow=4, padding=2, normalize=True)
    
    # Convert to numpy and plot
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    return plt.gcf()