import torch
import torch.nn as nn
import numpy as np
import sys
import os


# Add model path to system path
sys.path.append('./realpdebench/realpdebench/model/TRANSOLVER_libs')

# Import model modules
from Transolver_Structured_Mesh_3D import Model
from Physics_Attention import Physics_Attention_Structured_Mesh_3D
from Embedding import timestep_embedding

def create_transolver_3d_model():
    """
    Create a Transolver_Structured_Mesh_3D model instance.
    """
    # Model configuration
    model_config = {
        'space_dim': 3,           # spatial dimension (x, y, z)
        'n_layers': 5,            # number of Transformer layers
        'n_hidden': 256,          # hidden dimension
        'dropout': 0.1,           # dropout rate
        'n_head': 8,              # number of attention heads
        'Time_Input': True,       # whether to use time input
        'act': 'gelu',            # activation function
        'mlp_ratio': 4,           # MLP expansion ratio
        'fun_dim': 1,             # function dimension
        'out_dim': 1,             # output dimension
        'slice_num': 32,          # number of slices
        'ref': 8,                 # reference grid size
        'unified_pos': False,     # whether to use unified positional encoding
        'H': 32,                  # height
        'W': 32,                  # width
        'D': 32,                  # depth
    }

    # Create model
    model = Model(**model_config)
    
    return model, model_config

def generate_sample_data(batch_size=2, H=32, W=32, D=32, space_dim=3, fun_dim=1):
    """
    Generate sample data.
    """
    # Generate spatial coordinates (x, y, z)
    x_coords = torch.linspace(0, 1, H).reshape(1, H, 1, 1).repeat(batch_size, 1, W, D)
    y_coords = torch.linspace(0, 1, W).reshape(1, 1, W, 1).repeat(batch_size, H, 1, D)
    z_coords = torch.linspace(0, 1, D).reshape(1, 1, 1, D).repeat(batch_size, H, W, 1)
    
    # Combine spatial coordinates
    spatial_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)  # [B, H, W, D, 3]

    # Generate function values (e.g., simple sine wave)
    x_grid, y_grid, z_grid = torch.meshgrid(
        torch.linspace(0, 2*np.pi, H),
        torch.linspace(0, 2*np.pi, W),
        torch.linspace(0, 2*np.pi, D),
        indexing='ij'
    )
    function_values = torch.sin(x_grid + y_grid + z_grid).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1, fun_dim)
    
    # Reshape to model-expected format
    # Model expects: [batch_size, H*W*D, space_dim] and [batch_size, H*W*D, fun_dim]
    x = spatial_coords.reshape(batch_size, H*W*D, space_dim)
    fx = function_values.reshape(batch_size, H*W*D, fun_dim)

    # Generate time steps
    T = torch.tensor([0.5, 1.0])  # time steps for two batches
    
    return x, fx, T

def main():
    """
    Main function: demonstrate model usage.
    """
    print("=== Transolver_Structured_Mesh_3D Model Usage Example ===\n")
    
    # 1. Create model
    print("1. Creating model...")
    model, config = create_transolver_3d_model()
    print(f"Model config: {config}")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 2. Generate sample data
    print("2. Generating sample data...")
    batch_size = 2
    x, fx, T = generate_sample_data(
        batch_size=batch_size,
        H=config['H'],
        W=config['W'],
        D=config['D'],
        space_dim=config['space_dim'],
        fun_dim=config['fun_dim']
    )
    
    print(f"Input data shapes:")
    print(f"  x (spatial coords): {x.shape}")
    print(f"  fx (function values): {fx.shape}")
    print(f"  T (time step): {T.shape}")
    print()
    
    # 3. Forward pass
    print("3. Running forward pass...")
    model.eval()
    
    with torch.no_grad():
        if config['Time_Input']:
            output = model(x, fx, T)
            print(f"Output shape (with time input): {output.shape}")
        else:
            output = model(x, fx)
            print(f"Output shape (without time input): {output.shape}")
    
    print()
    
    # 4. Demonstrate different input modes
    print("4. Demonstrating different input modes...")
    
    print("Mode 1: spatial coords only (no function values)")
    with torch.no_grad():
        output1 = model(x, fx=None, T=T if config['Time_Input'] else None)
        print(f"  Output shape: {output1.shape}")
    
    print("Mode 2: spatial coords + function values")
    with torch.no_grad():
        output2 = model(x, fx, T=T if config['Time_Input'] else None)
        print(f"  Output shape: {output2.shape}")
    
    print("Mode 3: without time input")
    model_no_time = Model(Time_Input=False, **{k:v for k,v in config.items() if k != 'Time_Input'})
    with torch.no_grad():
        output3 = model_no_time(x, fx)
        print(f"  Output shape: {output3.shape}")
    
    print()
    
    # 5. Training example
    print("5. Training example...")
    model.train()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    target = torch.randn_like(output2)
    
    pred = model(x, fx, T=T if config['Time_Input'] else None)
    loss = criterion(pred, target)
    
    print(f"Training loss: {loss.item():.6f}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Backward pass complete")
    print()
    
    # 6. Model info
    print("6. Model details...")
    print(f"Model name: {model.__name__}")
    print(f"Model structure:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameter statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"  Model size: {size_all_mb:.2f} MB")

if __name__ == "__main__":
    main() 