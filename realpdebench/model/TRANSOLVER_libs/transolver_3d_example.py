import torch
import torch.nn as nn
import numpy as np
import sys
import os


# 添加模型路径到系统路径
sys.path.append('./realpdebench/realpdebench/model/TRANSOLVER_libs')

# 导入模型相关模块
from Transolver_Structured_Mesh_3D import Model
from Physics_Attention import Physics_Attention_Structured_Mesh_3D
from Embedding import timestep_embedding

def create_transolver_3d_model():
    """
    创建Transolver_Structured_Mesh_3D模型实例
    """
    # 模型参数配置
    model_config = {
        'space_dim': 3,           # 空间维度 (x, y, z)
        'n_layers': 5,            # Transformer层数
        'n_hidden': 256,          # 隐藏层维度
        'dropout': 0.1,           # Dropout率
        'n_head': 8,              # 注意力头数
        'Time_Input': True,       # 是否使用时间输入
        'act': 'gelu',            # 激活函数
        'mlp_ratio': 4,           # MLP扩展比例
        'fun_dim': 1,             # 函数维度
        'out_dim': 1,             # 输出维度
        'slice_num': 32,          # 切片数量
        'ref': 8,                 # 参考网格大小
        'unified_pos': False,     # 是否使用统一位置编码
        'H': 32,                  # 高度
        'W': 32,                  # 宽度
        'D': 32,                  # 深度
    }
    
    # 创建模型
    model = Model(**model_config)
    
    return model, model_config

def generate_sample_data(batch_size=2, H=32, W=32, D=32, space_dim=3, fun_dim=1):
    """
    生成示例数据
    """
    # 生成空间坐标 (x, y, z)
    x_coords = torch.linspace(0, 1, H).reshape(1, H, 1, 1).repeat(batch_size, 1, W, D)
    y_coords = torch.linspace(0, 1, W).reshape(1, 1, W, 1).repeat(batch_size, H, 1, D)
    z_coords = torch.linspace(0, 1, D).reshape(1, 1, 1, D).repeat(batch_size, H, W, 1)
    
    # 组合空间坐标
    spatial_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)  # [B, H, W, D, 3]
    
    # 生成函数值 (例如：简单的正弦波)
    x_grid, y_grid, z_grid = torch.meshgrid(
        torch.linspace(0, 2*np.pi, H),
        torch.linspace(0, 2*np.pi, W),
        torch.linspace(0, 2*np.pi, D),
        indexing='ij'
    )
    function_values = torch.sin(x_grid + y_grid + z_grid).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1, fun_dim)
    
    # 重塑为模型期望的格式
    # 模型期望输入: [batch_size, H*W*D, space_dim] 和 [batch_size, H*W*D, fun_dim]
    x = spatial_coords.reshape(batch_size, H*W*D, space_dim)
    fx = function_values.reshape(batch_size, H*W*D, fun_dim)
    
    # 生成时间步
    T = torch.tensor([0.5, 1.0])  # 两个批次的时间步
    
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