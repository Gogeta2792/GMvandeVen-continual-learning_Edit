"""
Example usage of AttentionRouter for modular continual learning.

This script demonstrates how to use the AttentionRouter with ModularLayer
for your FYP project "Modular Continual Learning with Task-Agnostic Subnetwork Routing via Attention Mechanisms."

The example shows:
1. Local and global attention routing
2. Router regularization for stable attention
3. Integration with ModularLayer
4. Training loop with router regularizers
5. Analysis of attention patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.attention_router import (
    AttentionRouter, RouterConfig, LocalAttentionRouter, GlobalAttentionRouter,
    create_local_router, create_global_router
)
from models.modular_layer import ModularLayer, ModularLayerConfig, create_mlp_modular_layer


def example_local_router():
    """Example of using local attention router."""
    print("=== Local Attention Router Example ===")
    
    # Create a local router for 4 modules
    router = create_local_router(
        num_modules=4,
        input_dim=100,
        hidden_dim=128,
        entropy_coef=0.01,  # Encourage sparse attention
        drift_coef=0.1      # Encourage stable attention
    )
    
    print(f"Created local router with {router.num_modules} modules")
    print(f"Router parameters: {sum(p.numel() for p in router.parameters()):,}")
    
    # Test with different inputs
    batch_sizes = [1, 32, 128]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 100)
        logits, attention, regularizers = router(x)
        
        print(f"Batch size {batch_size:3d}:")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Attention shape: {attention.shape}")
        print(f"  Attention sums: {attention.sum(dim=-1).mean().item():.6f}")
        print(f"  Regularizers: {list(regularizers.keys())}")
    
    # Show attention patterns
    x = torch.randn(32, 100)
    logits, attention, regularizers = router(x)
    
    print(f"\nAttention patterns (first 5 samples):")
    for i in range(5):
        attn_weights = attention[i].detach().numpy()
        print(f"  Sample {i}: {attn_weights}")
    
    return router


def example_global_router():
    """Example of using global attention router."""
    print("\n=== Global Attention Router Example ===")
    
    # Create a global router
    router = create_global_router(
        num_modules=6,
        input_dim=100,
        global_dim=50,
        hidden_dim=128,
        entropy_coef=0.01,
        drift_coef=0.1
    )
    
    print(f"Created global router with {router.num_modules} modules")
    print(f"Router parameters: {sum(p.numel() for p in router.parameters()):,}")
    
    # Test with local and global inputs
    x = torch.randn(32, 100)
    global_context = torch.randn(32, 50)
    
    logits, attention, regularizers = router(x, global_context)
    
    print(f"Input shapes: local {x.shape}, global {global_context.shape}")
    print(f"Output shapes: logits {logits.shape}, attention {attention.shape}")
    print(f"Regularizers: {list(regularizers.keys())}")
    
    return router


def example_modular_layer_with_router():
    """Example of ModularLayer with attention routing."""
    print("\n=== ModularLayer with Router Example ===")
    
    # Create modular layer with router
    layer = create_mlp_modular_layer(
        in_dim=784,      # MNIST input
        out_dim=400,     # Hidden size
        num_modules=4,   # 4 parallel modules
        use_router=True, # Enable attention routing
        router_hidden_dim=128,
        router_entropy_coef=0.01,
        router_drift_coef=0.1
    )
    
    print(f"Created modular layer with router")
    print(f"Total parameters: {sum(p.numel() for p in layer.parameters()):,}")
    print(f"Router parameters: {sum(p.numel() for p in layer.router.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(32, 784)
    
    # Without attention info
    output, attention_info = layer(x)
    print(f"Output shape (no attention info): {output.shape}")
    
    # With attention info
    output, attention_info = layer(x, return_attention=True)
    print(f"Output shape (with attention info): {output.shape}")
    print(f"Attention shape: {attention_info['attention'].shape}")
    print(f"Regularizers: {list(attention_info['regularizers'].keys())}")
    
    # Show attention patterns
    attention = attention_info['attention']
    print(f"\nAttention patterns (first 3 samples):")
    for i in range(3):
        attn_weights = attention[i].detach().numpy()
        print(f"  Sample {i}: {attn_weights}")
    
    return layer


def example_training_with_router():
    """Example of training with router regularizers."""
    print("\n=== Training with Router Regularizers Example ===")
    
    # Create a simple network with modular layers
    class ModularNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.modular_layer1 = create_mlp_modular_layer(
                in_dim=784, out_dim=400, num_modules=4, use_router=True,
                router_entropy_coef=0.01, router_drift_coef=0.1
            )
            self.modular_layer2 = create_mlp_modular_layer(
                in_dim=400, out_dim=200, num_modules=4, use_router=True,
                router_entropy_coef=0.01, router_drift_coef=0.1
            )
            self.classifier = nn.Linear(200, 10)
        
        def forward(self, x, return_regularizers=False):
            # Flatten input
            x = x.view(x.size(0), -1)
            
            # First modular layer
            x, attn_info1 = self.modular_layer1(x, return_attention=True)
            regularizers = attn_info1['regularizers']
            
            # Second modular layer
            x, attn_info2 = self.modular_layer2(x, return_attention=True)
            for key, value in attn_info2['regularizers'].items():
                if key in regularizers:
                    regularizers[key] += value
                else:
                    regularizers[key] = value
            
            # Classification
            output = self.classifier(x)
            
            if return_regularizers:
                return output, regularizers
            else:
                return output
    
    # Create network and optimizer
    network = ModularNetwork()
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # Simulate training loop
    network.train()
    for epoch in range(3):
        total_loss = 0
        total_regularizer_loss = 0
        
        for batch in range(5):  # 5 batches per epoch
            # Simulate batch data
            x = torch.randn(32, 784)
            y = torch.randint(0, 10, (32,))
            
            # Forward pass
            output, regularizers = network(x, return_regularizers=True)
            
            # Compute losses
            ce_loss = nn.CrossEntropyLoss()(output, y)
            reg_loss = sum(regularizers.values())
            total_loss_batch = ce_loss + reg_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += ce_loss.item()
            total_regularizer_loss += reg_loss.item()
        
        print(f"Epoch {epoch+1}:")
        print(f"  CE Loss: {total_loss/5:.4f}")
        print(f"  Regularizer Loss: {total_regularizer_loss/5:.4f}")
        print(f"  Regularizers: {list(regularizers.keys())}")
    
    return network


def example_attention_analysis():
    """Example of analyzing attention patterns."""
    print("\n=== Attention Pattern Analysis Example ===")
    
    # Create a trained-like router (with some weights)
    router = create_local_router(
        num_modules=4,
        input_dim=100,
        hidden_dim=128,
        entropy_coef=0.01,
        drift_coef=0.1
    )
    
    # Simulate some training to update EMA
    router.train()
    for _ in range(10):
        x = torch.randn(32, 100)
        _, _, _ = router(x)
    
    # Analyze attention patterns
    router.eval()
    with torch.no_grad():
        x = torch.randn(100, 100)  # 100 samples
        logits, attention, regularizers = router(x)
        
        print(f"Attention analysis over {x.size(0)} samples:")
        print(f"  Mean attention per module: {attention.mean(dim=0).numpy()}")
        print(f"  Std attention per module: {attention.std(dim=0).numpy()}")
        print(f"  Max attention per sample: {attention.max(dim=-1)[0].mean().item():.4f}")
        print(f"  Min attention per sample: {attention.min(dim=-1)[0].mean().item():.4f}")
        
        # Entropy analysis
        entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
        print(f"  Mean entropy: {entropy.mean().item():.4f}")
        print(f"  Std entropy: {entropy.std().item():.4f}")
        
        # Module usage analysis
        module_usage = (attention > 0.1).float().mean(dim=0)  # Usage threshold
        print(f"  Module usage (>0.1 threshold): {module_usage.numpy()}")
        
        # Router statistics
        stats = router.get_attention_stats()
        print(f"  EMA attention: {stats.get('ema_attention', 'Not available')}")
        print(f"  EMA entropy: {stats.get('ema_entropy', 'Not available'):.4f}")
    
    return router


def example_router_regularization_effects():
    """Example showing effects of different regularization coefficients."""
    print("\n=== Router Regularization Effects Example ===")
    
    # Test different regularization settings
    configs = [
        {"entropy_coef": 0.0, "drift_coef": 0.0, "name": "No regularization"},
        {"entropy_coef": 0.01, "drift_coef": 0.0, "name": "Entropy only"},
        {"entropy_coef": 0.0, "drift_coef": 0.1, "name": "Drift only"},
        {"entropy_coef": 0.01, "drift_coef": 0.1, "name": "Both regularizers"},
    ]
    
    x = torch.randn(32, 100)
    
    for config in configs:
        router = create_local_router(
            num_modules=4,
            input_dim=100,
            hidden_dim=128,
            entropy_coef=config["entropy_coef"],
            drift_coef=config["drift_coef"]
        )
        
        # Simulate some training
        router.train()
        for _ in range(5):
            _, _, _ = router(x)
        
        # Analyze final attention
        router.eval()
        with torch.no_grad():
            _, attention, regularizers = router(x)
            
            entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
            mean_entropy = entropy.mean().item()
            attention_std = attention.std(dim=0).mean().item()
            
            print(f"{config['name']}:")
            print(f"  Mean entropy: {mean_entropy:.4f}")
            print(f"  Attention std: {attention_std:.4f}")
            print(f"  Regularizers: {list(regularizers.keys())}")
            if regularizers:
                for key, value in regularizers.items():
                    print(f"    {key}: {value.item():.6f}")


def main():
    """Run all examples."""
    print("AttentionRouter Examples for Continual Learning FYP")
    print("=" * 60)
    
    # Run examples
    local_router = example_local_router()
    global_router = example_global_router()
    modular_layer = example_modular_layer_with_router()
    trained_network = example_training_with_router()
    analysis_router = example_attention_analysis()
    example_router_regularization_effects()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("\nKey features demonstrated:")
    print("1. [OK] Local and global attention routing")
    print("2. [OK] Router regularization for stable attention")
    print("3. [OK] Integration with ModularLayer")
    print("4. [OK] Training loop with router regularizers")
    print("5. [OK] Attention pattern analysis")
    print("6. [OK] Effects of different regularization coefficients")
    
    print("\nNext steps for your FYP:")
    print("1. Integrate AttentionRouter into your SplitMNIST experiments")
    print("2. Experiment with different regularization coefficients")
    print("3. Analyze attention patterns across different tasks")
    print("4. Compare with baseline methods (SI/EWC/ER)")
    print("5. Extend to more complex scenarios (CIFAR-10/100)")


if __name__ == "__main__":
    main()
