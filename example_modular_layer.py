"""
Example usage of ModularLayer for continual learning.

This script demonstrates how to use the ModularLayer class for your FYP project
"Modular Continual Learning with Task-Agnostic Subnetwork Routing via Attention Mechanisms."

The example shows:
1. Creating MLP and Conv modular layers
2. Forward pass with different batch sizes
3. Parameter counting and analysis
4. Integration with the existing continual learning framework
"""

import torch
import torch.nn as nn
from models.modular_layer import ModularLayer, ModularLayerConfig, create_mlp_modular_layer, create_conv_modular_layer


def example_mlp_modular_layer():
    """Example of using MLP modular layer for SplitMNIST."""
    print("=== MLP Modular Layer Example ===")
    
    # Create a modular layer for SplitMNIST (784 input features)
    config = ModularLayerConfig(
        in_dim=784,      # MNIST flattened image size
        out_dim=400,     # Hidden layer size
        num_modules=4,   # 4 parallel modules
        block_type='mlp',
        hidden_dim=400,  # Same as output for simplicity
        dropout=0.1,     # 10% dropout
        device='cpu'
    )
    
    layer = ModularLayer(config)
    print(f"Created MLP modular layer with {layer.num_modules} modules")
    print(f"Total parameters: {layer.get_total_parameters():,}")
    print(f"Parameters per module: {layer.get_parameters_per_module():,}")
    
    # Test with different batch sizes
    batch_sizes = [1, 32, 128]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 784)
        output = layer(x)
        print(f"Batch size {batch_size:3d}: Input {x.shape} -> Output {output.shape}")
    
    # Show individual module outputs
    x = torch.randn(32, 784)
    individual_outputs = layer.get_module_outputs(x)
    print(f"Individual module outputs: {len(individual_outputs)} modules, each with shape {individual_outputs[0].shape}")
    
    return layer


def example_conv_modular_layer():
    """Example of using Conv modular layer for CIFAR."""
    print("\n=== Conv Modular Layer Example ===")
    
    # Create a modular layer for CIFAR (3 channels, 32x32 images)
    config = ModularLayerConfig(
        in_dim=64,       # Input channels (after some processing)
        out_dim=128,     # Output channels
        num_modules=6,   # 6 parallel modules
        block_type='conv',
        dropout=0.1,
        device='cpu'
    )
    
    layer = ModularLayer(config)
    print(f"Created Conv modular layer with {layer.num_modules} modules")
    print(f"Total parameters: {layer.get_total_parameters():,}")
    print(f"Parameters per module: {layer.get_parameters_per_module():,}")
    
    # Test with different batch sizes
    batch_sizes = [1, 16, 64]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 64, 32, 32)  # [B, C, H, W]
        output = layer(x)
        print(f"Batch size {batch_size:2d}: Input {x.shape} -> Output {output.shape}")
    
    return layer


def example_integration_with_existing_framework():
    """Example of how ModularLayer integrates with existing continual learning framework."""
    print("\n=== Integration with Existing Framework ===")
    
    # Create a simple network using modular layers
    class ModularNetwork(nn.Module):
        def __init__(self, input_size=784, hidden_sizes=[400, 400], num_classes=10, num_modules=4):
            super().__init__()
            
            # Create modular layers
            self.modular_layers = nn.ModuleList()
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                config = ModularLayerConfig(
                    in_dim=prev_size,
                    out_dim=hidden_size,
                    num_modules=num_modules,
                    block_type='mlp',
                    dropout=0.1
                )
                self.modular_layers.append(ModularLayer(config))
                prev_size = hidden_size
            
            # Final classification layer
            self.classifier = nn.Linear(hidden_sizes[-1], num_classes)
            
        def forward(self, x):
            # Flatten input if needed
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            # Pass through modular layers
            for layer in self.modular_layers:
                x = layer(x)  # Shape: [B, M, D]
                # For now, just take the first module output (later will use attention)
                x = x[:, 0, :]  # Shape: [B, D]
            
            # Final classification
            return self.classifier(x)
        
        def list_init_layers(self):
            """Compatible with existing framework initialization."""
            init_layers = []
            for layer in self.modular_layers:
                init_layers.extend(layer.list_init_layers())
            init_layers.append(self.classifier)
            return init_layers
    
    # Create and test the network
    network = ModularNetwork(input_size=784, hidden_sizes=[400, 400], num_classes=10, num_modules=4)
    print(f"Created modular network with {len(network.modular_layers)} modular layers")
    print(f"Total network parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(32, 784)  # Batch of 32 MNIST samples
    output = network(x)
    print(f"Network forward pass: {x.shape} -> {output.shape}")
    
    # Show initialization layers (compatible with existing framework)
    init_layers = network.list_init_layers()
    print(f"Initialization layers: {len(init_layers)} layers")
    for i, layer in enumerate(init_layers[:5]):  # Show first 5
        print(f"  Layer {i}: {type(layer).__name__}")
    
    return network


def example_attention_routing_preparation():
    """Example showing how the current implementation prepares for attention routing."""
    print("\n=== Preparation for Attention Routing ===")
    
    # Create a modular layer
    layer = create_mlp_modular_layer(in_dim=100, out_dim=50, num_modules=4)
    
    # Simulate input
    x = torch.randn(32, 100)
    
    # Current: Get all module outputs
    all_outputs = layer(x)  # Shape: [32, 4, 50]
    print(f"All module outputs shape: {all_outputs.shape}")
    
    # Future: This is where attention routing will be added
    # The attention mechanism will compute weights for each module
    # and combine the outputs accordingly
    
    # For demonstration, show how attention weights might be applied
    # (This is just a placeholder - real attention will be learned)
    attention_weights = torch.softmax(torch.randn(32, 4), dim=1)  # [B, M]
    print(f"Example attention weights shape: {attention_weights.shape}")
    print(f"Attention weights sum per sample: {attention_weights.sum(dim=1)}")
    
    # Apply attention weights (this is what the future router will do)
    attended_output = torch.sum(all_outputs * attention_weights.unsqueeze(-1), dim=1)
    print(f"Attended output shape: {attended_output.shape}")
    
    print("\nNote: The current implementation provides all module outputs.")
    print("Future attention routing will learn to combine these outputs")
    print("based on input characteristics and task requirements.")


def main():
    """Run all examples."""
    print("ModularLayer Examples for Continual Learning FYP")
    print("=" * 50)
    
    # Run examples
    mlp_layer = example_mlp_modular_layer()
    conv_layer = example_conv_modular_layer()
    network = example_integration_with_existing_framework()
    example_attention_routing_preparation()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("\nNext steps for your FYP:")
    print("1. Integrate ModularLayer into your existing continual learning pipeline")
    print("2. Implement attention-based routing mechanism")
    print("3. Train on SplitMNIST and compare with baseline methods (SI/EWC/ER)")
    print("4. Extend to CIFAR-10/100 for more complex scenarios")


if __name__ == "__main__":
    main()
