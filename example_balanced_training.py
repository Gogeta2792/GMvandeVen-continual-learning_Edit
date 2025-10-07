"""
Example Usage of Balanced Training Schedule

This script demonstrates how to use the balanced training system for modular
continual learning with proper loss balancing and router stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from train_splitmnist_balanced import BalancedModularNetwork, BalancedTrainer, create_balanced_args


def create_simple_data(num_tasks=3, samples_per_task=500, input_dim=784, num_classes=2):
    """Create simple synthetic data for demonstration."""
    train_tasks = []
    test_tasks = []
    
    for task_id in range(num_tasks):
        # Create task-specific data
        X_train = torch.randn(samples_per_task, input_dim)
        y_train = torch.randint(0, num_classes, (samples_per_task,))
        
        X_test = torch.randn(samples_per_task // 2, input_dim)
        y_test = torch.randint(0, num_classes, (samples_per_task // 2,))
        
        train_tasks.append((X_train, y_train))
        test_tasks.append((X_test, y_test))
    
    return train_tasks, test_tasks


def demonstrate_basic_usage():
    """Demonstrate basic usage of balanced training."""
    print("=" * 60)
    print("BASIC BALANCED TRAINING DEMONSTRATION")
    print("=" * 60)
    
    # Create simple data
    train_tasks, test_tasks = create_simple_data(num_tasks=3, samples_per_task=200)
    
    # Create model
    device = torch.device('cpu')
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400, 400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        router_entropy_coef=0.01,
        router_drift_coef=0.1,
        sparsity_coef=0.001,
        device=device
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Router parameters: {sum(p.numel() for p in model.get_router_parameters()):,}")
    print(f"Module parameters: {sum(p.numel() for p in model.get_module_parameters()):,}")
    
    # Create trainer
    class SimpleArgs:
        lr_modules = 0.001
        lr_router = 0.0001
        freeze_router_steps = 100
        use_router = True
        log_dir = None
    
    args = SimpleArgs()
    trainer = BalancedTrainer(model, device, args)
    
    # Train on first task
    print("\nTraining on Task 1...")
    train_data, train_labels = train_tasks[0]
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=32, shuffle=True)
    
    for epoch in range(5):
        epoch_losses = {'loss_cls': 0, 'loss_router': 0, 'loss_sparsity': 0, 'total_loss': 0}
        
        for batch_idx, (data, target) in enumerate(train_loader):
            step_stats = trainer.train_step(data, target, task_id=0)
            
            for key in epoch_losses:
                epoch_losses[key] += step_stats[key]
        
        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        print(f"  Epoch {epoch}: Loss={avg_losses['total_loss']:.4f}, "
              f"CE={avg_losses['loss_cls']:.4f}, "
              f"Router={avg_losses['loss_router']:.4f}, "
              f"Sparsity={avg_losses['loss_sparsity']:.4f}")
    
    print("Basic demonstration completed!")


def demonstrate_loss_components():
    """Demonstrate different loss components."""
    print("\n" + "=" * 60)
    print("LOSS COMPONENTS DEMONSTRATION")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Test different entropy coefficients
    entropy_coefs = [0.0, 0.01, 0.1]
    
    for entropy_coef in entropy_coefs:
        print(f"\nTesting entropy_coef = {entropy_coef}")
        
        model = BalancedModularNetwork(
            input_size=784,
            hidden_sizes=[400],
            num_classes=2,
            num_modules=4,
            use_router=True,
            router_entropy_coef=entropy_coef,
            router_drift_coef=0.1,
            sparsity_coef=0.001,
            device=device
        )
        
        # Create dummy data
        x = torch.randn(32, 784)
        y = torch.randint(0, 2, (32,))
        
        # Forward pass
        output, regularizers, attention = model(x, return_regularizers=True, return_attention=True)
        
        # Compute losses
        loss_cls = nn.CrossEntropyLoss()(output, y)
        loss_router = sum(regularizers.values()) if regularizers else 0.0
        loss_sparsity = model.compute_sparsity_loss(attention)
        
        print(f"  Classification loss: {loss_cls.item():.4f}")
        print(f"  Router loss: {loss_router.item():.4f}")
        print(f"  Sparsity loss: {loss_sparsity.item():.4f}")
        print(f"  Total loss: {(loss_cls + loss_router + loss_sparsity).item():.4f}")
        
        # Show attention patterns
        if attention:
            for layer_name, attn in attention.items():
                entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean()
                print(f"  {layer_name} attention entropy: {entropy.item():.4f}")


def demonstrate_router_collapse_detection():
    """Demonstrate router collapse detection."""
    print("\n" + "=" * 60)
    print("ROUTER COLLAPSE DETECTION DEMONSTRATION")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Create model with high entropy coefficient to encourage collapse
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        router_entropy_coef=0.0,  # No entropy penalty - might collapse
        router_drift_coef=0.0,    # No drift penalty
        sparsity_coef=0.0,        # No sparsity penalty
        device=device
    )
    
    # Simulate training steps
    x = torch.randn(32, 784)
    
    print("Simulating training steps...")
    for step in range(10):
        # Forward pass
        output, regularizers, attention = model(x, return_regularizers=True, return_attention=True)
        
        # Check for collapse
        collapsed = model.check_router_collapse(attention)
        
        if attention:
            for layer_name, attn in attention.items():
                max_attn = attn.max(dim=1)[0].mean()
                entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean()
                print(f"  Step {step}: {layer_name} max_attention={max_attn:.4f}, "
                      f"entropy={entropy:.4f}, collapsed={collapsed}")
        
        if collapsed:
            print(f"  Router collapsed detected at step {step}!")
            break


def demonstrate_curriculum_learning():
    """Demonstrate curriculum learning with router freezing."""
    print("\n" + "=" * 60)
    print("CURRICULUM LEARNING DEMONSTRATION")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        device=device
    )
    
    print("Initial state:")
    print(f"  Router frozen: {model.router_frozen}")
    
    # Freeze router
    model.freeze_router(freeze=True)
    print("After freezing router:")
    print(f"  Router frozen: {model.router_frozen}")
    
    # Check parameter gradients
    router_params = model.get_router_parameters()
    print(f"  Router parameters require_grad: {[p.requires_grad for p in router_params[:3]]}")
    
    # Unfreeze router
    model.freeze_router(freeze=False)
    print("After unfreezing router:")
    print(f"  Router frozen: {model.router_frozen}")
    print(f"  Router parameters require_grad: {[p.requires_grad for p in router_params[:3]]}")


def demonstrate_two_optimizer_system():
    """Demonstrate two-optimizer system."""
    print("\n" + "=" * 60)
    print("TWO-OPTIMIZER SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        device=device
    )
    
    # Create optimizers
    opt_modules = optim.Adam(model.get_module_parameters(), lr=0.001)
    opt_router = optim.Adam(model.get_router_parameters(), lr=0.0001)
    
    print(f"Module optimizer: {opt_modules}")
    print(f"Router optimizer: {opt_router}")
    print(f"Module parameters: {len(list(model.get_module_parameters()))}")
    print(f"Router parameters: {len(list(model.get_router_parameters()))}")
    
    # Simulate training step
    x = torch.randn(32, 784)
    y = torch.randint(0, 2, (32,))
    
    # Forward pass
    output, regularizers, attention = model(x, return_regularizers=True, return_attention=True)
    
    # Compute losses
    loss_cls = nn.CrossEntropyLoss()(output, y)
    loss_router = sum(regularizers.values()) if regularizers else 0.0
    loss_sparsity = model.compute_sparsity_loss(attention)
    total_loss = loss_cls + loss_router + loss_sparsity
    
    print(f"\nLosses:")
    print(f"  Classification: {loss_cls.item():.4f}")
    print(f"  Router: {loss_router.item():.4f}")
    print(f"  Sparsity: {loss_sparsity.item():.4f}")
    print(f"  Total: {total_loss.item():.4f}")
    
    # Backward pass
    opt_modules.zero_grad()
    opt_router.zero_grad()
    total_loss.backward()
    
    # Check gradients
    module_grads = [p.grad.norm().item() for p in model.get_module_parameters() if p.grad is not None]
    router_grads = [p.grad.norm().item() for p in model.get_router_parameters() if p.grad is not None]
    
    print(f"\nGradient norms:")
    print(f"  Module gradients: {module_grads[:3]}...")
    print(f"  Router gradients: {router_grads[:3]}...")
    
    # Update parameters
    opt_modules.step()
    opt_router.step()
    
    print("Two-optimizer system demonstration completed!")


def main():
    """Run all demonstrations."""
    print("BALANCED TRAINING SYSTEM DEMONSTRATIONS")
    print("=" * 80)
    
    # Run demonstrations
    demonstrate_basic_usage()
    demonstrate_loss_components()
    demonstrate_router_collapse_detection()
    demonstrate_curriculum_learning()
    demonstrate_two_optimizer_system()
    
    print("\n" + "=" * 80)
    print("ALL DEMONSTRATIONS COMPLETED!")
    print("=" * 80)
    
    print("\nTo run the full training script:")
    print("python train_splitmnist_balanced.py --use_router --gpu --lr_modules 0.001 --lr_router 0.0001")


if __name__ == "__main__":
    main()
