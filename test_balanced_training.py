"""
Test script for balanced training implementation.

This script verifies that the balanced training system works correctly
and produces expected results.
"""

import torch
import torch.nn as nn
import numpy as np
from train_splitmnist_balanced import BalancedModularNetwork, BalancedTrainer


def test_model_creation():
    """Test that the model can be created correctly."""
    print("Testing model creation...")
    
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400, 400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        device='cpu'
    )
    
    # Check parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    router_params = sum(p.numel() for p in model.get_router_parameters())
    module_params = sum(p.numel() for p in model.get_module_parameters())
    
    assert total_params > 0, "Model should have parameters"
    assert router_params > 0, "Router should have parameters"
    assert module_params > 0, "Modules should have parameters"
    assert total_params == router_params + module_params, "Parameter counts should match"
    
    print(f"[OK] Model creation successful: {total_params:,} total parameters")
    return model


def test_forward_pass():
    """Test forward pass with different return options."""
    print("Testing forward pass...")
    
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        device='cpu'
    )
    
    x = torch.randn(32, 784)
    
    # Test basic forward pass
    output = model(x)
    assert output.shape == (32, 2), f"Expected shape (32, 2), got {output.shape}"
    
    # Test with regularizers
    output, regularizers = model(x, return_regularizers=True)
    assert isinstance(regularizers, dict), "Regularizers should be a dictionary"
    assert 'layer_0_entropy_penalty' in regularizers, "Should have entropy penalty"
    assert 'layer_0_drift_penalty' in regularizers, "Should have drift penalty"
    
    # Test with attention
    output, attention = model(x, return_attention=True)
    assert isinstance(attention, dict), "Attention should be a dictionary"
    assert 'layer_0' in attention, "Should have layer attention"
    assert attention['layer_0'].shape == (32, 4), f"Expected attention shape (32, 4), got {attention['layer_0'].shape}"
    
    # Test attention sums to 1
    attn_sum = attention['layer_0'].sum(dim=1)
    assert torch.allclose(attn_sum, torch.ones(32), atol=1e-6), "Attention should sum to 1"
    
    print("[OK] Forward pass successful")


def test_loss_computation():
    """Test loss computation components."""
    print("Testing loss computation...")
    
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        sparsity_coef=0.001,
        device='cpu'
    )
    
    x = torch.randn(32, 784)
    y = torch.randint(0, 2, (32,))
    
    # Forward pass
    output, regularizers, attention = model(x, return_regularizers=True, return_attention=True)
    
    # Classification loss
    loss_cls = nn.CrossEntropyLoss()(output, y)
    assert loss_cls.item() > 0, "Classification loss should be positive"
    
    # Router loss
    loss_router = sum(regularizers.values())
    assert loss_router.item() >= 0, "Router loss should be non-negative"
    
    # Sparsity loss
    loss_sparsity = model.compute_sparsity_loss(attention)
    assert loss_sparsity.item() >= 0, "Sparsity loss should be non-negative"
    
    # Total loss
    total_loss = loss_cls + loss_router + loss_sparsity
    assert total_loss.item() > 0, "Total loss should be positive"
    
    print(f"[OK] Loss computation successful: CE={loss_cls.item():.4f}, "
          f"Router={loss_router.item():.4f}, Sparsity={loss_sparsity.item():.4f}")


def test_router_collapse_detection():
    """Test router collapse detection."""
    print("Testing router collapse detection...")
    
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        device='cpu'
    )
    
    # Create normal attention (should not collapse)
    normal_attention = {
        'layer_0': torch.softmax(torch.randn(32, 4), dim=1)
    }
    assert not model.check_router_collapse(normal_attention), "Normal attention should not collapse"
    
    # Create collapsed attention (should collapse)
    collapsed_attention = {
        'layer_0': torch.tensor([[0.96, 0.01, 0.01, 0.01]] * 32)
    }
    assert model.check_router_collapse(collapsed_attention), "Collapsed attention should be detected"
    
    print("[OK] Router collapse detection successful")


def test_curriculum_learning():
    """Test curriculum learning with router freezing."""
    print("Testing curriculum learning...")
    
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        device='cpu'
    )
    
    # Test initial state
    assert not model.router_frozen, "Router should not be frozen initially"
    
    # Test freezing
    model.freeze_router(freeze=True)
    assert model.router_frozen, "Router should be frozen"
    
    # Check parameter gradients
    router_params = model.get_router_parameters()
    for param in router_params:
        assert not param.requires_grad, "Router parameters should not require grad when frozen"
    
    # Test unfreezing
    model.freeze_router(freeze=False)
    assert not model.router_frozen, "Router should not be frozen after unfreezing"
    
    for param in router_params:
        assert param.requires_grad, "Router parameters should require grad when unfrozen"
    
    print("[OK] Curriculum learning successful")


def test_two_optimizer_system():
    """Test two-optimizer system."""
    print("Testing two-optimizer system...")
    
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        device='cpu'
    )
    
    # Create optimizers
    opt_modules = torch.optim.Adam(model.get_module_parameters(), lr=0.001)
    opt_router = torch.optim.Adam(model.get_router_parameters(), lr=0.0001)
    
    # Test parameter separation
    module_params = set(id(p) for p in model.get_module_parameters())
    router_params = set(id(p) for p in model.get_router_parameters())
    assert len(module_params.intersection(router_params)) == 0, "Module and router parameters should be disjoint"
    
    # Test training step
    x = torch.randn(32, 784)
    y = torch.randint(0, 2, (32,))
    
    # Forward pass
    output, regularizers, attention = model(x, return_regularizers=True, return_attention=True)
    
    # Compute loss
    loss_cls = nn.CrossEntropyLoss()(output, y)
    loss_router = sum(regularizers.values())
    loss_sparsity = model.compute_sparsity_loss(attention)
    total_loss = loss_cls + loss_router + loss_sparsity
    
    # Backward pass
    opt_modules.zero_grad()
    opt_router.zero_grad()
    total_loss.backward()
    
    # Check gradients exist
    module_grads = [p.grad for p in model.get_module_parameters() if p.grad is not None]
    router_grads = [p.grad for p in model.get_router_parameters() if p.grad is not None]
    
    assert len(module_grads) > 0, "Module parameters should have gradients"
    assert len(router_grads) > 0, "Router parameters should have gradients"
    
    # Update parameters
    opt_modules.step()
    opt_router.step()
    
    print("[OK] Two-optimizer system successful")


def test_trainer_integration():
    """Test trainer integration."""
    print("Testing trainer integration...")
    
    model = BalancedModularNetwork(
        input_size=784,
        hidden_sizes=[400],
        num_classes=2,
        num_modules=4,
        use_router=True,
        device='cpu'
    )
    
    # Create simple args
    class SimpleArgs:
        lr_modules = 0.001
        lr_router = 0.0001
        freeze_router_steps = 100
        use_router = True
        log_dir = None
    
    args = SimpleArgs()
    trainer = BalancedTrainer(model, 'cpu', args)
    
    # Test training step
    x = torch.randn(32, 784)
    y = torch.randint(0, 2, (32,))
    
    step_stats = trainer.train_step(x, y, task_id=0)
    
    # Check step stats
    required_keys = ['loss_cls', 'loss_router', 'loss_sparsity', 'total_loss', 'router_collapsed', 'router_frozen']
    for key in required_keys:
        assert key in step_stats, f"Step stats should contain {key}"
    
    assert step_stats['total_loss'] > 0, "Total loss should be positive"
    assert isinstance(step_stats['router_collapsed'], bool), "Router collapsed should be boolean"
    assert isinstance(step_stats['router_frozen'], bool), "Router frozen should be boolean"
    
    print("[OK] Trainer integration successful")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING BALANCED TRAINING TESTS")
    print("=" * 60)
    
    try:
        test_model_creation()
        test_forward_pass()
        test_loss_computation()
        test_router_collapse_detection()
        test_curriculum_learning()
        test_two_optimizer_system()
        test_trainer_integration()
        
        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
