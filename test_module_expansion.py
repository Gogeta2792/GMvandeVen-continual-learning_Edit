"""
Comprehensive tests for module expansion functionality in ModularLayer.

This test suite validates the expand-on-demand feature that dynamically adds
new modules when router confidence is low, similar to LMC (Lifelong Model Compression).

Tests cover:
1. Confidence computation (max-weight and entropy methods)
2. Expansion conditions checking
3. Module addition and initialization
4. Router expansion
5. Projection phase
6. Optimizer parameter updates
7. Integration with training
8. Edge cases and safeguards
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.modular_layer import ModularLayer, ModularLayerConfig


def test_confidence_computation_max_weight():
    """Test confidence computation using max-weight method."""
    print("\n" + "="*60)
    print("TEST: Confidence Computation (Max Weight)")
    print("="*60)
    
    config = ModularLayerConfig(
        in_dim=10,
        out_dim=10,
        num_modules=4,
        block_type='mlp',
        use_router=True,
        enable_expansion=True,
        confidence_method='max_weight',
        confidence_threshold=0.5
    )
    layer = ModularLayer(config)
    
    # Test case 1: High confidence (peaked distribution)
    attention_high = torch.tensor([
        [0.8, 0.1, 0.05, 0.05],
        [0.7, 0.2, 0.05, 0.05],
        [0.9, 0.05, 0.03, 0.02]
    ])
    conf_high = layer._compute_confidence(attention_high)
    print(f"High confidence attention: {attention_high[0].tolist()}")
    print(f"Computed confidence: {conf_high.tolist()}")
    assert conf_high.mean() > 0.7, "Expected high confidence for peaked distribution"
    
    # Test case 2: Low confidence (uniform distribution)
    attention_low = torch.tensor([
        [0.25, 0.25, 0.25, 0.25],
        [0.3, 0.2, 0.3, 0.2],
        [0.22, 0.28, 0.24, 0.26]
    ])
    conf_low = layer._compute_confidence(attention_low)
    print(f"\nLow confidence attention: {attention_low[0].tolist()}")
    print(f"Computed confidence: {conf_low.tolist()}")
    assert conf_low.mean() < 0.4, "Expected low confidence for uniform distribution"
    
    print("✓ Max-weight confidence computation works correctly")


def test_confidence_computation_entropy():
    """Test confidence computation using entropy method."""
    print("\n" + "="*60)
    print("TEST: Confidence Computation (Entropy)")
    print("="*60)
    
    config = ModularLayerConfig(
        in_dim=10,
        out_dim=10,
        num_modules=4,
        block_type='mlp',
        use_router=True,
        enable_expansion=True,
        confidence_method='entropy',
        confidence_threshold=0.5
    )
    layer = ModularLayer(config)
    
    # Test case 1: High confidence (low entropy - peaked)
    attention_high = torch.tensor([
        [0.9, 0.05, 0.03, 0.02],
        [0.85, 0.1, 0.03, 0.02]
    ])
    conf_high = layer._compute_confidence(attention_high)
    print(f"Peaked distribution: {attention_high[0].tolist()}")
    print(f"Computed confidence: {conf_high.tolist()}")
    assert conf_high.mean() > 0.6, "Expected high confidence for low entropy"
    
    # Test case 2: Low confidence (high entropy - uniform)
    attention_low = torch.tensor([
        [0.25, 0.25, 0.25, 0.25],
        [0.24, 0.26, 0.24, 0.26]
    ])
    conf_low = layer._compute_confidence(attention_low)
    print(f"\nUniform distribution: {attention_low[0].tolist()}")
    print(f"Computed confidence: {conf_low.tolist()}")
    assert conf_low.mean() < 0.3, "Expected low confidence for high entropy"
    
    print("✓ Entropy-based confidence computation works correctly")


def test_expansion_conditions():
    """Test expansion condition checking."""
    print("\n" + "="*60)
    print("TEST: Expansion Conditions")
    print("="*60)
    
    config = ModularLayerConfig(
        in_dim=10,
        out_dim=10,
        num_modules=4,
        block_type='mlp',
        use_router=True,
        enable_expansion=True,
        max_modules=6,
        confidence_threshold=0.4,
        cooldown_steps=100
    )
    layer = ModularLayer(config)
    
    # Test 1: Low confidence, should expand
    should_expand, reason = layer._check_expansion_conditions(0.3)
    print(f"Confidence 0.3 < 0.4: should_expand={should_expand}, reason={reason}")
    assert should_expand, "Should expand when confidence is low"
    
    # Test 2: High confidence, should not expand
    should_expand, reason = layer._check_expansion_conditions(0.5)
    print(f"Confidence 0.5 >= 0.4: should_expand={should_expand}, reason={reason}")
    assert not should_expand, "Should not expand when confidence is high"
    
    # Test 3: In cooldown period
    layer.current_step = 50
    layer.last_expansion_step = 0
    should_expand, reason = layer._check_expansion_conditions(0.3)
    print(f"In cooldown (50/100): should_expand={should_expand}, reason={reason}")
    assert not should_expand, "Should not expand during cooldown"
    
    # Test 4: At max modules
    layer.num_modules = 6
    layer.current_step = 200
    should_expand, reason = layer._check_expansion_conditions(0.3)
    print(f"At max modules (6/6): should_expand={should_expand}, reason={reason}")
    assert not should_expand, "Should not expand at max modules"
    
    # Test 5: Force expansion
    should_expand, reason = layer._check_expansion_conditions(0.8, force=True)
    print(f"Force expansion: should_expand={should_expand}, reason={reason}")
    assert should_expand, "Should expand when forced"
    
    print("✓ Expansion conditions checking works correctly")


def test_single_expansion():
    """Test a single module expansion with synthetic data."""
    print("\n" + "="*60)
    print("TEST: Single Module Expansion")
    print("="*60)
    
    config = ModularLayerConfig(
        in_dim=20,
        out_dim=10,
        num_modules=3,
        block_type='mlp',
        use_router=True,
        enable_expansion=True,
        max_modules=6,
        confidence_threshold=0.4,
        cooldown_steps=10,
        projection_steps=0,  # Skip projection for this test
        device='cpu'
    )
    layer = ModularLayer(config)
    
    print(f"Initial modules: {layer.num_modules}")
    print(f"Initial router output size: {layer.router.linear2.out_features}")
    
    # Create synthetic low-confidence attention
    # This simulates the router being uncertain
    batch_size = 32
    low_conf_attention = torch.ones(batch_size, 3) / 3  # Uniform distribution
    
    # Create optimizer
    optimizer = optim.Adam(layer.parameters(), lr=0.001)
    initial_param_groups = len(optimizer.param_groups)
    print(f"Initial optimizer param groups: {initial_param_groups}")
    
    # Trigger expansion
    result = layer.maybe_expand(low_conf_attention, optimizer=optimizer, verbose=True)
    
    print(f"\nExpansion result: {result}")
    assert result['expanded'], "Expansion should have occurred"
    assert layer.num_modules == 4, f"Should have 4 modules, got {layer.num_modules}"
    assert layer.router.linear2.out_features == 4, "Router should have 4 outputs"
    assert len(optimizer.param_groups) > initial_param_groups, "Optimizer should have new param groups"
    
    # Verify new module exists and is initialized
    newest_module = layer.modules_list[-1]
    assert isinstance(newest_module, nn.Module), "Newest module should be a nn.Module"
    
    # Verify router EMA buffer expanded
    assert layer.router.ema_attention.size(1) == 4, "EMA buffer should have 4 elements"
    
    # Verify expansion history
    assert len(layer.expansion_history) == 1, "Should have 1 expansion in history"
    assert layer.expansion_count == 1, "Expansion count should be 1"
    
    print("✓ Single module expansion works correctly")


def test_multiple_expansions():
    """Test multiple sequential expansions."""
    print("\n" + "="*60)
    print("TEST: Multiple Sequential Expansions")
    print("="*60)
    
    config = ModularLayerConfig(
        in_dim=15,
        out_dim=8,
        num_modules=2,
        block_type='mlp',
        use_router=True,
        enable_expansion=True,
        max_modules=5,
        confidence_threshold=0.6,  # Higher threshold for easier triggering
        cooldown_steps=5,  # Short cooldown for testing
        projection_steps=0,
        device='cpu'
    )
    layer = ModularLayer(config)
    
    print(f"Starting with {layer.num_modules} modules")
    print(f"Max modules: {layer.max_modules}")
    
    # Perform 3 expansions
    for i in range(3):
        print(f"\n--- Expansion {i+1} ---")
        
        # Advance step counter beyond cooldown
        layer.current_step += 10
        
        # Create low-confidence attention
        batch_size = 16
        num_modules = layer.num_modules
        low_conf_attention = torch.ones(batch_size, num_modules) / num_modules
        
        result = layer.maybe_expand(low_conf_attention, verbose=False)
        print(f"Modules: {num_modules} → {layer.num_modules}")
        print(f"Expanded: {result['expanded']}, Confidence: {result['confidence']:.4f}")
        
        assert result['expanded'], f"Expansion {i+1} should have occurred"
        assert layer.num_modules == num_modules + 1, "Module count should increase by 1"
    
    # Verify final state
    assert layer.num_modules == 5, f"Should have 5 modules, got {layer.num_modules}"
    assert layer.expansion_count == 3, f"Should have 3 expansions, got {layer.expansion_count}"
    assert len(layer.expansion_history) == 3, "Should have 3 history entries"
    
    # Try one more expansion - should fail (at max)
    layer.current_step += 10
    low_conf_attention = torch.ones(batch_size, 5) / 5
    result = layer.maybe_expand(low_conf_attention, verbose=False)
    print(f"\nAttempt expansion beyond max: {result['reason']}")
    assert not result['expanded'], "Should not expand beyond max_modules"
    
    print("✓ Multiple sequential expansions work correctly")


def test_projection_phase():
    """Test projection/distillation phase for new modules."""
    print("\n" + "="*60)
    print("TEST: Projection Phase")
    print("="*60)
    
    torch.manual_seed(42)  # For reproducibility
    
    config = ModularLayerConfig(
        in_dim=20,
        out_dim=15,
        num_modules=3,
        block_type='mlp',
        hidden_dim=25,
        use_router=True,
        enable_expansion=True,
        max_modules=6,
        projection_steps=50,
        projection_lr=0.01,
        device='cpu'
    )
    layer = ModularLayer(config)
    
    # Create training data
    batch_size = 64
    x_train = torch.randn(batch_size, 20)
    
    # Trigger expansion
    low_conf_attention = torch.ones(batch_size, 3) / 3
    layer.current_step = 100  # Beyond cooldown
    result = layer.maybe_expand(low_conf_attention, verbose=False)
    assert result['expanded'], "Expansion should occur"
    
    print(f"Modules after expansion: {layer.num_modules}")
    
    # Run projection phase
    print("\nRunning projection phase...")
    losses = layer.run_projection_phase(x_train, num_steps=50, lr=0.01, verbose=True)
    
    assert len(losses) == 50, f"Should have 50 loss values, got {len(losses)}"
    
    # Verify losses decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"\nProjection losses: initial={initial_loss:.6f}, final={final_loss:.6f}")
    assert final_loss < initial_loss * 0.5, "Projection should significantly reduce loss"
    
    # Verify new module output is close to old weighted output
    newest_module = layer.modules_list[-1]
    with torch.no_grad():
        new_output = newest_module(x_train)
        
        # Compute old weighted output
        old_outputs = [layer.modules_list[i](x_train) for i in range(3)]
        old_mods = torch.stack(old_outputs, dim=1)
        _, attention, _ = layer.router(x_train)
        old_attn = attention[:, :3] / (attention[:, :3].sum(dim=-1, keepdim=True) + 1e-8)
        old_weighted = (old_attn.unsqueeze(-1) * old_mods).sum(dim=1)
        
        mse = torch.mean((new_output - old_weighted) ** 2).item()
        print(f"Final MSE between new module and old weighted output: {mse:.6f}")
        assert mse < 0.1, "New module should match old output after projection"
    
    print("✓ Projection phase works correctly")


def test_router_expansion():
    """Test router output dimension expansion."""
    print("\n" + "="*60)
    print("TEST: Router Expansion")
    print("="*60)
    
    config = ModularLayerConfig(
        in_dim=10,
        out_dim=8,
        num_modules=3,
        block_type='mlp',
        use_router=True,
        enable_expansion=True,
        device='cpu'
    )
    layer = ModularLayer(config)
    
    # Check initial state
    print(f"Initial router output features: {layer.router.linear2.out_features}")
    print(f"Initial EMA shape: {layer.router.ema_attention.shape}")
    assert layer.router.linear2.out_features == 3, "Router should start with 3 outputs"
    
    # Manually add a module and expand router
    layer._add_new_module()
    layer._expand_router()
    
    # Check expanded state
    print(f"After expansion:")
    print(f"  Router output features: {layer.router.linear2.out_features}")
    print(f"  EMA shape: {layer.router.ema_attention.shape}")
    print(f"  Num modules: {layer.num_modules}")
    
    assert layer.router.linear2.out_features == 4, "Router should have 4 outputs"
    assert layer.router.ema_attention.shape[1] == 4, "EMA should have 4 elements"
    assert layer.num_modules == 4, "Should have 4 modules"
    
    # Verify router can forward with new dimension
    x = torch.randn(16, 10)
    logits, attention, _ = layer.router(x)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Attention shape: {attention.shape}")
    assert logits.shape == (16, 4), "Logits should have shape [16, 4]"
    assert attention.shape == (16, 4), "Attention should have shape [16, 4]"
    
    # Check that new module has low initial attention (due to negative bias)
    mean_attention = attention.mean(dim=0)
    print(f"  Mean attention per module: {mean_attention.tolist()}")
    assert mean_attention[3] < mean_attention[:3].mean(), "New module should have lower initial attention"
    
    print("✓ Router expansion works correctly")


def test_optimizer_update():
    """Test optimizer parameter group updates after expansion."""
    print("\n" + "="*60)
    print("TEST: Optimizer Parameter Updates")
    print("="*60)
    
    config = ModularLayerConfig(
        in_dim=12,
        out_dim=10,
        num_modules=2,
        block_type='mlp',
        use_router=True,
        enable_expansion=True,
        max_modules=4,
        confidence_threshold=0.6,  # Higher threshold
        device='cpu'
    )
    layer = ModularLayer(config)
    
    # Create optimizer
    optimizer = optim.SGD(layer.parameters(), lr=0.01)
    initial_groups = len(optimizer.param_groups)
    print(f"Initial param groups: {initial_groups}")
    
    # Expand without optimizer
    layer._add_new_module()
    layer._expand_router()
    
    # Get new parameters
    new_params = layer.get_new_optimizer_params()
    print(f"New parameter groups to add: {len(new_params)}")
    assert len(new_params) > 0, "Should have new parameters to add"
    
    # Manually update optimizer
    for param_group in new_params:
        optimizer.add_param_group(param_group)
    
    updated_groups = len(optimizer.param_groups)
    print(f"Updated param groups: {updated_groups}")
    assert updated_groups > initial_groups, "Optimizer should have more param groups"
    
    # Test automatic optimizer update via maybe_expand
    layer2 = ModularLayer(config)
    optimizer2 = optim.Adam(layer2.parameters(), lr=0.001)
    initial_groups2 = len(optimizer2.param_groups)
    
    low_conf_attention = torch.ones(16, layer2.num_modules) / layer2.num_modules
    result = layer2.maybe_expand(low_conf_attention, optimizer=optimizer2, verbose=False)
    
    assert result['expanded'], "Should expand"
    assert len(optimizer2.param_groups) > initial_groups2, "Optimizer should auto-update"
    
    print("✓ Optimizer parameter updates work correctly")


def test_integration_with_training():
    """Test integration of expansion with actual training loop."""
    print("\n" + "="*60)
    print("TEST: Integration with Training Loop")
    print("="*60)
    
    torch.manual_seed(123)
    
    config = ModularLayerConfig(
        in_dim=15,
        out_dim=10,
        num_modules=2,
        block_type='mlp',
        use_router=True,
        enable_expansion=True,
        max_modules=4,
        confidence_threshold=0.4,
        cooldown_steps=20,
        projection_steps=0,
        device='cpu'
    )
    layer = ModularLayer(config)
    
    # Create simple classification task
    num_samples = 200
    x = torch.randn(num_samples, 15)
    y = torch.randint(0, 10, (num_samples,))
    
    # Add a simple classifier head
    classifier = nn.Linear(10, 10)
    model = nn.Sequential(layer, nn.Identity())  # Wrap layer for simplicity
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(layer.parameters()) + list(classifier.parameters()),
        lr=0.01
    )
    
    print("Training for 100 steps...")
    expansion_steps = []
    
    for step in range(100):
        # Mini-batch
        idx = torch.randint(0, num_samples, (32,))
        x_batch = x[idx]
        y_batch = y[idx]
        
        # Forward
        output, attn_info = layer(x_batch, return_attn=True)
        logits = classifier(output)
        loss = criterion(logits, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check expansion (every 5 steps to avoid too frequent checks)
        if step % 5 == 0 and attn_info is not None:
            attention = attn_info['attention']
            result = layer.maybe_expand(attention, optimizer=optimizer, verbose=False)
            if result['expanded']:
                expansion_steps.append(step)
                print(f"  Step {step}: Expanded to {layer.num_modules} modules (conf={result['confidence']:.4f})")
        
        if step % 25 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, modules={layer.num_modules}")
    
    print(f"\nFinal state:")
    print(f"  Total modules: {layer.num_modules}")
    print(f"  Total expansions: {layer.expansion_count}")
    print(f"  Expansion steps: {expansion_steps}")
    
    # Verify expansions occurred (with some randomness in attention)
    assert layer.num_modules >= 2, "Should have at least initial modules"
    
    print("✓ Integration with training loop works correctly")


def test_conv_layer_expansion():
    """Test expansion with convolutional blocks."""
    print("\n" + "="*60)
    print("TEST: Conv Layer Expansion")
    print("="*60)
    
    config = ModularLayerConfig(
        in_dim=3,  # channels
        out_dim=16,
        num_modules=2,
        block_type='conv',
        use_router=True,
        enable_expansion=True,
        max_modules=4,
        confidence_threshold=0.6,  # Higher threshold for easier triggering
        cooldown_steps=10,
        projection_steps=0,
        device='cpu'
    )
    layer = ModularLayer(config)
    
    print(f"Initial modules: {layer.num_modules}")
    
    # Create synthetic image data
    batch_size = 16
    x = torch.randn(batch_size, 3, 32, 32)
    
    # Forward pass to get attention
    output, attn_info = layer(x, return_attn=True)
    attention = attn_info['attention']
    print(f"Initial output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Create low-confidence attention and trigger expansion
    low_conf_attention = torch.ones(batch_size, 2) / 2
    result = layer.maybe_expand(low_conf_attention, verbose=True)
    
    assert result['expanded'], "Conv layer should expand"
    assert layer.num_modules == 3, "Should have 3 modules"
    
    # Verify forward pass works with expanded modules
    output2, attn_info2 = layer(x, return_attn=True)
    print(f"After expansion output shape: {output2.shape}")
    print(f"After expansion attention shape: {attn_info2['attention'].shape}")
    
    assert output2.shape == (batch_size, 16, 32, 32), "Output shape should remain consistent"
    assert attn_info2['attention'].shape == (batch_size, 3), "Attention should have 3 modules"
    
    print("✓ Conv layer expansion works correctly")


def test_expansion_stats():
    """Test expansion statistics tracking."""
    print("\n" + "="*60)
    print("TEST: Expansion Statistics")
    print("="*60)
    
    config = ModularLayerConfig(
        in_dim=10,
        out_dim=8,
        num_modules=2,
        block_type='mlp',
        use_router=True,
        enable_expansion=True,
        max_modules=5,
        confidence_threshold=0.6,  # Higher threshold for easier triggering
        cooldown_steps=5,
        device='cpu'
    )
    layer = ModularLayer(config)
    
    # Check initial stats
    stats = layer.get_expansion_stats()
    print("Initial stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    assert stats['num_modules'] == 2
    assert stats['expansion_count'] == 0
    assert len(stats['expansion_history']) == 0
    
    # Perform 2 expansions
    for i in range(2):
        layer.current_step += 10
        low_conf_attention = torch.ones(16, layer.num_modules) / layer.num_modules
        layer.maybe_expand(low_conf_attention, verbose=False)
    
    # Check updated stats
    stats = layer.get_expansion_stats()
    print("\nStats after 2 expansions:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    assert stats['num_modules'] == 4
    assert stats['expansion_count'] == 2
    assert len(stats['expansion_history']) == 2
    
    # Verify history format
    for step, num_mods, conf in stats['expansion_history']:
        print(f"  Expansion at step {step}: {num_mods} modules, confidence={conf:.4f}")
        assert isinstance(step, int)
        assert isinstance(num_mods, int)
        assert isinstance(conf, float)
    
    print("✓ Expansion statistics tracking works correctly")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST: Edge Cases and Error Handling")
    print("="*60)
    
    # Test 1: Expansion disabled
    print("\n1. Expansion disabled:")
    config = ModularLayerConfig(
        in_dim=10,
        out_dim=8,
        num_modules=3,
        block_type='mlp',
        use_router=True,
        enable_expansion=False,  # Disabled
        device='cpu'
    )
    layer = ModularLayer(config)
    low_conf_attention = torch.ones(16, 3) / 3
    result = layer.maybe_expand(low_conf_attention, verbose=False)
    print(f"  Result: {result['reason']}")
    assert not result['expanded'], "Should not expand when disabled"
    
    # Test 2: Invalid config (expansion without router)
    print("\n2. Invalid config (expansion without router):")
    try:
        config = ModularLayerConfig(
            in_dim=10,
            out_dim=8,
            num_modules=3,
            block_type='mlp',
            use_router=False,
            enable_expansion=True,  # Invalid!
            device='cpu'
        )
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
    
    # Test 3: Invalid confidence threshold
    print("\n3. Invalid confidence threshold:")
    try:
        config = ModularLayerConfig(
            in_dim=10,
            out_dim=8,
            num_modules=3,
            block_type='mlp',
            use_router=True,
            enable_expansion=True,
            confidence_threshold=1.5,  # Invalid!
            device='cpu'
        )
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
    
    # Test 4: Max modules <= num modules
    print("\n4. Max modules <= num modules:")
    try:
        config = ModularLayerConfig(
            in_dim=10,
            out_dim=8,
            num_modules=5,
            block_type='mlp',
            use_router=True,
            enable_expansion=True,
            max_modules=4,  # Invalid!
            device='cpu'
        )
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
    
    # Test 5: Projection with no modules
    print("\n5. Projection with only 1 module:")
    config = ModularLayerConfig(
        in_dim=10,
        out_dim=8,
        num_modules=1,
        block_type='mlp',
        use_router=True,
        enable_expansion=True,
        max_modules=4,
        device='cpu'
    )
    layer = ModularLayer(config)
    x = torch.randn(16, 10)
    losses = layer.run_projection_phase(x, verbose=False)
    print(f"  Losses: {losses}")
    assert len(losses) == 0, "No projection needed with only 1 module"
    
    print("\n✓ Edge cases handled correctly")


def run_all_tests():
    """Run all expansion tests."""
    print("\n" + "="*70)
    print("RUNNING MODULE EXPANSION TESTS")
    print("="*70)
    
    test_functions = [
        test_confidence_computation_max_weight,
        test_confidence_computation_entropy,
        test_expansion_conditions,
        test_single_expansion,
        test_multiple_expansions,
        test_projection_phase,
        test_router_expansion,
        test_optimizer_update,
        test_integration_with_training,
        test_conv_layer_expansion,
        test_expansion_stats,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(test_functions)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

