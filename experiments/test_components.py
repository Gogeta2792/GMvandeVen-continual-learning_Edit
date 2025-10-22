"""
Test script to validate experiment components without running full pipeline.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from experiments import utils, metrics, expansion, train_splitmnist
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utils...")
    
    try:
        from experiments.utils import set_global_seeds, get_env_metadata, auto_device
        
        # Test seed setting
        set_global_seeds(42)
        print("✓ Seed setting works")
        
        # Test environment metadata
        env = get_env_metadata()
        assert 'timestamp' in env
        assert 'torch_version' in env
        print(f"✓ Environment metadata: {list(env.keys())}")
        
        # Test device detection
        device = auto_device()
        assert device in ['cuda', 'cpu']
        print(f"✓ Device auto-detection: {device}")
        
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metric functions."""
    print("\nTesting metrics...")
    
    try:
        from experiments.metrics import (
            compute_forgetting, sanity_check_task0, build_csv_row
        )
        
        # Test forgetting computation
        task_history = [
            [0.95],
            [0.92, 0.94],
            [0.88, 0.91, 0.96]
        ]
        forgetting = compute_forgetting(task_history, num_tasks=3)
        assert isinstance(forgetting, float)
        print(f"✓ Forgetting computation: {forgetting:.4f}")
        
        # Test sanity check
        passed, notes = sanity_check_task0(0.95)
        assert passed
        print(f"✓ Sanity check: passed={passed}")
        
        # Test CSV row building
        config = {
            'seed': 0,
            'router': 'none',
            'module_expansion': 'off',
            'num_tasks': 5,
            'epochs_per_task': 5,
            'batch_size': 64,
            'lr': 3e-4,
            'optimizer': 'adam',
        }
        env_metadata = {
            'timestamp': '20250422_120000',
            'commit_sha': 'abc123',
            'device_name': 'cpu'
        }
        row = build_csv_row(
            config=config,
            env_metadata=env_metadata,
            task_accs=[0.98, 0.96, 0.95, 0.94, 0.93],
            forgetting=0.02,
            final_params=634402,
            peak_params=634402,
            train_time_s=120.5,
            peak_vram_mb=None,
            notes=""
        )
        assert 'final_avg_acc' in row
        print(f"✓ CSV row building: {len(row)} columns")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_expansion():
    """Test expansion logic."""
    print("\nTesting expansion...")
    
    try:
        from experiments.expansion import (
            ExpansionTracker, compute_confidence_from_attention
        )
        import torch
        
        # Test expansion tracker
        tracker = ExpansionTracker(
            threshold=0.3,
            cooldown_tasks=1,
            max_modules=3,
            initial_modules=1
        )
        
        # Update with low confidence
        for _ in range(50):
            tracker.update_confidence(0.2)
        
        # Check if expansion triggers
        should_expand = tracker.check_trigger()
        print(f"✓ Expansion tracker: should_expand={should_expand}, "
              f"confidence={tracker.get_average_confidence():.4f}")
        
        # Test confidence computation
        attention = torch.softmax(torch.randn(10, 4), dim=-1)
        confidence = compute_confidence_from_attention(attention, method='max_weight')
        assert 0.0 <= confidence <= 1.0
        print(f"✓ Confidence computation: {confidence:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Expansion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from experiments.train_splitmnist import SimpleModularMLP, NullRouter
        import torch
        
        # Test null router
        router = NullRouter(num_modules=3)
        x = torch.randn(8, 784)
        logits, weights, regs = router(x)
        assert weights.shape == (8, 3)
        print(f"✓ Null router: output shape {weights.shape}")
        
        # Test simple MLP
        model = SimpleModularMLP(
            input_dim=784,
            hidden_dim=100,
            num_tasks=5,
            classes_per_task=2,
            use_router=False,
            num_modules=1,
            device='cpu'
        )
        x = torch.randn(4, 1, 28, 28)
        features, attn = model(x)
        assert features.shape == (4, 100)
        print(f"✓ Simple MLP: feature shape {features.shape}")
        
        # Test with router
        model_with_router = SimpleModularMLP(
            input_dim=784,
            hidden_dim=100,
            num_tasks=5,
            classes_per_task=2,
            use_router=True,
            num_modules=2,
            device='cpu'
        )
        features, attn = model_with_router(x, return_attention=True)
        assert features.shape == (4, 100)
        print(f"✓ MLP with router: feature shape {features.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading."""
    print("\nTesting data loading...")
    
    try:
        from experiments.train_splitmnist import RelabeledSubset
        from torchvision import datasets, transforms
        
        # Create a dummy dataset
        transform = transforms.ToTensor()
        dataset = datasets.FakeData(
            size=100,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform
        )
        
        # Test relabeled subset
        indices = list(range(20))
        subset = RelabeledSubset(dataset, indices, base_class=2)
        
        assert len(subset) == 20
        x, y = subset[0]
        assert y in [0, 1]  # Should be relabeled
        print(f"✓ Relabeled subset: {len(subset)} samples, label range [0, 1]")
        
        return True
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("Component Validation Tests")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Utils", test_utils),
        ("Metrics", test_metrics),
        ("Expansion", test_expansion),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    print("="*70)
    
    return total_passed == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

