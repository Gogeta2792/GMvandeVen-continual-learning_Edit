"""
SplitMNIST task-incremental training with modular networks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, List, Any, Optional
import time

from torchvision import datasets, transforms

from experiments.metrics import (
    compute_task_accs, compute_forgetting, count_parameters,
    try_peak_vram, sanity_check_task0, build_csv_row
)
from experiments.expansion import (
    ExpansionTracker, apply_expansion, compute_confidence_from_attention
)


class NullRouter(nn.Module):
    """
    Null router that returns fixed deterministic weights.
    This prevents capacity inflation by selecting a single path.
    """
    def __init__(self, num_modules: int):
        super().__init__()
        self.num_modules = num_modules
        # Use a fixed selection (e.g., first module)
        self.fixed_selection = 0
    
    def forward(self, x):
        batch_size = x.size(0)
        # Return one-hot for the fixed module
        weights = torch.zeros(batch_size, self.num_modules, device=x.device)
        weights[:, self.fixed_selection] = 1.0
        
        # Return logits, attention, and empty regularizers
        logits = torch.zeros(batch_size, self.num_modules, device=x.device)
        logits[:, self.fixed_selection] = 10.0  # High logit for selected module
        
        return logits, weights, {}


class SimpleModularMLP(nn.Module):
    """
    Simple MLP with optional modular layer and routing.
    Task-incremental (multi-head) architecture.
    """
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 400,
        num_tasks: int = 5,
        classes_per_task: int = 2,
        use_router: bool = False,
        num_modules: int = 1,
        device: str = 'cpu',
        expansion_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.use_router = use_router
        self.num_modules = num_modules
        self.device_str = device
        
        # Feature extractor - use modular layer for both router types
        if use_router or num_modules >= 1:
            # Use modular layer with appropriate router configuration
            from models.modular_layer import ModularLayerConfig, ModularLayer
            
            expansion_enabled = expansion_config is not None and expansion_config.get('module_expansion') == 'on'
            
            # Handle expansion constraints: expansion requires use_router=True
            # For router='none' with expansion, we need to enable router for expansion to work
            effective_use_router = use_router or (expansion_enabled and num_modules >= 1)
            
            # Get max_modules and ensure it's greater than num_modules for expansion
            max_modules = expansion_config.get('max_modules_per_layer', 8) if expansion_config else 8
            if expansion_enabled and max_modules <= num_modules:
                # If expansion is enabled but max_modules is not greater than num_modules,
                # either disable expansion or increase max_modules
                if num_modules == 1:
                    max_modules = 8  # Default to 8 for single module
                else:
                    max_modules = num_modules + 1  # At least one more module
            
            layer_config = ModularLayerConfig(
                in_dim=input_dim,
                out_dim=hidden_dim,
                num_modules=num_modules,
                block_type='mlp',
                hidden_dim=hidden_dim,
                use_router=effective_use_router,  # Enable router if expansion is needed
                enable_expansion=expansion_enabled,  # Enable expansion if configured
                max_modules=max_modules,
                confidence_threshold=expansion_config.get('expansion_threshold', 0.45) if expansion_config else 0.45,  # Updated default
                cooldown_steps=expansion_config.get('expansion_cooldown', 1) * 100 if expansion_config else 100,  # Convert tasks to steps
                device=device
            )
            self.feature_layer = ModularLayer(layer_config)
            
            # Store original router setting for reference
            self.original_use_router = use_router
            self.effective_use_router = effective_use_router
        else:
            # Fallback to standard MLP (shouldn't happen with new config)
            self.feature_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.original_use_router = use_router
            self.effective_use_router = use_router
        
        # Task-specific heads (multi-head for task-incremental)
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, classes_per_task)
            for _ in range(num_tasks)
        ])
        
        self.to(device)
    
    def forward(self, x, return_attention=False):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Feature extraction
        if isinstance(self.feature_layer, nn.Sequential):
            features = self.feature_layer(x)
            attention_info = None
        else:
            # Modular layer
            features, attention_info = self.feature_layer(x, return_attn=return_attention)
            
            # Handle case when no router is used - select first module output
            if not self.effective_use_router and features.dim() == 3:
                # features shape: [batch_size, num_modules, out_dim]
                # Select first module output: [batch_size, out_dim]
                features = features[:, 0, :]
        
        return features, attention_info
    
    def feature_extractor(self, x):
        """Extract features without task head."""
        features, _ = self.forward(x, return_attention=False)
        return features


def get_splitmnist_data(data_dir: str = './store/datasets'):
    """
    Load and split MNIST into 5 binary tasks.
    
    Returns:
        Tuple of (train_datasets, test_datasets) where each is a list of 5 datasets
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full MNIST
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    # Split into 5 tasks: {0,1}, {2,3}, {4,5}, {6,7}, {8,9}
    train_datasets = []
    test_datasets = []
    
    for task_id in range(5):
        class_0 = task_id * 2
        class_1 = task_id * 2 + 1
        
        # Get indices for these classes
        train_indices = [
            i for i, (_, label) in enumerate(train_dataset)
            if label in [class_0, class_1]
        ]
        test_indices = [
            i for i, (_, label) in enumerate(test_dataset)
            if label in [class_0, class_1]
        ]
        
        # Create subsets with relabeled targets (0 or 1)
        train_task_dataset = RelabeledSubset(
            train_dataset, train_indices, class_0
        )
        test_task_dataset = RelabeledSubset(
            test_dataset, test_indices, class_0
        )
        
        train_datasets.append(train_task_dataset)
        test_datasets.append(test_task_dataset)
    
    return train_datasets, test_datasets


class RelabeledSubset:
    """Dataset subset with relabeled classes."""
    
    def __init__(self, dataset, indices, base_class):
        self.dataset = dataset
        self.indices = indices
        self.base_class = base_class
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        # Relabel to 0 or 1
        y_new = y - self.base_class
        return x, y_new


def run_one_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single SplitMNIST experiment.
    
    Args:
        config: Configuration dictionary with all experiment parameters
    
    Returns:
        Dictionary with results for CSV row
    """
    from experiments.utils import set_global_seeds, get_env_metadata
    
    # Set seed for reproducibility
    set_global_seeds(config['seed'])
    
    # Get environment metadata
    env_metadata = get_env_metadata()
    
    # Device setup
    device = config.get('device', 'cpu')
    
    # Load data
    train_datasets, test_datasets = get_splitmnist_data(
        data_dir=config.get('data_dir', './store/datasets')
    )
    
    # Create model
    use_router = (config['router'] == 'attn')
    num_modules = config.get('initial_modules', 1)
    expansion_enabled = (config['module_expansion'] == 'on')
    
    # Configuration validation and diagnostics
    print(f"\n[CONFIG] Router: {config['router']}, Modules: {num_modules}, Expansion: {config['module_expansion']}")
    print(f"[CONFIG] use_router={use_router}, num_modules={num_modules}")
    
    # Note: When router='none' + expansion='on', the router is enabled internally
    # for expansion decisions (confidence computation), even though the configuration
    # is labeled as "no router". This is necessary because expansion relies on
    # attention weights to determine when to add new modules.
    if not use_router and expansion_enabled:
        print(f"[INFO] Router='none' + expansion='on': Router enabled internally for expansion mechanism")
    
    model = SimpleModularMLP(
        input_dim=784,
        hidden_dim=400,
        num_tasks=config['num_tasks'],
        classes_per_task=2,
        use_router=use_router,
        num_modules=num_modules,
        device=device,
        expansion_config=config if expansion_enabled else None
    )
    
    # Verify model architecture
    if hasattr(model.feature_layer, 'num_modules'):
        print(f"[VERIFY] Using ModularLayer with {model.feature_layer.num_modules} modules")
        print(f"[VERIFY] Original router setting: {model.original_use_router}")
        print(f"[VERIFY] Effective router setting: {model.effective_use_router}")
    else:
        print(f"[VERIFY] Using standard MLP (Sequential)")
    
    # Validate configuration consistency
    if config['router'] == 'attn' and num_modules < 2:
        print(f"[WARNING] Attention router with only {num_modules} module(s) - routing may be trivial")
    elif config['router'] == 'none' and num_modules > 1:
        print(f"[WARNING] No router with {num_modules} modules - using deterministic selection")
    elif config['router'] == 'none' and expansion_enabled:
        print(f"[INFO] Router='none' with expansion enabled - using router internally for expansion")
    
    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Expansion tracker
    expansion_tracker = None
    expansion_events = []
    
    if expansion_enabled:
        expansion_tracker = ExpansionTracker(
            threshold=config['expansion_threshold'],
            cooldown_tasks=config['expansion_cooldown'],
            max_modules=config['max_modules_per_layer'],
            initial_modules=num_modules
        )
    
    # Metrics tracking
    task_history = []  # List of accuracy lists after each task
    param_history = []
    start_time = time.time()
    
    # Reset CUDA memory stats if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Training loop over tasks
    for task_id in range(config['num_tasks']):
        print(f"\n{'='*60}")
        print(f"Training Task {task_id} (classes {task_id*2}-{task_id*2+1})")
        print(f"{'='*60}")
        
        # Get task data
        train_loader = DataLoader(
            train_datasets[task_id],
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        # Calculate number of iterations
        total_samples = len(train_datasets[task_id])
        iters_per_epoch = total_samples // config['batch_size']
        total_iters = iters_per_epoch * config['epochs_per_task']
        
        # Train on this task
        model.train()
        for epoch in range(config['epochs_per_task']):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                features, attention_info = model(x, return_attention=True)
                logits = model.task_heads[task_id](features)
                
                # Compute loss
                loss = criterion(logits, y)
                
                # Add router regularization if using attention
                if attention_info is not None and 'regularizers' in attention_info:
                    for reg_name, reg_value in attention_info['regularizers'].items():
                        loss += reg_value
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                epoch_correct += (predicted == y).sum().item()
                epoch_total += y.size(0)
                
                # Update expansion tracker confidence
                if expansion_tracker is not None and attention_info is not None:
                    if 'attention' in attention_info:
                        confidence = compute_confidence_from_attention(
                            attention_info['attention'], method='max_weight'
                        )
                        expansion_tracker.update_confidence(confidence)
            
            # Epoch stats
            epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            print(f"  Epoch {epoch+1}/{config['epochs_per_task']}: "
                  f"Loss={epoch_loss/len(train_loader):.4f}, Acc={epoch_acc:.4f}")
        
        # Track parameters before any expansion
        current_params = count_parameters(model)
        param_history.append(current_params)
        
        # Check if expansion should occur (after task training)
        if expansion_tracker is not None and hasattr(model.feature_layer, 'maybe_expand'):
            if expansion_tracker.check_trigger():
                print(f"\n[EXPANSION TRIGGERED at Task {task_id}]")
                print(f"  Avg Confidence: {expansion_tracker.get_average_confidence():.4f}")
                print(f"  Threshold: {expansion_tracker.threshold:.4f}")
                
                # Get current attention weights for expansion decision
                # Use dummy attention based on confidence
                avg_confidence = expansion_tracker.get_average_confidence()
                dummy_attention = torch.ones(1, model.feature_layer.num_modules, device=device) * avg_confidence
                
                # Apply real expansion to the modular layer
                expansion_result = model.feature_layer.maybe_expand(
                    attention=dummy_attention,
                    optimizer=optimizer,
                    force=False,
                    verbose=True
                )
                
                if expansion_result['expanded']:
                    # Count parameters after expansion
                    params_after = count_parameters(model)
                    params_added = params_after - current_params
                    
                    # Update expansion tracker
                    expansion_tracker.current_modules = expansion_result['num_modules']
                    expansion_tracker.tasks_since_expansion = 0
                    
                    event = {
                        'task_id': task_id,
                        'layer': 'feature_layer',
                        'new_module_id': expansion_result['num_modules'] - 1,
                        'params_added': params_added,
                        'confidence': expansion_result['confidence']
                    }
                    expansion_events.append(event)
                    print(f"  New module count: {expansion_result['num_modules']}")
                    print(f"  Parameters added: {params_added:,}")
                    print(f"  Total parameters: {params_after:,}\n")
                else:
                    print(f"  Expansion skipped: {expansion_result['reason']}\n")
            
            # Reset for next task
            expansion_tracker.reset_after_task()
        
        # Evaluate on all tasks seen so far
        task_accs = compute_task_accs(model, test_datasets[:task_id+1], device, task_id+1)
        task_history.append(task_accs)
        
        print(f"\nAfter Task {task_id}:")
        for i, acc in enumerate(task_accs):
            print(f"  Task {i} Accuracy: {acc:.4f}")
        print(f"  Average Accuracy: {np.mean(task_accs):.4f}")
        print(f"  Parameters: {count_parameters(model):,}")
    
    # Final evaluation
    end_time = time.time()
    train_time_s = end_time - start_time
    
    # Final metrics - recompute parameters to ensure accuracy after any expansions
    final_task_accs = task_history[-1]
    final_forgetting = compute_forgetting(task_history, config['num_tasks'])
    
    # Recompute final parameters to ensure accuracy after expansions
    final_params = count_parameters(model)
    peak_params = max(param_history + [final_params])  # Include final count in peak calculation
    peak_vram_mb = try_peak_vram()
    
    print(f"\nFinal parameter count: {final_params:,}")
    print(f"Peak parameter count: {peak_params:,}")
    
    # Sanity check
    passed, notes = sanity_check_task0(final_task_accs[0], threshold=0.90)
    if not passed:
        print(f"\n[WARNING] Sanity check failed: {notes}")
    
    # Print expansion events
    if len(expansion_events) > 0:
        print(f"\n{'='*60}")
        print(f"Expansion Events: {len(expansion_events)}")
        total_params_added = 0
        for event in expansion_events:
            print(f"  Task {event['task_id']}: Added module {event['new_module_id']} "
                  f"(confidence={event['confidence']:.4f}, params_added={event['params_added']:,})")
            total_params_added += event['params_added']
        print(f"  Total parameters added by expansion: {total_params_added:,}")
        print(f"{'='*60}")
    
    # Build result row
    result = build_csv_row(
        config=config,
        env_metadata=env_metadata,
        task_accs=final_task_accs,
        forgetting=final_forgetting,
        final_params=final_params,
        peak_params=peak_params,
        train_time_s=train_time_s,
        peak_vram_mb=peak_vram_mb,
        lit_info=None,
        notes=notes
    )
    
    # Add expansion info
    result['expansion_events'] = expansion_events
    result['task_history'] = task_history
    
    return result

