"""
Metrics computation for continual learning experiments.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import time


def compute_task_accs(model, test_datasets, device: str, num_tasks: int) -> List[float]:
    """
    Compute per-task accuracy.
    
    Args:
        model: The trained model
        test_datasets: List of test datasets (one per task)
        device: Device to run evaluation on
        num_tasks: Number of tasks
    
    Returns:
        List of accuracies for each task
    """
    from torch.utils.data import DataLoader
    
    model.eval()
    task_accs = []
    
    for task_id in range(num_tasks):
        test_loader = DataLoader(test_datasets[task_id], batch_size=128, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # Get predictions for this task's classes
                # In task-incremental (multi-head), we use the task-specific head
                if hasattr(model, 'task_heads'):
                    # Multi-head model
                    logits = model.task_heads[task_id](model.feature_extractor(x))
                else:
                    # Single model - extract task-specific logits
                    logits = model(x)
                    # For SplitMNIST, each task has 2 classes
                    task_start = task_id * 2
                    task_end = task_start + 2
                    logits = logits[:, task_start:task_end]
                
                # Predictions are relative to task's classes (0 or 1 for 2-class tasks)
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        task_accs.append(accuracy)
    
    return task_accs


def compute_forgetting(task_history: List[List[float]], num_tasks: int) -> float:
    """
    Compute forgetting metric (Chaudhry et al.).
    
    For each task t, forgetting is: best_accuracy_so_far - final_accuracy
    Average forgetting is the mean over all tasks (excluding the last one).
    
    Args:
        task_history: List of accuracy lists, where task_history[i] contains
                     accuracies on all seen tasks after training task i.
                     Shape: [num_training_steps][num_tasks_seen]
        num_tasks: Total number of tasks
    
    Returns:
        Average forgetting metric
    """
    if len(task_history) < 2:
        return 0.0
    
    # For each task, track the maximum accuracy achieved
    max_accs = [0.0] * num_tasks
    
    # Build max accuracy per task across all training steps
    for step_idx, accs in enumerate(task_history):
        for task_idx in range(len(accs)):
            max_accs[task_idx] = max(max_accs[task_idx], accs[task_idx])
    
    # Final accuracies (after training all tasks)
    final_accs = task_history[-1]
    
    # Compute forgetting for each task (except the last one, as it was just learned)
    forgetting_per_task = []
    for task_idx in range(len(final_accs) - 1):
        forget = max_accs[task_idx] - final_accs[task_idx]
        forgetting_per_task.append(forget)
    
    # Average forgetting
    if len(forgetting_per_task) == 0:
        return 0.0
    
    return float(np.mean(forgetting_per_task))


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def try_peak_vram() -> Optional[float]:
    """
    Try to get peak VRAM usage in MB.
    
    Returns:
        Peak VRAM in MB, or None if CUDA not available
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        return float(peak_memory)
    except Exception:
        return None


def sanity_check_task0(task0_acc: float, threshold: float = 0.90) -> tuple[bool, str]:
    """
    Check if task 0 accuracy meets minimum threshold.
    
    Args:
        task0_acc: Accuracy on task 0
        threshold: Minimum acceptable accuracy
    
    Returns:
        Tuple of (passed, notes)
    """
    if task0_acc >= threshold:
        return True, ""
    else:
        return False, f"sanity_low_acc_task0_{task0_acc:.3f}"


def build_csv_row(
    config: Dict[str, Any],
    env_metadata: Dict[str, Any],
    task_accs: List[float],
    forgetting: float,
    final_params: int,
    peak_params: int,
    train_time_s: float,
    peak_vram_mb: Optional[float],
    lit_info: Optional[Dict[str, Any]] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Build a single CSV row from experiment results.
    
    Args:
        config: Experiment configuration
        env_metadata: Environment metadata
        task_accs: List of per-task accuracies
        forgetting: Average forgetting metric
        final_params: Final parameter count
        peak_params: Peak parameter count
        train_time_s: Training time in seconds
        peak_vram_mb: Peak VRAM usage in MB
        lit_info: Optional literature comparison info
        notes: Additional notes
    
    Returns:
        Dictionary representing one CSV row
    """
    # Compute final average accuracy
    final_avg_acc = float(np.mean(task_accs)) if len(task_accs) > 0 else 0.0
    
    row = {
        # Environment
        'timestamp': env_metadata['timestamp'],
        'commit_sha': env_metadata.get('commit_sha', ''),
        'device': env_metadata['device_name'],
        
        # Config
        'seed': config['seed'],
        'router': config['router'],
        'module_expansion': config['module_expansion'],
        'expansion_threshold': config.get('expansion_threshold', ''),
        'expansion_cooldown': config.get('expansion_cooldown', ''),
        'max_modules_per_layer': config.get('max_modules_per_layer', ''),
        
        # Training config
        'num_tasks': config['num_tasks'],
        'epochs_per_task': config['epochs_per_task'],
        'batch_size': config['batch_size'],
        'lr': config['lr'],
        'optimizer': config['optimizer'],
        
        # Results - per-task accuracies
        'task0_acc': task_accs[0] if len(task_accs) > 0 else 0.0,
        'task1_acc': task_accs[1] if len(task_accs) > 1 else 0.0,
        'task2_acc': task_accs[2] if len(task_accs) > 2 else 0.0,
        'task3_acc': task_accs[3] if len(task_accs) > 3 else 0.0,
        'task4_acc': task_accs[4] if len(task_accs) > 4 else 0.0,
        
        # Aggregate metrics
        'final_avg_acc': final_avg_acc,
        'final_forgetting': forgetting,
        'final_params': final_params,
        'peak_params': peak_params,
        'train_time_s': train_time_s,
        'peak_vram_mb': peak_vram_mb if peak_vram_mb is not None else '',
        
        # Literature comparison
        'lit_paper_key': lit_info.get('paper_key', '') if lit_info else '',
        'lit_reported_avg_acc': lit_info.get('reported_avg_acc', '') if lit_info else '',
        'protocol_match': lit_info.get('protocol_match', 'no') if lit_info else 'no',
        
        # Notes
        'notes': notes,
    }
    
    return row

