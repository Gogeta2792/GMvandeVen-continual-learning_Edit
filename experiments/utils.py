"""
Utility functions for reproducibility and environment tracking.
"""
import random
import numpy as np
import torch
import subprocess
import os
from datetime import datetime
from typing import Dict, Any


def set_global_seeds(seed: int):
    """
    Set global random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Enable deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For PyTorch >= 1.8
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some operations don't support deterministic mode
            pass


def get_env_metadata() -> Dict[str, Any]:
    """
    Collect environment metadata for reproducibility tracking.
    
    Returns:
        Dictionary with environment information
    """
    metadata = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_name': None,
        'commit_sha': None,
        'git_dirty': False,
    }
    
    # Get device name
    if torch.cuda.is_available():
        metadata['device_name'] = torch.cuda.get_device_name(0)
    else:
        metadata['device_name'] = 'cpu'
    
    # Get git information if available
    try:
        # Get commit SHA
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            metadata['commit_sha'] = result.stdout.strip()
        
        # Check if working directory is dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            metadata['git_dirty'] = len(result.stdout.strip()) > 0
    except Exception:
        # Git not available or not a git repo
        pass
    
    return metadata


def auto_device() -> str:
    """
    Automatically select the best available device.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def create_run_directory(base_dir: str, timestamp: str = None) -> str:
    """
    Create a timestamped directory for experiment outputs.
    
    Args:
        base_dir: Base directory for results
        timestamp: Optional timestamp string (if None, generated automatically)
    
    Returns:
        Path to created directory
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    run_dir = os.path.join(base_dir, f'splitmnist_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'per_seed_json'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    
    return run_dir

