"""
Main orchestrator for SplitMNIST ablation experiments.

This script runs a 2×2 ablation study:
- Router: {none, attn}
- Module Expansion: {off, on}

For each configuration × seed, it runs the experiment and aggregates results.
"""
import argparse
import json
import csv
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import warnings

import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import set_global_seeds, get_env_metadata, auto_device, create_run_directory
from experiments.train_splitmnist import run_one_experiment


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run SplitMNIST ablation experiments with dynamic routing and module expansion'
    )
    
    # Core ablation parameters
    parser.add_argument('--router', type=str, default='attn', choices=['none', 'attn'],
                       help='Router type: none (fixed) or attn (attention-based)')
    parser.add_argument('--module-expansion', type=str, default='off', choices=['on', 'off'],
                       help='Enable module expansion')
    parser.add_argument('--expansion-threshold', type=float, default=0.4,
                       help='Confidence threshold for triggering expansion')
    parser.add_argument('--expansion-cooldown', type=int, default=1,
                       help='Tasks to wait after expansion before next expansion')
    parser.add_argument('--max-modules-per-layer', type=int, default=8,
                       help='Maximum modules per layer (budget for expansion)')
    
    # Training parameters
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                       help='Random seeds to run')
    parser.add_argument('--epochs-per-task', type=int, default=5,
                       help='Epochs per task')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    # Protocol parameters
    parser.add_argument('--num-tasks', type=int, default=5,
                       help='Number of tasks in SplitMNIST')
    parser.add_argument('--task-incremental', action='store_true', default=True,
                       help='Use task-incremental (multi-head) protocol')
    
    # IO parameters
    parser.add_argument('--outdir', type=str, default='results_for_fyp',
                       help='Output directory')
    parser.add_argument('--lit-file', type=str, default='literature/baselines_splitmnist.yaml',
                       help='Literature baselines YAML file')
    parser.add_argument('--data-dir', type=str, default='./store/datasets',
                       help='Data directory')
    
    # Execution parameters
    parser.add_argument('--run-full-grid', action='store_true', default=False,
                       help='Run full 2×2 ablation grid (ignores --router and --module-expansion for grid generation)')
    parser.add_argument('--single-config', action='store_true', default=False,
                       help='Run only the single configuration specified by CLI args (opposite of --run-full-grid)')
    
    return parser.parse_args()


def build_config_grid(args) -> List[Dict[str, Any]]:
    """
    Build the 2×2 configuration grid.
    
    Returns:
        List of configuration dictionaries
    """
    if args.run_full_grid or (not args.single_config and not args.run_full_grid):
        # Build 2×2 grid (default behavior)
        routers = ['none', 'attn']
        expansions = ['off', 'on']
    else:
        # Single configuration from CLI args
        routers = [args.router]
        expansions = [args.module_expansion]
    
    configs = []
    
    for router in routers:
        for expansion in expansions:
            # Set initial_modules based on router type
            if router == 'attn':
                initial_modules = 2  # Enable non-trivial routing from start
            elif router == 'none':
                initial_modules = 1  # Deterministic single module baseline
            else:
                initial_modules = 1  # Fallback
                
            # Base configuration
            config_template = {
                'router': router,
                'module_expansion': expansion,
                'expansion_threshold': args.expansion_threshold,
                'expansion_cooldown': args.expansion_cooldown,
                'max_modules_per_layer': args.max_modules_per_layer,
                'num_tasks': args.num_tasks,
                'epochs_per_task': args.epochs_per_task,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'optimizer': args.optimizer,
                'task_incremental': args.task_incremental,
                'data_dir': args.data_dir,
                'initial_modules': initial_modules,
            }
            
            # Device
            if args.device == 'auto':
                config_template['device'] = auto_device()
            else:
                config_template['device'] = args.device
            
            # Check device availability
            if config_template['device'] == 'cuda':
                import torch
                if not torch.cuda.is_available():
                    warnings.warn("CUDA not available, falling back to CPU")
                    config_template['device'] = 'cpu'
            
            configs.append(config_template)
    
    return configs


def load_literature_baselines(lit_file: str) -> Dict[str, Any]:
    """Load literature baseline configurations."""
    if not os.path.exists(lit_file):
        warnings.warn(f"Literature file not found: {lit_file}")
        return {}
    
    try:
        with open(lit_file, 'r') as f:
            baselines = yaml.safe_load(f)
        return {b['paper_key']: b for b in baselines} if baselines else {}
    except Exception as e:
        warnings.warn(f"Error loading literature file: {e}")
        return {}


def match_protocol(config: Dict[str, Any], lit_baselines: Dict[str, Any]) -> Dict[str, Any]:
    """
    Match experiment protocol with literature baselines.
    
    Returns:
        Dictionary with literature comparison info
    """
    # For now, just return default "no match"
    # In a real implementation, we would compare protocols
    return {
        'paper_key': '',
        'reported_avg_acc': '',
        'protocol_match': 'no'
    }


def compute_aggregate_stats(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute aggregate statistics (mean and std) for each configuration.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        List of aggregate result dictionaries
    """
    # Group by configuration (router, module_expansion)
    config_groups = {}
    
    for result in results:
        key = (result['router'], result['module_expansion'])
        if key not in config_groups:
            config_groups[key] = []
        config_groups[key].append(result)
    
    aggregate_results = []
    
    # Compute mean and std for each group
    for (router, expansion), group_results in config_groups.items():
        if len(group_results) == 0:
            continue
        
        # Numeric fields to aggregate
        numeric_fields = [
            'task0_acc', 'task1_acc', 'task2_acc', 'task3_acc', 'task4_acc',
            'final_avg_acc', 'final_forgetting', 'final_params', 'peak_params',
            'train_time_s'
        ]
        
        # Compute means
        mean_result = group_results[0].copy()
        mean_result['seed'] = 'agg_mean'
        
        for field in numeric_fields:
            values = [r[field] for r in group_results if isinstance(r.get(field), (int, float))]
            if len(values) > 0:
                mean_result[field] = np.mean(values)
            else:
                # Handle case where field is missing from all results (failed experiments)
                mean_result[field] = 0.0
        
        aggregate_results.append(mean_result)
        
        # Compute stds
        std_result = group_results[0].copy()
        std_result['seed'] = 'agg_std'
        
        for field in numeric_fields:
            values = [r[field] for r in group_results if isinstance(r.get(field), (int, float))]
            if len(values) > 1:
                std_result[field] = np.std(values, ddof=1)
            else:
                std_result[field] = 0.0
        
        aggregate_results.append(std_result)
    
    return aggregate_results


def write_csv(results: List[Dict[str, Any]], csv_path: str):
    """Write results to CSV file."""
    if len(results) == 0:
        print("No results to write!")
        return
    
    # Define column order
    columns = [
        'timestamp', 'commit_sha', 'device',
        'seed', 'router', 'module_expansion', 'expansion_threshold',
        'expansion_cooldown', 'max_modules_per_layer',
        'num_tasks', 'epochs_per_task', 'batch_size', 'lr', 'optimizer',
        'task0_acc', 'task1_acc', 'task2_acc', 'task3_acc', 'task4_acc',
        'final_avg_acc', 'final_forgetting', 'final_params', 'peak_params',
        'train_time_s', 'peak_vram_mb',
        'lit_paper_key', 'lit_reported_avg_acc', 'protocol_match', 'notes'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for result in results:
            # Filter to only include columns in the schema
            row = {col: result.get(col, '') for col in columns}
            writer.writerow(row)
    
    print(f"\nResults written to: {os.path.abspath(csv_path)}")


def main():
    """Main execution function."""
    args = parse_args()
    
    print("="*70)
    print("SplitMNIST Experiment Harness - 2×2 Ablation Study")
    print("="*70)
    print("\nThis will run 4 configurations:")
    print("  1. Router=none, Expansion=off (baseline)")
    print("  2. Router=none, Expansion=on")
    print("  3. Router=attn, Expansion=off")
    print("  4. Router=attn, Expansion=on")
    print(f"\nConfiguration:")
    print(f"  Seeds: {args.seeds}")
    print(f"  Epochs per task: {args.epochs_per_task}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Max modules per layer: {args.max_modules_per_layer}")
    print(f"  Expansion threshold: {args.expansion_threshold}")
    print(f"  Device: {args.device}")
    print(f"  Output directory: {args.outdir}")
    print()
    
    # Get environment metadata
    env_metadata = get_env_metadata()
    
    # Create output directory
    run_dir = create_run_directory(args.outdir, env_metadata['timestamp'])
    
    print(f"Output directory: {run_dir}\n")
    
    # Save configuration
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Save environment metadata
    env_path = os.path.join(run_dir, 'env.json')
    with open(env_path, 'w') as f:
        json.dump(env_metadata, f, indent=2)
    
    # Load literature baselines
    lit_baselines = load_literature_baselines(args.lit_file)
    
    # Build configuration grid
    config_templates = build_config_grid(args)
    
    print(f"Running {len(config_templates)} configurations × {len(args.seeds)} seeds = "
          f"{len(config_templates) * len(args.seeds)} total experiments\n")
    
    # Run all experiments
    all_results = []
    
    for config_idx, config_template in enumerate(config_templates):
        print(f"\n{'='*70}")
        print(f"Configuration {config_idx+1}/{len(config_templates)}")
        print(f"  Router: {config_template['router']}")
        print(f"  Module Expansion: {config_template['module_expansion']}")
        print(f"{'='*70}")
        
        for seed_idx, seed in enumerate(args.seeds):
            print(f"\n--- Seed {seed} ({seed_idx+1}/{len(args.seeds)}) ---")
            
            # Create config for this run
            config = config_template.copy()
            config['seed'] = seed
            
            try:
                # Run experiment
                result = run_one_experiment(config)
                
                # Add literature comparison
                lit_info = match_protocol(config, lit_baselines)
                result.update({
                    'lit_paper_key': lit_info['paper_key'],
                    'lit_reported_avg_acc': lit_info['reported_avg_acc'],
                    'protocol_match': lit_info['protocol_match']
                })
                
                # Save per-run JSON
                config_name = f"router-{config['router']}_exp-{config['module_expansion']}"
                json_path = os.path.join(
                    run_dir, 'per_seed_json',
                    f"{config_name}_seed{seed}.json"
                )
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                all_results.append(result)
                
                # Print summary
                print(f"\nResults for {config_name}, seed {seed}:")
                print(f"  Final Avg Acc: {result['final_avg_acc']:.4f}")
                print(f"  Forgetting: {result['final_forgetting']:.4f}")
                print(f"  Parameters: {result['final_params']:,}")
                
            except Exception as e:
                print(f"ERROR running experiment: {e}")
                import traceback
                traceback.print_exc()
                
                # Create error result
                error_result = {
                    'seed': seed,
                    'router': config['router'],
                    'module_expansion': config['module_expansion'],
                    'notes': f"ERROR: {str(e)}",
                    'final_avg_acc': 0.0,
                    'final_forgetting': 0.0,
                }
                all_results.append(error_result)
    
    # Compute aggregate statistics
    print(f"\n{'='*70}")
    print("Computing aggregate statistics...")
    print(f"{'='*70}")
    
    aggregate_results = compute_aggregate_stats(all_results)
    
    # Combine individual runs and aggregates
    final_results = all_results + aggregate_results
    
    # Write CSV
    csv_path = os.path.join(run_dir, 'results.csv')
    write_csv(final_results, csv_path)
    
    # Write summary log
    log_path = os.path.join(run_dir, 'logs', 'run.log')
    with open(log_path, 'w') as f:
        f.write(f"SplitMNIST Experiment Run\n")
        f.write(f"{'='*70}\n")
        f.write(f"Timestamp: {env_metadata['timestamp']}\n")
        f.write(f"Commit SHA: {env_metadata.get('commit_sha', 'N/A')}\n")
        f.write(f"Git Dirty: {env_metadata.get('git_dirty', 'N/A')}\n")
        f.write(f"Device: {env_metadata['device_name']}\n")
        f.write(f"\nTotal experiments: {len(all_results)}\n")
        f.write(f"Configurations: {len(config_templates)}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"\nResults saved to: {csv_path}\n")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal runs: {len(all_results)}")
    print(f"Results CSV: {os.path.abspath(csv_path)}")
    print(f"Output directory: {os.path.abspath(run_dir)}")
    print(f"\nAggregate Results:")
    print(f"{'-'*70}")
    
    for result in aggregate_results:
        if result['seed'] == 'agg_mean':
            print(f"\n{result['router']:6s} + expansion={result['module_expansion']:3s}:")
            print(f"  Avg Acc:   {result['final_avg_acc']:.4f}")
            print(f"  Forgetting: {result['final_forgetting']:.4f}")
            print(f"  Params:     {result.get('final_params', 0):,.0f}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()

