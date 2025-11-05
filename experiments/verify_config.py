"""Quick verification script to show what experiments will be run."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import parse_args, build_config_grid

# Simulate default arguments
class Args:
    router = 'attn'
    module_expansion = 'off'
    expansion_threshold = 0.4
    expansion_cooldown = 0
    max_modules_per_layer = 8
    seeds = [0, 1, 2]
    epochs_per_task = 5
    batch_size = 64
    lr = 3e-4
    optimizer = 'adam'
    device = 'auto'
    num_tasks = 5
    task_incremental = True
    outdir = 'results_for_fyp'
    lit_file = 'literature/baselines_splitmnist.yaml'
    data_dir = './store/datasets'
    run_full_grid = False
    single_config = False

args = Args()
configs = build_config_grid(args)

print('='*70)
print('SplitMNIST Experiment Configuration Verification')
print('='*70)
print(f'\nNumber of configurations: {len(configs)}')
print(f'\nConfigurations to be run:')
for i, cfg in enumerate(configs, 1):
    print(f'  {i}. Router={cfg["router"]}, Expansion={cfg["module_expansion"]}, ' +
          f'Initial Modules={cfg["initial_modules"]}, Max Modules={cfg["max_modules_per_layer"]}')

print(f'\nSeeds: {args.seeds}')
print(f'Total experiments: {len(configs)} configs × {len(args.seeds)} seeds = {len(configs) * len(args.seeds)} experiments')
print(f'\nEpochs per task: {args.epochs_per_task}')
print(f'Expansion threshold: {args.expansion_threshold}')
print(f'Expansion cooldown: {args.expansion_cooldown}')
print('='*70)

