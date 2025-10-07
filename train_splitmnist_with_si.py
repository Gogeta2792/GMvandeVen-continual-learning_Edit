"""
Training script for SplitMNIST with Modular Continual Learning using existing framework.

This script integrates the modular layer with the existing continual learning
framework to properly implement Synaptic Intelligence (SI) and reach >97% accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.modular_layer import create_mlp_modular_layer
from models.cl.continual_learner import ContinualLearner
from data.load import get_context_set
from params import options
from params.param_values import set_method_options, set_default_values, check_for_errors


class ModularContinualLearner(ContinualLearner):
    """Continual learner with modular layers and attention routing."""
    
    def __init__(self, input_size=784, hidden_sizes=[400, 400], num_classes=2, 
                 num_modules=4, use_router=True, router_entropy_coef=0.001, 
                 router_drift_coef=0.01):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.num_modules = num_modules
        self.use_router = use_router
        
        # Create modular layers
        self.modular_layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layer = create_mlp_modular_layer(
                in_dim=prev_size,
                out_dim=hidden_size,
                num_modules=num_modules,
                use_router=use_router,
                router_hidden_dim=128,
                router_entropy_coef=router_entropy_coef,
                router_drift_coef=router_drift_coef
            )
            self.modular_layers.append(layer)
            prev_size = hidden_size
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Set label for compatibility
        self.label = "ModularClassifier"
        self.name = f"ModularClassifier-{num_modules}modules"
        
    def forward(self, x, return_regularizers=False):
        """Forward pass through modular layers with attention routing."""
        # Flatten input
        x = x.view(x.size(0), -1)
        
        total_regularizers = {}
        
        # Pass through modular layers
        for i, layer in enumerate(self.modular_layers):
            x, attn_info = layer(x, return_attn=True)
            
            # Collect router regularizers
            if attn_info and 'regularizers' in attn_info:
                for key, value in attn_info['regularizers'].items():
                    reg_key = f"layer_{i}_{key}"
                    if reg_key in total_regularizers:
                        total_regularizers[reg_key] += value
                    else:
                        total_regularizers[reg_key] = value
        
        # Final classification
        output = self.classifier(x)
        
        if return_regularizers:
            return output, total_regularizers
        else:
            return output
    
    def list_init_layers(self):
        """Return list of modules for initialization."""
        init_layers = []
        for layer in self.modular_layers:
            init_layers.extend(layer.list_init_layers())
        init_layers.append(self.classifier)
        return init_layers


def create_splitmnist_args():
    """Create arguments for SplitMNIST experiment."""
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # General options
    parser.add_argument('--experiment', type=str, default='splitMNIST', help='experiment name')
    parser.add_argument('--scenario', type=str, default='task', help='scenario')
    parser.add_argument('--contexts', type=int, default=5, help='number of contexts')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', action='store_true', help='use GPU')
    
    # Data options
    parser.add_argument('--d_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--normalize', action='store_true', help='normalize data')
    
    # Model options
    parser.add_argument('--fc_layers', type=int, default=2, help='number of fc layers')
    parser.add_argument('--fc_units', type=int, default=400, help='number of fc units')
    parser.add_argument('--fc_drop', type=float, default=0.0, help='fc dropout')
    parser.add_argument('--fc_bn', action='store_true', help='fc batch norm')
    parser.add_argument('--fc_nl', type=str, default='relu', help='fc non-linearity')
    
    # Training options
    parser.add_argument('--iters', type=int, default=2000, help='iterations per context')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    
    # SI options
    parser.add_argument('--si', action='store_true', help='use Synaptic Intelligence')
    parser.add_argument('--epsilon', type=float, default=0.1, help='SI epsilon')
    parser.add_argument('--reg_strength', type=float, default=1.0, help='SI regularization strength')
    
    # Modular options
    parser.add_argument('--num_modules', type=int, default=4, help='number of modules per layer')
    parser.add_argument('--use_router', action='store_true', help='use attention router')
    parser.add_argument('--router_entropy_coef', type=float, default=0.001, help='router entropy coefficient')
    parser.add_argument('--router_drift_coef', type=float, default=0.01, help='router drift coefficient')
    
    # Other options
    parser.add_argument('--train', action='store_true', help='train model')
    parser.add_argument('--save', action='store_true', help='save model')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    
    args = parser.parse_args([])  # Empty list for default args
    
    # Set SI options
    args.si = True
    args.epsilon = 0.1
    args.reg_strength = 1.0
    args.importance_weighting = 'si'
    args.weight_penalty = True
    
    # Set modular options
    args.use_router = True
    args.num_modules = 4
    args.router_entropy_coef = 0.001
    args.router_drift_coef = 0.01
    
    return args


def main():
    """Main training function using existing framework."""
    print("Training Modular Continual Learning on SplitMNIST with SI")
    print("=" * 60)
    
    # Create arguments
    args = create_splitmnist_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Load data using existing framework
    print("Loading SplitMNIST data...")
    (train_datasets, test_datasets), config = get_context_set(
        name=args.experiment, 
        scenario=args.scenario, 
        contexts=args.contexts, 
        data_dir=args.d_dir,
        normalize=args.normalize, 
        verbose=args.verbose
    )
    
    print(f"Loaded {len(train_datasets)} training contexts and {len(test_datasets)} test contexts")
    
    # Create model
    print("Creating modular continual learner...")
    model = ModularContinualLearner(
        input_size=config['size'],
        hidden_sizes=[args.fc_units] * args.fc_layers,
        num_classes=config['classes_per_context'],
        num_modules=args.num_modules,
        use_router=args.use_router,
        router_entropy_coef=args.router_entropy_coef,
        router_drift_coef=args.router_drift_coef
    ).to(device)
    
    # Set SI parameters
    if args.si:
        model.importance_weighting = 'si'
        model.epsilon = args.epsilon
        model.reg_strength = args.reg_strength
        model.weight_penalty = True
    
    # Set optimizer
    model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    if model.optim_type == 'adam':
        model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    task_accuracies = []
    
    for context_id in range(args.contexts):
        print(f"\n--- Context {context_id + 1} ---")
        
        # Get current context data
        train_dataset = train_datasets[context_id]
        test_dataset = test_datasets[context_id]
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Train on current context
        model.train()
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        
        for iteration in range(args.iters):
            for batch_idx, (data, target) in enumerate(train_loader):
                if iteration * len(train_loader) + batch_idx >= args.iters:
                    break
                    
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output, regularizers = model(data, return_regularizers=True)
                ce_loss = F.cross_entropy(output, target)
                
                # Router regularizers
                router_loss = sum(regularizers.values()) if regularizers else 0.0
                
                # SI regularization (handled by the framework)
                si_loss = model.reg_strength * model.reg_loss() if hasattr(model, 'reg_loss') else 0.0
                
                # Total loss
                total_loss = ce_loss + router_loss + si_loss
                
                # Backward pass
                model.optimizer.zero_grad()
                total_loss.backward()
                model.optimizer.step()
                
                # Update SI importance weights
                if args.si and hasattr(model, 'update_importance_weights'):
                    model.update_importance_weights()
            
            if iteration % 500 == 0:
                print(f"  Iteration {iteration}: Loss={total_loss.item():.4f}")
        
        # Update SI old parameters
        if args.si and hasattr(model, 'update_old_params'):
            model.update_old_params()
        
        # Evaluate on all contexts seen so far
        print("Evaluating...")
        model.eval()
        context_accs = []
        
        for eval_context_id in range(context_id + 1):
            eval_test_dataset = test_datasets[eval_context_id]
            test_loader = DataLoader(eval_test_dataset, batch_size=args.batch, shuffle=False)
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            
            acc = correct / total
            context_accs.append(acc)
            print(f"  Context {eval_context_id + 1} accuracy: {acc:.4f}")
        
        # Store average accuracy
        avg_acc = np.mean(context_accs)
        task_accuracies.append(avg_acc)
        print(f"Average accuracy so far: {avg_acc:.4f}")
    
    # Final results
    total_time = time.time() - start_time
    final_avg_acc = task_accuracies[-1]
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Final average accuracy: {final_avg_acc:.4f}")
    print(f"Context accuracies: {[f'{acc:.4f}' for acc in task_accuracies]}")
    
    if final_avg_acc > 0.97:
        print("[SUCCESS] Achieved >97% average accuracy!")
    else:
        print(f"[INFO] Did not reach 97% target (got {final_avg_acc:.4f})")
    
    return model, task_accuracies


if __name__ == "__main__":
    model, accuracies = main()
