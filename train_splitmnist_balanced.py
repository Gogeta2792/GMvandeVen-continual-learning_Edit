"""
Balanced Training Schedule for Modular Continual Learning

This script implements a comprehensive training loop that alternates emphasis between
router and modules to prevent routing collapse. It includes:

1. Balanced loss computation: loss_cls + loss_router + optional sparsity_loss
2. Two-optimizer system: opt_modules (higher LR) + opt_router (lower LR)
3. Curriculum learning: freeze router for first N steps to let modules learn
4. Early stopping: detect router collapse (all weight on one module for K steps)
5. CLI flags for all coefficients and learning rates
6. Integration with existing --experiment=splitMNIST --scenario=task pipeline

Based on the documentation from:
- ATTENTION_ROUTER_README.md
- ATTENTION_WEIGHTED_SUM_README.md  
- MODULAR_LAYER_README.md
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Import existing framework components
from data.load import get_context_set
from models.modular_layer import create_mlp_modular_layer
from models.cl.continual_learner import ContinualLearner


class BalancedModularNetwork(nn.Module):
    """Modular network with balanced training for router and modules."""
    
    def __init__(self, input_size, hidden_sizes, num_classes, num_modules=4, 
                 use_router=True, router_hidden_dim=128, router_entropy_coef=0.01, 
                 router_drift_coef=0.1, sparsity_coef=0.001, device='cpu'):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_modules = num_modules
        self.use_router = use_router
        self.sparsity_coef = sparsity_coef
        self.device = device
        
        # Create modular layers
        self.modular_layers = nn.ModuleList()
        prev_dim = input_size
        
        for hidden_size in hidden_sizes:
            layer = create_mlp_modular_layer(
                in_dim=prev_dim,
                out_dim=hidden_size,
                num_modules=num_modules,
                use_router=use_router,
                router_hidden_dim=router_hidden_dim,
                router_entropy_coef=router_entropy_coef,
                router_drift_coef=router_drift_coef,
                device=device
            )
            self.modular_layers.append(layer)
            prev_dim = hidden_size
        
        # Final classifier
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Router collapse detection
        self.router_collapse_threshold = 0.95  # If any module gets >95% attention
        self.collapse_steps = 0
        self.max_collapse_steps = 100  # Stop if collapsed for 100 steps
        
        # Training state
        self.router_frozen = False
        self.freeze_router_steps = 0
        
    def forward(self, x, return_regularizers=False, return_attention=False):
        """Forward pass with optional attention and regularizer returns."""
        x = x.view(x.size(0), -1)  # Flatten input
        
        all_regularizers = {}
        all_attention = {}
        
        for i, layer in enumerate(self.modular_layers):
            if self.use_router:
                x, attn_info = layer(x, return_attn=True)
                if attn_info is not None:
                    all_regularizers.update({f'layer_{i}_{k}': v for k, v in attn_info['regularizers'].items()})
                    all_attention[f'layer_{i}'] = attn_info['attention']
            else:
                x = layer(x)[0]  # No attention routing
        
        output = self.classifier(x)
        
        if return_regularizers and return_attention:
            return output, all_regularizers, all_attention
        elif return_regularizers:
            return output, all_regularizers
        elif return_attention:
            return output, all_attention
        else:
            return output
    
    def compute_sparsity_loss(self, attention_dict):
        """Compute L1 sparsity loss on mean attention to keep few modules active."""
        if not attention_dict or self.sparsity_coef <= 0:
            return 0.0
        
        sparsity_loss = 0.0
        for layer_name, attention in attention_dict.items():
            # Mean attention across batch: [num_modules]
            mean_attention = attention.mean(dim=0)
            # L1 penalty on mean attention
            sparsity_loss += torch.sum(torch.abs(mean_attention))
        
        return self.sparsity_coef * sparsity_loss
    
    def check_router_collapse(self, attention_dict):
        """Check if router has collapsed to one module."""
        if not attention_dict:
            return False
        
        for layer_name, attention in attention_dict.items():
            # Check if any module gets >95% attention
            max_attention = attention.max(dim=1)[0]  # Max attention per sample
            if (max_attention > self.router_collapse_threshold).all():
                return True
        
        return False
    
    def freeze_router(self, freeze=True):
        """Freeze or unfreeze router parameters."""
        self.router_frozen = freeze
        for layer in self.modular_layers:
            if hasattr(layer, 'router') and layer.router is not None:
                for param in layer.router.parameters():
                    param.requires_grad = not freeze
    
    def get_router_parameters(self):
        """Get all router parameters for separate optimizer."""
        router_params = []
        for layer in self.modular_layers:
            if hasattr(layer, 'router') and layer.router is not None:
                router_params.extend(layer.router.parameters())
        return router_params
    
    def get_module_parameters(self):
        """Get all module parameters (excluding router)."""
        module_params = []
        for layer in self.modular_layers:
            # Add module parameters
            for module in layer.modules_list:
                module_params.extend(module.parameters())
            # Add classifier parameters
            if layer == self.modular_layers[-1]:  # Last layer
                module_params.extend(self.classifier.parameters())
        return module_params


class BalancedTrainer:
    """Trainer with balanced emphasis on router and modules."""
    
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        
        # Two-optimizer system
        self.opt_modules = optim.Adam(
            model.get_module_parameters(), 
            lr=args.lr_modules, 
            betas=(0.9, 0.999)
        )
        
        if args.use_router:
            self.opt_router = optim.Adam(
                model.get_router_parameters(), 
                lr=args.lr_router, 
                betas=(0.9, 0.999)
            )
        else:
            self.opt_router = None
        
        # Training state
        self.step = 0
        self.task_id = 0
        self.router_collapsed = False
        
        # TensorBoard logging
        if args.log_dir and SummaryWriter is not None:
            self.writer = SummaryWriter(args.log_dir)
        else:
            self.writer = None
    
    def train_step(self, data, target, task_id):
        """Single training step with balanced loss computation."""
        self.model.train()
        self.step += 1
        
        # Forward pass
        if self.args.use_router:
            output, regularizers, attention = self.model(
                data, return_regularizers=True, return_attention=True
            )
        else:
            output, regularizers = self.model(data, return_regularizers=True)
            attention = {}
        
        # Classification loss
        loss_cls = F.cross_entropy(output, target)
        
        # Router loss: entropy_coef*H(attn) + drift_coef*||attn-ema||^2
        loss_router = 0.0
        if regularizers:
            loss_router = sum(regularizers.values())
        
        # Optional sparsity loss: L1 on mean attn to keep few modules active
        loss_sparsity = self.model.compute_sparsity_loss(attention)
        
        # Total loss
        total_loss = loss_cls + loss_router + loss_sparsity
        
        # Backward pass with two-optimizer system
        self.opt_modules.zero_grad()
        if self.opt_router is not None:
            self.opt_router.zero_grad()
        
        total_loss.backward()
        
        # Update modules (always)
        self.opt_modules.step()
        
        # Update router (only if not frozen and not collapsed)
        if (self.opt_router is not None and 
            not self.model.router_frozen and 
            not self.router_collapsed):
            self.opt_router.step()
        
        # Check for router collapse
        if self.args.use_router and self.model.check_router_collapse(attention):
            self.model.collapse_steps += 1
            if self.model.collapse_steps >= self.model.max_collapse_steps:
                self.router_collapsed = True
                print(f"WARNING: Router collapsed at step {self.step}! Stopping router updates.")
        else:
            self.model.collapse_steps = 0
        
        # Logging
        if self.writer and self.step % 100 == 0:
            self.writer.add_scalar('Loss/Classification', loss_cls.item(), self.step)
            self.writer.add_scalar('Loss/Router', loss_router.item(), self.step)
            self.writer.add_scalar('Loss/Sparsity', loss_sparsity.item(), self.step)
            self.writer.add_scalar('Loss/Total', total_loss.item(), self.step)
            
            # Log attention statistics
            for layer_name, attn in attention.items():
                self.writer.add_histogram(f'Attention/{layer_name}', attn, self.step)
                self.writer.add_scalar(f'Attention/{layer_name}_entropy', 
                                     -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean(), 
                                     self.step)
        
        return {
            'loss_cls': loss_cls.item(),
            'loss_router': loss_router.item(),
            'loss_sparsity': loss_sparsity.item(),
            'total_loss': total_loss.item(),
            'router_collapsed': self.router_collapsed,
            'router_frozen': self.model.router_frozen
        }
    
    def train_task(self, train_loader, task_id, epochs=20):
        """Train on a single task with curriculum learning."""
        print(f"\n--- Training Task {task_id + 1} ---")
        
        # Curriculum: freeze router for first N steps to let modules learn
        if self.args.freeze_router_steps > 0 and task_id == 0:
            print(f"Freezing router for first {self.args.freeze_router_steps} steps...")
            self.model.freeze_router(freeze=True)
        
        total_steps = len(train_loader) * epochs
        step_count = 0
        
        for epoch in range(epochs):
            epoch_losses = {'loss_cls': 0, 'loss_router': 0, 'loss_sparsity': 0, 'total_loss': 0}
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Unfreeze router after curriculum period
                if (self.args.freeze_router_steps > 0 and 
                    step_count >= self.args.freeze_router_steps and 
                    self.model.router_frozen):
                    print(f"Unfreezing router at step {step_count}")
                    self.model.freeze_router(freeze=False)
                
                # Training step
                step_stats = self.train_step(data, target, task_id)
                
                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += step_stats[key]
                
                step_count += 1
                
                # Early stopping if router collapsed
                if self.router_collapsed:
                    print(f"Early stopping due to router collapse at step {step_count}")
                    break
            
            # Print epoch statistics
            if epoch % 5 == 0:
                avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
                print(f"  Epoch {epoch}: Loss={avg_losses['total_loss']:.4f}, "
                      f"CE={avg_losses['loss_cls']:.4f}, "
                      f"Router={avg_losses['loss_router']:.4f}, "
                      f"Sparsity={avg_losses['loss_sparsity']:.4f}")
        
        # Reset router collapse detection for next task
        self.model.collapse_steps = 0
        self.router_collapsed = False


def create_balanced_args():
    """Create arguments for balanced training."""
    parser = argparse.ArgumentParser(description='Balanced Modular Continual Learning Training')
    
    # General options (compatible with existing framework)
    parser.add_argument('--experiment', type=str, default='splitMNIST', help='experiment name')
    parser.add_argument('--scenario', type=str, default='task', help='scenario')
    parser.add_argument('--contexts', type=int, default=5, help='number of contexts')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', action='store_true', help='use GPU')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    
    # Data options
    parser.add_argument('--d_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--normalize', action='store_true', help='normalize data')
    
    # Model options
    parser.add_argument('--fc_layers', type=int, default=2, help='number of fc layers')
    parser.add_argument('--fc_units', type=int, default=400, help='number of fc units')
    parser.add_argument('--num_modules', type=int, default=4, help='number of modules per layer')
    parser.add_argument('--use_router', action='store_true', help='use attention router')
    
    # Training options
    parser.add_argument('--iters', type=int, default=2000, help='iterations per context')
    parser.add_argument('--epochs', type=int, default=20, help='epochs per task')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    
    # Two-optimizer learning rates
    parser.add_argument('--lr_modules', type=float, default=0.001, help='learning rate for modules')
    parser.add_argument('--lr_router', type=float, default=0.0001, help='learning rate for router')
    
    # Loss coefficients
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy regularization coefficient')
    parser.add_argument('--drift_coef', type=float, default=0.1, help='drift regularization coefficient')
    parser.add_argument('--sparsity_coef', type=float, default=0.001, help='sparsity regularization coefficient')
    
    # Curriculum learning
    parser.add_argument('--freeze_router_steps', type=int, default=500, help='steps to freeze router initially')
    
    # Router collapse detection
    parser.add_argument('--collapse_threshold', type=float, default=0.95, help='router collapse threshold')
    parser.add_argument('--max_collapse_steps', type=int, default=100, help='max steps before stopping on collapse')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--save_dir', type=str, default='./store/models', help='model save directory')
    
    return parser.parse_args()


def evaluate_model(model, test_loaders, device, task_id):
    """Evaluate model on all tasks seen so far."""
    model.eval()
    task_accuracies = []
    
    with torch.no_grad():
        for eval_task_id in range(task_id + 1):
            correct = 0
            total = 0
            
            for data, target in test_loaders[eval_task_id]:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
            
            accuracy = correct / total
            task_accuracies.append(accuracy)
            print(f"  Task {eval_task_id + 1} accuracy: {accuracy:.4f}")
    
    return task_accuracies


def main():
    """Main training function with balanced training schedule."""
    print("Balanced Modular Continual Learning Training")
    print("=" * 60)
    
    # Parse arguments
    args = create_balanced_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
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
    print("Creating balanced modular network...")
    model = BalancedModularNetwork(
        input_size=config['size'],
        hidden_sizes=[args.fc_units] * args.fc_layers,
        num_classes=config['classes_per_context'],
        num_modules=args.num_modules,
        use_router=args.use_router,
        router_entropy_coef=args.entropy_coef,
        router_drift_coef=args.drift_coef,
        sparsity_coef=args.sparsity_coef,
        device=device
    ).to(device)
    
    # Set router collapse parameters
    model.router_collapse_threshold = args.collapse_threshold
    model.max_collapse_steps = args.max_collapse_steps
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Router parameters: {sum(p.numel() for p in model.get_router_parameters()):,}")
    print(f"Module parameters: {sum(p.numel() for p in model.get_module_parameters()):,}")
    
    # Create trainer
    trainer = BalancedTrainer(model, device, args)
    
    # Create data loaders
    train_loaders = []
    test_loaders = []
    
    for i in range(len(train_datasets)):
        train_loader = DataLoader(train_datasets[i], batch_size=args.batch, shuffle=True)
        test_loader = DataLoader(test_datasets[i], batch_size=args.batch, shuffle=False)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    # Training loop
    print("\nStarting balanced training...")
    all_task_accuracies = []
    start_time = time.time()
    
    for task_id in range(len(train_datasets)):
        print(f"\n{'='*60}")
        print(f"TASK {task_id + 1}/{len(train_datasets)}")
        print(f"{'='*60}")
        
        # Train on current task
        trainer.train_task(train_loaders[task_id], task_id, epochs=args.epochs)
        
        # Evaluate on all tasks seen so far
        print("Evaluating...")
        task_accuracies = evaluate_model(model, test_loaders, device, task_id)
        all_task_accuracies.append(task_accuracies)
        
        avg_acc = np.mean(task_accuracies)
        print(f"Average accuracy so far: {avg_acc:.4f}")
        
        # Log to TensorBoard
        if trainer.writer:
            trainer.writer.add_scalar('Accuracy/Average', avg_acc, task_id)
            for i, acc in enumerate(task_accuracies):
                trainer.writer.add_scalar(f'Accuracy/Task_{i+1}', acc, task_id)
    
    # Final results
    total_time = time.time() - start_time
    final_avg_acc = all_task_accuracies[-1]
    
    print("\n" + "=" * 60)
    print("BALANCED TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Final average accuracy: {final_avg_acc:.4f}")
    print(f"Task accuracies: {[f'{acc:.4f}' for acc in all_task_accuracies[-1]]}")
    
    # Success criteria
    if final_avg_acc > 0.97:
        print("✅ SUCCESS: Achieved >97% accuracy!")
    else:
        print("❌ Did not reach 97% accuracy target")
    
    # Close TensorBoard writer
    if trainer.writer:
        trainer.writer.close()
    
    return all_task_accuracies


if __name__ == "__main__":
    main()
