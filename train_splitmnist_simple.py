"""
Simplified training script for SplitMNIST with Modular Continual Learning.

This script trains a modular network with attention routing on SplitMNIST
using basic regularization. The goal is to reach >97% average accuracy
in task-IL setting with a simpler approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from models.modular_layer import create_mlp_modular_layer


class SimpleModularNetwork(nn.Module):
    """Simple modular network with attention routing."""
    
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


def create_splitmnist_data(num_tasks=5, samples_per_task=1000):
    """Create SplitMNIST dataset."""
    from torchvision import datasets, transforms
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Split into tasks (0-1, 2-3, 4-5, 6-7, 8-9)
    train_tasks = []
    test_tasks = []
    
    for task_id in range(num_tasks):
        class1 = task_id * 2
        class2 = task_id * 2 + 1
        
        # Training data
        train_mask = (train_dataset.targets == class1) | (train_dataset.targets == class2)
        train_data = train_dataset.data[train_mask]
        train_labels = train_dataset.targets[train_mask]
        
        # Remap labels to 0, 1 for each task
        train_labels = torch.where(train_labels == class1, 0, 1)
        
        # Sample subset for faster training
        if len(train_data) > samples_per_task:
            indices = torch.randperm(len(train_data))[:samples_per_task]
            train_data = train_data[indices]
            train_labels = train_labels[indices]
        
        train_tasks.append((train_data, train_labels))
        
        # Test data
        test_mask = (test_dataset.targets == class1) | (test_dataset.targets == class2)
        test_data = test_dataset.data[test_mask]
        test_labels = test_dataset.targets[test_mask]
        test_labels = torch.where(test_labels == class1, 0, 1)
        
        test_tasks.append((test_data, test_labels))
    
    return train_tasks, test_tasks


def train_task(model, train_data, train_labels, device, epochs=15, lr=0.001):
    """Train model on a single task."""
    model.train()
    
    # Create data loader
    dataset = TensorDataset(train_data.float() / 255.0, train_labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        total_ce_loss = 0
        total_router_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output, regularizers = model(data, return_regularizers=True)
            ce_loss = F.cross_entropy(output, target)
            
            # Router regularizers
            router_loss = sum(regularizers.values()) if regularizers else 0.0
            
            # Total loss
            total_loss_batch = ce_loss + router_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_ce_loss += ce_loss.item()
            total_router_loss += router_loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss={total_loss/len(dataloader):.4f}, "
                  f"CE={total_ce_loss/len(dataloader):.4f}, "
                  f"Router={total_router_loss/len(dataloader):.4f}")


def evaluate_task(model, test_data, test_labels, device):
    """Evaluate model on a single task."""
    model.eval()
    
    dataset = TensorDataset(test_data.float() / 255.0, test_labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    return accuracy


def main():
    """Main training loop for SplitMNIST."""
    print("Training Simple Modular Continual Learning on SplitMNIST")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data
    print("Creating SplitMNIST dataset...")
    train_tasks, test_tasks = create_splitmnist_data(num_tasks=5, samples_per_task=2000)
    print(f"Created {len(train_tasks)} tasks")
    
    # Create model
    model = SimpleModularNetwork(
        input_size=784,
        hidden_sizes=[400, 400],
        num_classes=2,  # Binary classification per task
        num_modules=4,
        use_router=True,
        router_entropy_coef=0.001,  # Low entropy coefficient
        router_drift_coef=0.01      # Low drift coefficient
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    task_accuracies = []
    start_time = time.time()
    
    for task_id in range(len(train_tasks)):
        print(f"\n--- Task {task_id + 1} ---")
        
        train_data, train_labels = train_tasks[task_id]
        test_data, test_labels = test_tasks[task_id]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        
        # Train on current task
        print("Training...")
        train_task(model, train_data, train_labels, device, epochs=15, lr=0.001)
        
        # Evaluate on all tasks seen so far
        print("Evaluating...")
        task_accs = []
        for eval_task_id in range(task_id + 1):
            eval_test_data, eval_test_labels = test_tasks[eval_task_id]
            acc = evaluate_task(model, eval_test_data, eval_test_labels, device)
            task_accs.append(acc)
            print(f"  Task {eval_task_id + 1} accuracy: {acc:.4f}")
        
        # Store average accuracy
        avg_acc = np.mean(task_accs)
        task_accuracies.append(avg_acc)
        print(f"Average accuracy so far: {avg_acc:.4f}")
    
    # Final results
    total_time = time.time() - start_time
    final_avg_acc = task_accuracies[-1]
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Final average accuracy: {final_avg_acc:.4f}")
    print(f"Task accuracies: {[f'{acc:.4f}' for acc in task_accuracies]}")
    
    if final_avg_acc > 0.97:
        print("[SUCCESS] Achieved >97% average accuracy!")
    else:
        print(f"[INFO] Did not reach 97% target (got {final_avg_acc:.4f})")
    
    return model, task_accuracies


if __name__ == "__main__":
    model, accuracies = main()
