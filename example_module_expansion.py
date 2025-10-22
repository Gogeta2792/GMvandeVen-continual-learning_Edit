"""
Example script demonstrating module expansion functionality.

This script shows how to use the expand-on-demand feature in ModularLayer
to dynamically add modules when the router has low confidence, similar to
the Lifelong Model Compression (LMC) approach.

The example trains a simple network on SplitMNIST with module expansion enabled,
showing how modules are added as the model encounters new task distributions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from models.modular_layer import ModularLayer, ModularLayerConfig


class ExpandableModularNetwork(nn.Module):
    """Simple network with expandable modular layers."""
    
    def __init__(self, input_dim=784, hidden_dim=400, output_dim=10,
                 num_initial_modules=2, max_modules=8,
                 confidence_threshold=0.4, enable_expansion=True):
        super().__init__()
        
        # First modular layer
        self.layer1_config = ModularLayerConfig(
            in_dim=input_dim,
            out_dim=hidden_dim,
            num_modules=num_initial_modules,
            block_type='mlp',
            hidden_dim=hidden_dim,
            use_router=True,
            enable_expansion=enable_expansion,
            max_modules=max_modules,
            confidence_threshold=confidence_threshold,
            cooldown_steps=500,  # Wait 500 steps between expansions
            projection_steps=100,  # 100 steps for projection phase
            projection_lr=0.001,
            router_hidden_dim=128,
            router_entropy_coef=0.01,
            router_drift_coef=0.1,
            device='cpu'
        )
        self.layer1 = ModularLayer(self.layer1_config)
        
        # Second modular layer
        self.layer2_config = ModularLayerConfig(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_modules=num_initial_modules,
            block_type='mlp',
            hidden_dim=hidden_dim,
            use_router=True,
            enable_expansion=enable_expansion,
            max_modules=max_modules,
            confidence_threshold=confidence_threshold,
            cooldown_steps=500,
            projection_steps=100,
            projection_lr=0.001,
            router_hidden_dim=128,
            router_entropy_coef=0.01,
            router_drift_coef=0.1,
            device='cpu'
        )
        self.layer2 = ModularLayer(self.layer2_config)
        
        # Output classifier
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, return_attn=False):
        """Forward pass with optional attention return."""
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Layer 1
        x, attn_info1 = self.layer1(x, return_attn=True)
        x = F.relu(x)
        
        # Layer 2
        x, attn_info2 = self.layer2(x, return_attn=True)
        x = F.relu(x)
        
        # Classifier
        logits = self.classifier(x)
        
        if return_attn:
            return logits, {'layer1': attn_info1, 'layer2': attn_info2}
        return logits


def create_split_mnist_tasks(num_tasks=5):
    """Create SplitMNIST dataset split into tasks.
    
    Each task contains 2 digit classes.
    Task 0: digits 0, 1
    Task 1: digits 2, 3
    Task 2: digits 4, 5
    Task 3: digits 6, 7
    Task 4: digits 8, 9
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Split into tasks
    tasks_train = []
    tasks_test = []
    
    for task_id in range(num_tasks):
        # Define classes for this task
        task_classes = [task_id * 2, task_id * 2 + 1]
        
        # Filter training data
        train_indices = [i for i, (_, label) in enumerate(train_dataset) 
                        if label in task_classes]
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        
        # Filter test data
        test_indices = [i for i, (_, label) in enumerate(test_dataset) 
                       if label in task_classes]
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        
        tasks_train.append(train_subset)
        tasks_test.append(test_subset)
    
    return tasks_train, tasks_test


def train_epoch(model, dataloader, optimizer, criterion, device='cpu', 
                check_expansion=True, expansion_interval=10):
    """Train for one epoch with optional module expansion."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    expansion_results = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, attn_info = model(data, return_attn=True)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
        
        # Check for expansion periodically
        if check_expansion and batch_idx % expansion_interval == 0:
            with torch.no_grad():
                # Check layer 1
                if attn_info['layer1'] is not None:
                    attn1 = attn_info['layer1']['attention']
                    result1 = model.layer1.maybe_expand(
                        attn1, optimizer=optimizer, verbose=True
                    )
                    if result1['expanded']:
                        expansion_results.append(('layer1', batch_idx, result1))
                
                # Check layer 2
                if attn_info['layer2'] is not None:
                    attn2 = attn_info['layer2']['attention']
                    result2 = model.layer2.maybe_expand(
                        attn2, optimizer=optimizer, verbose=True
                    )
                    if result2['expanded']:
                        expansion_results.append(('layer2', batch_idx, result2))
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, expansion_results


def test(model, dataloader, criterion, device='cpu'):
    """Test the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data, return_attn=False)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def run_projection_phase(model, dataloader, device='cpu'):
    """Run projection phase for newly added modules."""
    print("\nRunning projection phase for newly added modules...")
    
    # Get a batch of data for projection
    data_iter = iter(dataloader)
    data, _ = next(data_iter)
    data = data.to(device)
    data = data.view(data.size(0), -1)  # Flatten
    
    # Run projection for each layer
    if model.layer1.num_modules > model.layer1_config.num_modules:
        print(f"\nLayer 1 projection (modules: {model.layer1.num_modules}):")
        model.layer1.run_projection_phase(data, verbose=True)
    
    # Transform data through layer 1 for layer 2 projection
    with torch.no_grad():
        data_l2, _ = model.layer1(data, return_attn=False)
        data_l2 = F.relu(data_l2)
    
    if model.layer2.num_modules > model.layer2_config.num_modules:
        print(f"\nLayer 2 projection (modules: {model.layer2.num_modules}):")
        model.layer2.run_projection_phase(data_l2, verbose=True)


def main():
    """Main training loop demonstrating module expansion."""
    print("="*70)
    print("MODULE EXPANSION DEMONSTRATION ON SPLITMNIST")
    print("="*70)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    num_tasks = 5
    epochs_per_task = 5
    batch_size = 128
    learning_rate = 0.001
    
    # Create model with expansion enabled
    print("\nCreating expandable modular network...")
    print("  Initial modules per layer: 2")
    print("  Max modules per layer: 8")
    print("  Confidence threshold: 0.4")
    
    model = ExpandableModularNetwork(
        input_dim=784,
        hidden_dim=400,
        output_dim=10,
        num_initial_modules=2,
        max_modules=8,
        confidence_threshold=0.4,
        enable_expansion=True
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Create SplitMNIST tasks
    print(f"\nLoading SplitMNIST with {num_tasks} tasks...")
    tasks_train, tasks_test = create_split_mnist_tasks(num_tasks)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'expansions': [],
        'layer1_modules': [],
        'layer2_modules': []
    }
    
    # Train on each task
    for task_id in range(num_tasks):
        print("\n" + "="*70)
        print(f"TASK {task_id}: Training on digits {task_id*2}, {task_id*2+1}")
        print("="*70)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            tasks_train[task_id], batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            tasks_test[task_id], batch_size=batch_size, shuffle=False
        )
        
        # Train for multiple epochs on this task
        for epoch in range(epochs_per_task):
            print(f"\nEpoch {epoch+1}/{epochs_per_task}")
            
            train_loss, train_acc, expansions = train_epoch(
                model, train_loader, optimizer, criterion, device,
                check_expansion=True, expansion_interval=5
            )
            
            test_loss, test_acc = test(model, test_loader, criterion, device)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['layer1_modules'].append(model.layer1.num_modules)
            history['layer2_modules'].append(model.layer2.num_modules)
            
            if expansions:
                history['expansions'].extend(expansions)
                print(f"\n  ⚡ {len(expansions)} expansion(s) occurred this epoch!")
                for layer_name, batch_idx, result in expansions:
                    print(f"    {layer_name}: {result}")
                
                # Run projection phase after expansions
                run_projection_phase(model, train_loader, device)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"  Layer 1 modules: {model.layer1.num_modules}/{model.layer1.max_modules}")
            print(f"  Layer 2 modules: {model.layer2.num_modules}/{model.layer2.max_modules}")
    
    # Final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    
    print(f"\nLayer 1:")
    stats1 = model.layer1.get_expansion_stats()
    print(f"  Final modules: {stats1['num_modules']}/{stats1['max_modules']}")
    print(f"  Total expansions: {stats1['expansion_count']}")
    if stats1['expansion_history']:
        print(f"  Expansion history:")
        for step, num_mods, conf in stats1['expansion_history']:
            print(f"    Step {step}: {num_mods} modules (confidence={conf:.4f})")
    
    print(f"\nLayer 2:")
    stats2 = model.layer2.get_expansion_stats()
    print(f"  Final modules: {stats2['num_modules']}/{stats2['max_modules']}")
    print(f"  Total expansions: {stats2['expansion_count']}")
    if stats2['expansion_history']:
        print(f"  Expansion history:")
        for step, num_mods, conf in stats2['expansion_history']:
            print(f"    Step {step}: {num_mods} modules (confidence={conf:.4f})")
    
    # Plot training curves
    print("\nGenerating plots...")
    plot_training_history(history)
    
    print("\n✅ Training complete!")
    return model, history


def plot_training_history(history):
    """Plot training history with module counts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
    axes[0, 0].plot(history['test_loss'], label='Test Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', alpha=0.7)
    axes[0, 1].plot(history['test_acc'], label='Test Acc', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Test Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Module counts over time
    axes[1, 0].plot(history['layer1_modules'], label='Layer 1', marker='o', alpha=0.7)
    axes[1, 0].plot(history['layer2_modules'], label='Layer 2', marker='s', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Number of Modules')
    axes[1, 0].set_title('Module Growth Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Expansion events
    if history['expansions']:
        expansion_steps = [exp[1] for exp in history['expansions']]
        expansion_layers = ['L1' if exp[0] == 'layer1' else 'L2' 
                          for exp in history['expansions']]
        
        # Count expansions by layer
        l1_count = sum(1 for layer in expansion_layers if layer == 'L1')
        l2_count = sum(1 for layer in expansion_layers if layer == 'L2')
        
        axes[1, 1].bar(['Layer 1', 'Layer 2'], [l1_count, l2_count], alpha=0.7)
        axes[1, 1].set_ylabel('Number of Expansions')
        axes[1, 1].set_title('Total Expansions by Layer')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 1].text(0.5, 0.5, 'No expansions occurred', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Total Expansions by Layer')
    
    plt.tight_layout()
    plt.savefig('module_expansion_results.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to: module_expansion_results.png")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the demonstration
    model, history = main()

