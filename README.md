# Modular Continual Learning with Task-Agnostic Subnetwork Routing

This project explores **Modular Continual Learning with Task-Agnostic Subnetwork Routing via Attention Mechanisms**. The system enables continual learning without explicit task boundaries by dynamically routing information through subnetworks using attention mechanisms.

## 🎯 Key Features

- **Task-Agnostic Learning**: No explicit task identity required at test time
- **Dynamic Module Expansion**: Automatically grows capacity when needed
- **Attention-Based Routing**: Learns which modules to activate for each input
- **Balanced Training**: Prevents routing collapse with curriculum learning
- **Comprehensive Testing**: 100% test coverage with reproducible results

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**SETUP.md**](SETUP.md) | Installation, prerequisites, and quick start guide |
| [**ARCHITECTURE.md**](ARCHITECTURE.md) | Technical architecture and design details |
| [**TESTING.md**](TESTING.md) | Test suites and validation procedures |
| [**RESULTS.md**](RESULTS.md) | Experimental results and performance metrics |
| [**CHANGELOG.md**](CHANGELOG.md) | Project milestones and implementation history |

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run a Simple Experiment
```bash
./main.py --experiment=splitMNIST --scenario=task --si
```

### Run Modular Training with Expansion
```bash
python train_splitmnist_modular.py \
    --experiment splitMNIST \
    --scenario task \
    --enable_expansion \
    --confidence_threshold 0.4
```

### Run Full Experiment Harness
```bash
python -m experiments.run_experiments
```

## 🏗️ Repository Structure

### Core Components
- **`models/`**: Modular architectures, attention routers, and continual learning components
- **`data/`**: Dataset loading and stream utilities
- **`experiments/`**: Reproducible experiment harness and training scripts
- **`Documentation/`**: Detailed technical documentation

### Key Scripts
- **`main.py`**: Single continual learning experiments
- **`main_task_free.py`**: Task-free continual learning streams
- **`compare.py`**: Automated method comparisons
- **`train_splitmnist_*.py`**: Modular training implementations

For detailed setup instructions, see [**SETUP.md**](SETUP.md).

## 📖 Background

This is a PyTorch implementation of continual learning experiments with deep neural networks, extending the work described in:
* [Three types of incremental learning](https://www.nature.com/articles/s42256-022-00568-3) (2022, *Nature Machine Intelligence*)

The repository supports both *academic continual learning settings* (clearly separated contexts) and more flexible *task-free* continual learning experiments with gradual transitions between contexts.

## 🎓 Tutorials and Demos

### NeurIPS Tutorial "Lifelong Learning Machines"
This code repository is used for the [NeurIPS 2022 tutorial "Lifelong Learning Machines"](https://sites.google.com/view/neurips2022-llm-tutorial).

### ICLR Blog Post
This repository is also used for the [ICLR 2025 blog post "On the computation of the Fisher Information in continual learning"](https://iclr-blogposts.github.io/2025/blog/fisher/).

## 📋 Requirements

- **Python**: 3.10.4
- **PyTorch**: 1.11.0
- **Torchvision**: 0.12.0
- **Operating System**: Linux/macOS/Windows (tested on Fedora)

For complete installation instructions, see [**SETUP.md**](SETUP.md).

## 🎮 Demos

### Demo 1: Single Continual Learning Experiment
```bash
./main.py --experiment=splitMNIST --scenario=task --si
```
Runs Synaptic Intelligence on Split MNIST task-incremental learning scenario.
- **Runtime**: ~6 minutes on CPU (~3 minutes on GPU)
- **Output**: Training progress, model details, and results

### Demo 2: Method Comparison
```bash
./compare.py --experiment=splitMNIST --scenario=task
```
Compares multiple continual learning methods on Split MNIST.
- **Runtime**: ~100 minutes on CPU (~45 minutes on GPU)
- **Output**: Summary PDF with comparative results

### Demo 3: Modular Training with Expansion
```bash
python example_module_expansion.py
```
Demonstrates dynamic module expansion on SplitMNIST.
- **Features**: Automatic capacity growth, attention routing
- **Output**: Training curves and expansion statistics

## 🔬 Advanced Usage

### Reproducing Paper Results
The script `all_results.sh` provides step-by-step instructions for re-running experiments and recreating tables/figures from "Three types of incremental learning". Consider parallelizing experiments for faster execution.

### Custom Experiments
- **Academic Setting**: Use `main.py` with various methods (EWC, SI, LwF, ER, etc.)
- **Task-Free Setting**: Use `main_task_free.py` for flexible continual learning
- **Modular Training**: Use `train_splitmnist_*.py` for modular architectures

### Visualization
Enable on-the-fly plots with Visdom:
```bash
pip install visdom
python -m visdom.server
# Add --visdom flag to training commands
```

For detailed command-line options and examples, see [**SETUP.md**](SETUP.md).

## 📄 Citation

If you use this code in your research, please consider citing:

```bibtex
@article{vandeven2022three,
  title={Three types of incremental learning},
  author={van de Ven, Gido M and Tuytelaars, Tinne and Tolias, Andreas S},
  journal={Nature Machine Intelligence},
  volume={4},
  pages={1185--1197},
  year={2022}
}
```

## 🙏 Acknowledgments

This research has been supported by:
- IBRO-ISN Research Fellowship
- ERC-funded project *KeepOnLearning* (101021347)
- National Institutes of Health (NIH) awards R01MH109556 and P30EY002520
- DARPA *Lifelong Learning Machines* (L2M) program (HR0011-18-2-0025)
- IARPA via DoI/IBC contract D16PC00003

*Disclaimer: Views and conclusions are those of the authors and should not be interpreted as representing official policies or endorsements of NIH, DARPA, IARPA, DoI/IBC, or the U.S. Government.*

---

## 📊 Status

[![DOI](https://zenodo.org/badge/150479999.svg)](https://zenodo.org/badge/latestdoi/150479999)

- ✅ **Module Expansion**: Production ready with 100% test coverage
- ✅ **Balanced Training**: Fully implemented and tested
- ✅ **Experiment Harness**: Reproducible and comprehensive
- ✅ **Documentation**: Complete with examples and API reference

For the latest updates and implementation status, see [**CHANGELOG.md**](CHANGELOG.md).
