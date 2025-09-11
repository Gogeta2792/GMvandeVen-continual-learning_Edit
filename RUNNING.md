# Running this repository

The instructions below mirror and condense the demos and guidance given in `README.md`.

## 1) Prerequisites
- Python 3.10.4
- PyTorch 1.11.0 and Torchvision 0.12.0 (other versions may work but are untested here)

Install dependencies:
```bash
pip install -r requirements.txt
```

Make scripts executable (first time only):
```bash
chmod +x main*.py compare*.py all_results.sh
```

## 2) Quick sanity check
Run a small single experiment on Split MNIST with Synaptic Intelligence (SI):
```bash
./main.py --experiment=splitMNIST --scenario=task --si
```
Expected runtime: ~6 minutes on CPU (~3 minutes on GPU). The script prints data/model details, training progress, and outputs.

## 3) Compare methods
Run multiple methods on Split MNIST (task-incremental) and summarize results:
```bash
./compare.py --experiment=splitMNIST --scenario=task
```
Expected runtime: ~100 minutes on CPU (~45 minutes on GPU). Produces logs and a summary PDF.

## 4) Run your own academic-setting experiments
Use `main.py` with common options:
- `--experiment`  one of: `splitMNIST|permMNIST|CIFAR10|CIFAR100`
- `--contexts`    number of contexts
- `--scenario`    `task|domain|class`

Examples (pick methods as needed):
```bash
# Baselines
./main.py                    # None (lower target)
./main.py --joint            # Joint (upper target)

# Regularization / replay methods
./main.py --ewc              # EWC (see ICLRblogpost/README.md first)
./main.py --si               # Synaptic Intelligence
./main.py --lwf              # Learning without Forgetting
./main.py --fromp            # FROMP
./main.py --replay=buffer    # ER
./main.py --agem             # A-GEM
./main.py --replay=generative# DGR
./main.py --icarl            # iCaRL

# Other approaches
./main.py --separate-networks
./main.py --xdg              # Context-dependent gating (XdG)

# Discover options
./main.py -h
```

## 5) Task-free continual learning
Run more flexible streams without explicit boundaries using `main_task_free.py`:
- `--experiment`  `splitMNIST|permMNIST|CIFAR10|CIFAR100`
- `--contexts`    number of contexts
- `--stream`      `fuzzy-boundaries|academic-setting|random`
- `--scenario`    `task|domain|class`

Example:
```bash
./main_task_free.py --experiment=splitMNIST --stream=fuzzy-boundaries --scenario=task --si --update-every=500
```
Check all options:
```bash
./main_task_free.py -h
```

## 6) Optional: Visualize training with Visdom
Install and start visdom server:
```bash
pip install visdom
python -m visdom.server
```
Open `http://localhost:8097` in your browser and add `--visdom` to your run command, e.g.:
```bash
./main.py --experiment=splitMNIST --scenario=task --si --visdom
```

## 7) Reproducing paper results
See `all_results.sh` for step-by-step instructions to (re)create tables and figures from the article "Three types of incremental learning". Running as-is can be time-consuming; consider parallelizing experiments.

## 8) Data storage
Datasets and outputs are stored under `store/` by default. The code will download MNIST/CIFAR as needed; ensure you have internet access for first runs.


