# Jigsaw Puzzle Solver - FYP Project

A deep learning-based system for solving mixed jigsaw puzzles using clustering and reinforcement learning techniques.

## Overview

This project combines:
- **DEPP (Deep Embedded Puzzle Piece)**: Clustering puzzle pieces by puzzle identity
- **RLRS (Reinforcement Learning Reassembler)**: Solving individual puzzles using RL and DQN
- **Iterative Refinement**: Progressive improvement through constrained reclustering

## Features

- Mixed puzzle instance generation with ground truth annotations
- Automatic puzzle piece clustering
- RL-based puzzle reassembly
- Iterative refinement with constrained clustering
- Incorrect edge annotation (red line highlighting)
- Parallel processing for efficiency

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional but recommended)

### Dependencies

```
opencv-python>=4.5.0
numpy>=1.19.0
scipy>=1.5.0
torch>=1.9.0
torchvision>=0.10.0
stable-baselines3>=1.0.0
scikit-learn>=0.24.0
pillow>=8.0.0
tqdm>=4.50.0
networkx>=2.5
PyYAML>=5.3
```

### Setup

1. **Clone the repository**

```bash
cd d:/school/FYP/MM
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python numpy scipy torch torchvision stable-baselines3 scikit-learn pillow tqdm networkx PyYAML
```

3. **Prepare model files**

Place pre-trained models in the `./model/` directory:
- `dscl_vit_epoch_40.pth` - DEPP clustering model
- `hori_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth` - Horizontal edge classifier
- `vrti_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth` - Vertical edge classifier
- `best_efficientnetb3.pth` - Parity direction network (PDN)
- `jigsaw_dqn_final.pth` - DQN model weights

## Usage

### 1. Generate Mixed Puzzle Instances

Create mixed puzzle instances from a folder of images:

```bash
python DEPP/dataset_with_gt.py
```

Edit the script to configure:
- `source_folder`: Input image directory
- `out_folder`: Output directory
- `n_instances_per_mix`: Number of instances per mix
- `mix_list`: Puzzle mix sizes (e.g., [6, 12, 20, 30])

### 2. Solve Puzzles

Solve a mixed puzzle instance with command-line parameters:

#### Basic Usage

```bash
python main.py --image_path ./mix.png --meta_path ./meta.json
```

#### Command-Line Options

All configuration is done via command-line arguments (no code editing needed):

```bash
python main.py --help
```

Common options:

| Option | Default | Description |
|--------|---------|-------------|
| `--image_path` | `./sample/instance_000000/mix.png` | Path to mixed puzzle image |
| `--meta_path` | `./sample/instance_000000/meta.json` | Path to metadata JSON file |
| `--output_prefix` | `./output/output` | Prefix for output files |
| `--max_steps` | 20 | Maximum refinement iterations |
| `--workers` | 4 | Number of parallel workers |
| `--rl_steps` | 20 | RL steps per cluster |
| `--success_threshold` | 0.9 | Success score threshold (0-1) |
| `--highlight_wrong_edges` | False | Highlight incorrect edges in red |
| `--save_annotated` | False | Save annotated cluster images |
| `--early_stop_patience` | 100 | Early stopping patience |

#### Examples

**Basic solve with default parameters:**

```bash
python main.py --image_path ./puzzle_dataset/6_mix/test/instance_000000/mix.png \
               --meta_path ./puzzle_dataset/6_mix/test/instance_000000/meta.json
```

**With edge highlighting and annotation saving:**

```bash
python main.py --image_path ./mix.png \
               --meta_path ./meta.json \
               --highlight_wrong_edges \
               --save_annotated \
               --output_prefix ./output/result
```

**Custom parameters for faster solving:**

```bash
python main.py --image_path ./mix.png \
               --meta_path ./meta.json \
               --max_steps 50 \
               --workers 8 \
               --rl_steps 100 \
               --success_threshold 0.85
```

**Aggressive refinement for tough puzzles:**

```bash
python main.py --image_path ./mix.png \
               --meta_path ./meta.json \
               --max_steps 100 \
               --rl_steps 200 \
               --early_stop_patience 20
```

### 3. Rearrange Puzzle Tiles

Cut and rearrange puzzle images:

```bash
python test.py input.png output.png --output_rows 2 --output_cols 3
```

Options:
- `--tile_size`: Tile size (default: 96)
- `--order`: Optional tile order for rearrangement

Example with custom order:

```bash
python test.py input.png output.png --output_rows 3 --output_cols 3 --order 0 1 2 3 4 5 6 7 8
```

## Project Structure

```
.
├── main.py                          # Main puzzle solver
├── main_new.py                      # Alternative solver with edge highlighting
├── utils.py                         # Utility functions
├── test.py                          # Tile rearrangement tool
├── DEPP/
│   ├── DEPP.py                     # Clustering module
│   ├── dataset.py                  # Dataset utilities
│   ├── dataset_with_gt.py          # GT-aware dataset generation
│   ├── clustering_dataset/         # Pre-generated clustering dataset
│   └── __pycache__/
├── RLRS/
│   ├── RLRS.py                     # Reinforcement learning reassembler
│   ├── train.py                    # RL training script
│   ├── train_swap.py               # Swap-based training
│   ├── environment_swap.py         # RL environment
│   └── __pycache__/
├── model/                          # Pre-trained models
├── puzzle_dataset/                 # Test datasets
├── output/                         # Results directory
└── README.md                       # This file
```

## Output Files

When solving a puzzle, the system generates in the output directory:

1. **Step images** (`{output_prefix}_step_{N}.jpg`)
   - Composite images showing all cluster results at each iteration
   - Useful for debugging and visualization

2. **Annotated cluster images** (when `--save_annotated` is used)
   - `{output_prefix}_cluster_{ID}_step_{N}.jpg`
   - Red lines mark incorrect edges within clusters
   - Only created when `--highlight_wrong_edges` is enabled

3. **Console output**
   - Real-time progress updates
   - Per-cluster solving status
   - Final accuracy metrics

Example output structure:
```
output/
├── output_step_0.jpg           # Iteration 0 results
├── output_step_1.jpg           # Iteration 1 results
├── output_cluster_0_step_0.jpg # Cluster 0 annotated (if enabled)
├── output_cluster_1_step_0.jpg # Cluster 1 annotated (if enabled)
└── output_cluster_0_step_1.jpg # Cluster 0 iteration 1 (if enabled)
```

## Performance Metrics

The solver reports three accuracy metrics:

1. **Perfect Accuracy**: Percentage of puzzles completely solved correctly
2. **Neighbor Accuracy**: Percentage of tile pairs with correct adjacency
3. **Absolute Accuracy**: Percentage of individual tiles in correct positions

## API Reference

### Main Functions

#### `solve_puzzle()`

Solve a mixed puzzle instance programmatically.

```python
from main import solve_puzzle

result = solve_puzzle(
    image_path="./mix.png",
    meta_path="./meta.json",
    output_prefix="./output/result",
    max_refine_steps=30,
    num_workers=4,
    rl_max_steps=100,
    success_threshold=0.9,
    highlight_wrong_edges=True,
    save_annotated_cluster_images=True,
)

print(f"Perfect accuracy:  {result['perfect_accuracy']:.4f}")
print(f"Neighbor accuracy: {result['neighbor_accuracy']:.4f}")
print(f"Absolute accuracy: {result['absolute_accuracy']:.4f}")
```

#### `cut_image_into_tiles_and_rearrange()`

Rearrange puzzle tiles to a target grid.

```python
from test import cut_image_into_tiles_and_rearrange

tiles, arranged = cut_image_into_tiles_and_rearrange(
    image_path="input.png",
    output_shape=(2, 3),
    tile_size=96,
    save_path="output.png"
)
```

#### `build_dataset_multi_mix()`

Generate mixed puzzle instances from images.

```python
from DEPP.dataset_with_gt import build_dataset_multi_mix

build_dataset_multi_mix(
    source_folder="./images",
    out_folder="./datasets",
    n_instances_per_mix=1000,
    mix_list=[6, 12, 20, 30],
    num_workers=8
)
```

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA out of memory errors or system slowdown

**Solutions:**
- Reduce `--workers` to 2-4
- Reduce `--rl_steps` to 50-100
- Use smaller puzzle sizes

```bash
python main.py --image_path ./mix.png --meta_path ./meta.json --workers 2 --rl_steps 50
```

### Poor Accuracy

**Symptoms:** Low perfect/neighbor/absolute accuracy scores

**Solutions:**
- Increase `--max_steps` for more iterations
- Lower `--success_threshold` to be less strict
- Increase `--rl_steps` for better solving

```bash
python main.py --image_path ./mix.png --meta_path ./meta.json --max_steps 50 --rl_steps 150
```

### Model Not Found

**Symptoms:** `FileNotFoundError` about missing models

**Solutions:**
Ensure all model files are in `./model/`:

```bash
ls ./model/
# dscl_vit_epoch_40.pth
# hori_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth
# vrti_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth
# best_efficientnetb3.pth
# jigsaw_dqn_final.pth
```

## Citation
