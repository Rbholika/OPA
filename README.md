# One Pixel Attack (OPA)

This repository contains an implementation and experimental code for One‑Pixel Attack (OPA) — an adversarial attack that perturbs only a single pixel in an image to fool deep neural networks. The project collects code, experiments, and analysis related to the original One‑Pixel Attack research and follow‑up experiments.

> Note: This README is a general, ready‑to‑use description. If you want the README customized to exact file names and commands in this repo, please provide the repo files or give me permission to read the repository and I will update it accordingly.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Run attack (example)](#run-attack-example)
  - [Run attacks on a batch / dataset](#run-attacks-on-a-batch--dataset)
  - [Evaluate results](#evaluate-results)
- [Experiments & Results](#experiments--results)
- [Visualizations](#visualizations)
- [Reproducibility notes](#reproducibility-notes)
- [Citing](#citing)
- [License](#license)
- [Contact](#contact)

## Overview
One‑Pixel Attack demonstrates that carefully crafted changes to just one pixel in an image can cause state‑of‑the‑art classifiers to misclassify. This repository provides:
- Implementation of the one‑pixel attack algorithm (genetic algorithm or other optimization used in the original work).
- Scripts to run attacks against common architectures (e.g., ResNet, VGG, Inception) and datasets (e.g., CIFAR‑10, ImageNet subsets).
- Notebooks for exploratory analysis and visualization.
- Experiment results and plots.

## Features
- Configurable target/untargeted one‑pixel attack.
- Support for multiple model backends (PyTorch / TensorFlow / Keras) — adapt as needed.
- Batch attack and per‑image attack scripts.
- Logging and checkpointing of successful attacks.
- Utilities for evaluating attack success rate and perturbation analysis.

## Repository structure
(Replace the entries below with the real file/folder names from the repo.)
- `README.md` — This file.
- `src/` — Core attack implementation, model wrappers, utilities.
- `notebooks/` — Jupyter notebooks showing demo runs and visualizations.
- `experiments/` — Scripts and configurations to reproduce experiments.
- `models/` — Pretrained model checkpoints or instructions to download them.
- `data/` — Sample images or references to dataset download scripts.
- `results/` — Saved outputs from experiments (attack logs, images, metrics).
- `requirements.txt` — Python dependencies.
- `setup.py` / `pyproject.toml` — (Optional) packaging files.
- `LICENSE` — Licensing information.

If actual file names differ, I can update this section after inspecting the repo.

## Requirements
- Python 3.8+
- numpy, scipy
- torch (or tensorflow/keras) — depending on model implementation
- opencv-python (cv2) or Pillow
- tqdm
- matplotlib, seaborn (for plotting)
- (Optional) scikit‑learn, pandas

Install with pip:
pip install -r requirements.txt

If the repo uses Conda, create a conda environment:
conda env create -f environment.yml
conda activate opa

## Installation
1. Clone the repo:
   git clone https://github.com/<owner>/OPA.git
   cd OPA

2. Install requirements:
   pip install -r requirements.txt

3. (Optional) Download pretrained models or datasets following instructions in `models/` and `data/`.

## Usage

### Run attack (example)
A minimal example (adjust script names & CLI flags to match repository):

python src/attack.py \
  --model resnet50 \
  --backend pytorch \
  --image_path examples/dog.png \
  --targeted False \
  --max_iter 75 \
  --pop_size 400 \
  --output results/dog_attack.json

This runs a one‑pixel attack against a single image using a genetic optimizer and saves the results.

### Run attacks on a batch / dataset
Example (adjust to actual scripts in the repo):

python src/run_batch_attacks.py \
  --model resnet50 \
  --dataset data/cifar10/test_images/ \
  --n_samples 500 \
  --save_dir results/

This will attempt one‑pixel attacks on multiple images and save aggregated metrics.

### Evaluate results
Use evaluation script to compute success rate, average queries, and visualize per-class attack success:

python src/evaluate_results.py --results_dir results/ --output reports/summary.csv

or run provided notebooks in `notebooks/` for interactive analysis.

## Experiments & Results
The repository contains example experiment configurations showing:
- Untargeted attacks on CIFAR‑10 / ImageNet subsets
- Targeted attacks for selected source/target class pairs
- Sensitivity analysis vs. model architecture

Include links to any saved plots or notebooks in the `results/` and `notebooks/` folders. If you share the exact experiment outputs I can add the figures and summary tables here.

## Visualizations
Typical visualizations include:
- Original and adversarial images (highlight the changed pixel).
- Confusion matrix showing misclassifications induced by attacks.
- Success rate vs. number of queries or population size.
- Pixel location heatmaps showing most influential pixel positions.

## Reproducibility notes
- Random seed: ensure a seed is set for the optimizer and the model to reproduce results.
- Model preprocessing: confirm exact normalization (mean/std) and input resize used by the model.
- If using pretrained model downloads, note the model versions and commit/tag of the weights.

Example for deterministic runs:
python src/attack.py --seed 42 ...

## Citing
If you use this code in your research, please cite the original One‑Pixel Attack paper:

Jiawei Su, Danilo Vasconcellos Vargas, Kouichi Sakurai. "One Pixel Attack for Fooling Deep Neural Networks." IEEE Transactions on Evolutionary Computation (2019). arXiv:1710.08864

And optionally cite this repository (provide repo citation once available).

## License
Specify license here (e.g., MIT). If the repository already has a LICENSE file, use that. Example:
This project is licensed under the MIT License — see the LICENSE file for details.

## Contact
For questions or contributions, open an issue or contact:
- Maintainer: <Your Name> — (replace with your email or GitHub handle)
- Repository: https://github.com/<owner>/OPA

Contributions are welcome — feel free to open issues or PRs.

---

If you want, I can:
1. Try reading the repository again and then update this README to list exact scripts, parameters, and example commands; or
2. Commit this README directly to the repository on a branch of your choice (please confirm branch name, e.g., `main` or `add-readme`).

Which would you prefer? If you want an updated README based on actual files, grant read access or provide the file list and I will update it.
