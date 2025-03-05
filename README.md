[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Universal Semantic Embeddings for Elements (ElementBERT)

## Overview
This repository contains the official implementation of the paper: "Universal Semantic Embeddings of Chemical Elements for Enhanced Materials Inference and Discovery". ElementBERT is a novel approach for generating semantic embeddings of chemical elements to facilitate materials discovery and inference.

## Repository Structure

+ The pre-training of ElementBERT is shown in folder "ElementBERT_training"
+ The down-stream tasks code is shown in folder "downstream_tasks"
+ The requirements files are listed separately in the above two folders. We use different python envirnments to do these two studies, as the pre-training of ElementBERT is seperate from materails inferences.

## Requirements
- Python 3.10-3.12
- CUDA-compatible GPU (A100 40GB recommended for pre-training)
- Operating Systems: Windows or Linux

## Installation

### Environment Setup
1. Create and activate a conda environment:
```bash
# For pre-training
conda create -n trainingBERT python=3.10
conda activate trainingBERT

# For downstream tasks
conda create -n gprtests python=3.10
conda activate gprtests
```

2. Install dependencies:
```bash
pip install --ignore-installed -r requirements.txt
```

> **Note**: If package installation fails (especially PyTorch), try installing packages individually:
> ```bash
> pip install <package_name>
> # or
> conda install <package_name>
> ```

## Model Pre-training

### Data Preparation
- Source: Web of Science or Scopus databases
- Format: Extract abstracts from .xlsx/.csv files into .md/.txt format
- Data Processing Pipeline:
  1. `Alloy_json_combination.py`: Training/validation set splitting
  2. `Alloy_corpus_normalize.py`: BERT-style text normalization
  3. `Alloy_tokenizer_train.py`: Custom tokenizer training (DeBERTa-v3-xsmall based)
  4. `Alloy_model_train.py`: Model training using DebertaV2ForMaskedLM

### Training Infrastructure
- Computing Environment: HPC cluster with Slurm scheduler
- GPU Requirements: ~30GB VRAM (A100 40GB recommended)
- Execution: Use provided `Alloy_run.sh` for orchestrating the training pipeline

## Downstream Tasks

### Task Categories
1. Regression Analysis
2. Classification
3. Bayesian Optimization (BO) Inference
4. Base Model Verification

### Implementation Details
- Each task category has a dedicated directory
- Core Components:
  - Task-specific Slurm submission scripts (`.sh`)
  - Main task script (`verify_feature_eff.py`)
  - Model-specific hyperparameter tuning scripts for base models (MLP, GPR, RF, XGBoost `*_tuning.py`)

### Usage
To evaluate different alloy systems (SMAs, HEAs, Ti alloys):
1. Navigate to the respective task directory
2. Modify data directory paths in `verify_feature_eff.py`
3. Execute the verification script

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This project builds upon or incorporates code from the following repositories:
- [MGEdata/SteelScientist](https://github.com/mgedata/SteelScientist) - The pre-tarining part.
- [diegoxue/FeatureGradientBO](https://github.com/diegoxue/FeatureGradientBO) - The down-stream tasks part.
