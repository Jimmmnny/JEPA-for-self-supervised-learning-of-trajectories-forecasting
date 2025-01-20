# JEPA: Self-Supervised Learning for Trajectory Forecasting

This repository implements the **Joint Embedding Predictive Architecture (JEPA)** for self-supervised learning of trajectory forecasting. It focuses on predicting agent trajectories in static environments and evaluating the quality of predictive embeddings.

## Features

- **Data Handling**: Supports trajectory data in `.npy` format with dataloaders for seamless training and evaluation.
- **Modular Modeling**: Implements a JEPA-based architecture with separate components for encoding states, predicting actions, and representing environmental features.
- **Probing Evaluation**: Provides tools for probing tasks to assess the learned embeddings.
- **Data Augmentation**: Includes horizontal, vertical, and combined flipping augmentations to improve model performance.
- **Visualization Tools**: Enables visualization of agent trajectories and predictions for better insights.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- NVIDIA CUDA (for GPU support)

### Install Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt

## Repository Structure
configs.py : Manages training and evaluation parameters. dataset.py : Contains the WallDataset class and data loaders. evaluator.py : Handles probing evaluation and metric calculations. main.py : The main script for orchestrating training and evaluation workflows. models.py : Defines JEPA model components like encoders, predictors, and probers. normalizer.py : Provides data normalization utilities. output.py : Tools for result visualization and trajectory plotting. run.py : Command-line interface for training, evaluation, and augmentation. schedulers.py : Implements learning rate scheduling strategies. train.py : Contains the main training loop for JEPA models.
