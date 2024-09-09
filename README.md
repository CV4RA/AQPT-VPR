# Quadruplet Transformer for Robot Place Recognition

This repository contains the implementation of the paper: **"Quadruplet Transformer Modeling with Weighted Multi-Level Attention Aggregation for Robot Place Recognition"**. This work leverages a Vision Transformer (ViT) backbone with multi-level attention aggregation and a quadruplet loss function for robust large-scale visual place recognition tasks.

![alt text](framework.jpg)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Configuration](#configuration)


## Introduction

Visual place recognition (VPR) is a key problem in robotics and computer vision, where the goal is to determine whether a current scene has been previously visited. The proposed **Aggregated Quadruplet Pyramid Transformer (AQPT)** architecture offers enhanced performance by integrating multi-channel attention and utilizing a quadruplet loss function with Bayesian learning.

## Installation

### Prerequisites

Make sure you have the following dependencies installed:

- Python 3.7+
- PyTorch 1.9+
- torchvision 0.10+
- transformers
- NumPy
- Pillow (PIL)
- scikit-learn

### Install Dependencies

To install the required packages, run:

```bash
pip install -r requirements.txt
```
### training
Once the model is trained, you can evaluate its performance on the test set using the test.py script:
```bash
python train.py --model checkpoints/aqpt_model.pth --dataset data/KITTI/test/
```
### Testing
Once the model is trained, you can evaluate its performance on the test set using the test.py script:
```bash
python test.py --model checkpoints/aqpt_model.pth --dataset data/KITTI/test/
```
Arguments:

--model: Path to the saved model checkpoint.
--dataset: Path to the test dataset directory.
The script outputs evaluation metrics such as recall, precision, and visualizes the place recognition results.
              


