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
              
In addition, if you want to run the code, please ensure you have the necessary datasets (e.g., [KITTI](https://www.cvlibs.net/datasets/kitti/), [EuRoc](), [VPRICE](), [Nordland](https://nrkbeta.no/2013/01/15/nordlandsbanen-minute-by-minute-season-by-season/)) available for training and evaluation.

# Real-World Visual Place Recognition using AQPT

This document outlines the architecture and implementation for using the **Aggregated Quadruplet Pyramid Transformer (AQPT)** in a real-world visual place recognition experiment. The setup uses a robot platform with a camera and onboard computing hardware (Jetson Xavier NX) for live image capture and inference.

## 1. System Overview

The system is divided into several components:
- **Robot Platform**: The robot captures live image data using a mounted camera.
- **Jetson Xavier NX**: This onboard computing unit processes the images and runs the AQPT model.
- **AQPT Model**: The Aggregated Quadruplet Pyramid Transformer is used to extract features from the captured images and match them against a database of previously visited places.
- **Real-time Visualization**: The results of the place recognition are displayed in real-time for analysis.

### System Flow:
1. **Image Capture**: The robot captures real-time images of the environment using its camera.
2. **Preprocessing**: Captured images are resized, normalized, and prepared for inference.
3. **Feature Extraction**: The AQPT model extracts multi-scale features from the images.
4. **Place Matching**: The extracted features are compared against a reference database of known places.
5. **Results Visualization**: The matched place and its confidence score are displayed in real-time.

---

## 2. Hardware Setup

### Components:
- **Robot Platform**: A mobile robot equipped with wheels or tracks for navigating environments.
- **Jetson Xavier NX**: Onboard GPU-enabled processing unit for real-time inference.
- **Camera**: (e.g., Intel RealSense, ZED Stereo Camera) for live image capture.
- **Sensors (Optional)**: IMU or GPS for additional localization.

### Hardware Flow:
- **Camera** captures images → Images are sent to the **Jetson Xavier NX** → The AQPT model processes the images and performs inference → The robot's path is updated with recognized places.

---

## 3. Software Setup

### 3.1 Dependencies:
- **Operating System**: Ubuntu 18.04/20.04 with ROS (Robot Operating System) for robot control.
- **Deep Learning Libraries**: PyTorch, TorchVision, and TensorRT for optimized inference on the Jetson Xavier NX.
- **OpenCV**: For camera feed capture and visualization.

### 3.2 Model Setup:
- **Pre-trained AQPT Model**: The AQPT model pre-trained on large-scale place recognition datasets.
- **Image Preprocessing**: Resize, normalize, and convert images into the required format.

```python
# Install dependencies on Jetson Xavier NX
sudo apt-get update
sudo apt-get install python3-pip
pip3 install torch torchvision pillow opencv-python
```
Experiments are conducted in both day and night scenarios to test the model's robustness. The robot's driving trajectory is recorded, and the recognized places are visualized in real-time.

```python
# Run the real-time inference
python real_time_inference.py
```