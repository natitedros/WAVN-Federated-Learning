# Federated Learning for Wide-Area Visual Navigation

> **Paper:** [Link to paper — to be added upon publication]

## Overview

This repository contains the implementation for a federated learning approach to visual homing navigation in GPS-denied environments. Two models are developed: a **Federated Siamese CNN** and a **Graph Neural Network (GNN)**. The Siamese CNN learns directional homing actions from paired current and destination views using RGB and edge-enhanced inputs, enabling robots to collaboratively improve navigation policies without sharing raw visual data. Experiments conducted in a Gazebo-based multi-robot simulation show that the federated Siamese model achieves 85.8% overall accuracy, with clear gains from edge-augmented inputs and decentralized aggregation.

---

## Repository Structure

```
.
├── 01_Federated_Siamese_Model/     # Core federated Siamese CNN training and evaluation
│   ├── CNN_with_5_fold_cv.ipynb    # 5-fold cross-validation experiments
│   └── FL_results_acc/             # Accuracy, F1, precision, and recall result plots
├── 02_Dataset_Generation/          # Edge detection and image segmentation scripts
│   ├── Object_Segmentation_using_Edge_detection.py
│   └── edge_segmentation.py
├── 03_Model_and_Data_Comparison/   # Comparative analysis across models and input types
│   ├── CNN_comparisons_01192026.ipynb
│   └── CNN_edge_data_comparison.ipynb
├── hed_model/                      # HED pretrained model for edge-enhanced inputs
└── test_impl/                      # Federated simulation and training scripts
    ├── FL_simulation.ipynb
    ├── FL_with_5-fold_CV.ipynb
    ├── cnn_homing_fl.py
    └── train_test_federated.py
```

---

## Models

### Federated Siamese CNN

A Siamese convolutional neural network trained under a federated learning framework. The model takes paired image inputs — a current view and a destination view — and predicts the directional homing action. Training is distributed across simulated robot clients, with only model updates (not raw data) aggregated globally.

### GNN Model

A graph neural network model developed as part of the comparative analysis. Details available in `03_Model_and_Data_Comparison/`.

---

## Key Results

| Metric               | Value                                    |
| -------------------- | ---------------------------------------- |
| Overall Accuracy     | 85.8%                                    |
| Convergence          | Stable across federated rounds           |
| Edge-augmented input | Improved accuracy over RGB-only baseline |

Full plots (accuracy, F1, precision, recall) are available in [01_Federated_Siamese_Model/FL_results_acc/](01_Federated_Siamese_Model/FL_results_acc/).

---

## Setup

**Dependencies:** Python, Keras, TensorFlow, OpenCV

Install dependencies:

```bash
pip install tensorflow keras opencv-python
```

**Edge Detection:** The HED (Holistically-nested Edge Detection) pretrained model (`hed_model/hed_pretrained_bsds.caffemodel`) is required for edge-enhanced input generation. See [02_Dataset_Generation/](02_Dataset_Generation/) for preprocessing scripts.

---

## Usage

1. Generate edge-augmented dataset using scripts in [02_Dataset_Generation/](02_Dataset_Generation/) and the [hed_model/](hed_model/).
2. Run federated training via the notebook in [01_Federated_Siamese_Model/](01_Federated_Siamese_Model/).
3. Evaluate and compare results using notebooks in [03_Model_and_Data_Comparison/](03_Model_and_Data_Comparison/).

---

## Citation

> Citation to be added upon publication.
