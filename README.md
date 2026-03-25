# NTVGCL: Natural Twin Views Graph Contrastive Learning for Self-Supervised In-Vehicle Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyG](https://img.shields.io/badge/PyG-2.0+-green.svg)](https://www.pyg.org/)

This repository contains the official implementation of **NTVGCL**, a self-supervised in-vehicle intrusion detection method. By leveraging a **Siamese architecture** and the principle of temporal consistency, NTVGCL derives **Natural Twin Views (NTVs)** from temporally adjacent Message Interaction Graphs (MIGs), eliminating the need for hand-crafted augmentations.

---

## 🚀 Key Features

* **Self-Supervised Learning**: No labeled attack data is required during the representation learning phase.
* **Deterministic Twin Views**: Captures robust invariant representations of normal CAN traffic patterns via temporal adjacency, avoiding the instability of traditional unsupervised methods.
* **Superior Efficacy**: Consistently outperforms state-of-the-art models, achieving an **AUC of 99%** and **F1-scores ranging from 94% to 99%**.
* **Anomaly Discrimination**: Utilizes an autoencoder-based module to quantify deviations through reconstruction errors efficiently.

---

## 📂 Project Structure

The project follows a structured pipeline as reflected in the file naming convention:

* **one_data_process_*.py**: Scripts for raw CAN traffic parsing and Message Interaction Graph (MIG) construction.
* **two_twin_dataset.py**: Implementation of the Natural Twin Views (NTVs) data loader and dataset management.
* **three_train_twin_gcl_simsiam.py**: Core Siamese training logic for self-supervised representation learning.
* **three_grid_search.py & four_find_best_train_model.py**: Hyperparameter optimization and model selection.
* **four_infer_simsiam_unsuper-Offset=0.py**: Anomaly detection and inference module using reconstruction errors.
* ***_graphs_results.zip**: Compressed archives containing pre-trained models and detailed experimental results for each benchmark.

---

## 🛠️ Getting Started

### 1. Requirements
Ensure you have Python 3.8+ and the following dependencies installed:
**pip install torch torch-geometric pandas numpy scikit-learn**

### 2. Dataset Download
The experiments are conducted on two widely recognized benchmarks. Please download the raw datasets from the official OCS Lab:
* [Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
* [Survival-IDS Dataset](https://ocslab.hksecurity.net/Datasets/survival-ids)

### 3. Pipeline Execution
**Step 1. Preprocessing**: Transform raw CAN logs into Message Interaction Graphs (MIGs).
> python one_data_process_CarHacking.py
> python one_data_process_Survial.py

**Step 2. Training**: Train the NTVGCL model to learn normal pattern representations.
> python three_train_twin_gcl_simsiam.py

**Step 3. Inference**: Evaluate the detection performance using reconstruction errors.
> python four_infer_simsiam_unsuper-Offset=0.py

---

## 📊 Performance Summary
NTVGCL has been rigorously tested on four benchmark datasets.

* **AUC-ROC**: ~99%
* **F1-Score**: 94% - 99%

---

## 📧 Contact
**Yaru He** - heyaru@bupt.edu.cn
Beijing University of Posts and Telecommunications (BUPT).
