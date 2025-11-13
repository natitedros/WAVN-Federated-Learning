# 🛰️ Federated Visual Navigation System for GPS-Denied Environments

## Overview
This project implements a **visual navigation system** that enables **autonomous localization of robots** in **GPS-denied environments**. It combines **Federated Learning (FL)** with **Blockchain consensus mechanisms** to ensure secure, efficient, and reliable model training across distributed robotic nodes.

---

## 🚀 Key Features

- **Federated Learning Architecture:**  
  Distributed CNN models trained locally on each robot to preserve data privacy and minimize communication overhead.

- **Robust Visual Localization:**  
  Enables reliable navigation in environments where GPS is unavailable or unreliable.

---

## 🧠 System Architecture

1. **Local Model Training (on each robot):**  
   Each robot trains a CNN on locally captured visual data.

2. **Federated Aggregation:**  
   Model updates (not raw data) are shared with a blockchain-based aggregator.

4. **Global Model Update:**  
   The validated model is redistributed to all nodes for improved navigation accuracy.

---

## 🧩 Technologies Used

- **Python / Keras** – for CNN model training and evaluation
  
---

## 📊 Results

- Ongoing

---

## 🧪 Future Work

- Integrate reinforcement learning for adaptive path planning  
- Deploy on physical robot swarms for real-world testing  
- Extend blockchain layer with lightweight consensus for embedded devices  

