# ANFIS-Based Task Prioritization in IoV Environments

This repository contains the implementation of an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** for task priority determination in **Internet of Vehicles (IoV)** environments.

## Overview

Computational tasks in IoV environments may have different levels of criticality, deadline requirements, reliability requirements, and computational resource demands.

The developed ANFIS model determines task priority based on six input parameters:

- **Initial Task Priority**
- **Time Slack**
- **Deadline**
- **Reliability**
- **RAM Demand**
- **Vehicle Speed**

The obtained priority is used to support subsequent task **offloading and scheduling** decisions in dynamic vehicular environments.

## ANFIS Architecture

The proposed model combines the learning capability of neural networks with the reasoning capability of fuzzy inference systems.

The ANFIS architecture consists of the following stages:

1. Fuzzification
2. Fuzzy rule evaluation
3. Normalization
4. Defuzzification
5. Task priority determination

## Training

The model is trained using a synthetic dataset containing **6000 task samples** generated under different IoV conditions.

Training configuration:

- **Framework:** PyTorch
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Training Samples:** 6000

After training, the learned model is saved and can be loaded by the IoV simulation environment for task priority determination.

## Usage

The trained model can be imported into another Python module using:

```python
from FNN import get_task_priority
```

The `get_task_priority` function receives the required task and vehicle characteristics and returns the corresponding task priority.

## Requirements

The implementation requires Python 3.x and the following libraries:

- PyTorch
- NumPy
- Pandas
- Matplotlib
- OpenPyXL

Install the required dependencies using:

```bash
pip install torch numpy pandas matplotlib openpyxl
```

## Project Structure

```text
FNN-prioritize-tasks-in-IoV-Environments/
│
├── FNN.py
├── DAG_Generator.py
├── model/
├── results/
└── README.md
```

## Research Context

This implementation is developed for research on intelligent task management and computational offloading in **Internet of Vehicles (IoV)** and **Vehicular Edge Computing (VEC)** environments.

The ANFIS-based priority mechanism provides task priority information for subsequent **offloading, scheduling, and resource allocation** decisions.

## License

This repository is intended for **academic and research purposes**.
