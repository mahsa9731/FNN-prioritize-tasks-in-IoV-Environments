\# ANFIS-Based Task Prioritization in IoV Environments



This repository contains the implementation of an \*\*Adaptive Neuro-Fuzzy Inference System (ANFIS)\*\* for task priority determination in \*\*Internet of Vehicles (IoV)\*\* environments.



The proposed model determines task priority by considering task characteristics, deadline requirements, reliability, computational resource demands, and vehicle conditions. The resulting priority is used to support intelligent task management and offloading decisions in dynamic vehicular environments.



\## Overview



Computational tasks in IoV environments may have different levels of criticality and resource requirements. Therefore, assigning an appropriate priority to each task is essential for efficient task management.



The developed ANFIS model determines task priority based on six input parameters:



| Input | Description |

|---|---|

| Initial Task Priority | Initial criticality level of the task |

| Time Slack | Available time margin before the deadline |

| Deadline | Task execution deadline |

| Reliability | Required task reliability |

| RAM Demand | Memory requirement of the task |

| Vehicle Speed | Current speed of the vehicle |



The model produces a task priority value that can be used by subsequent \*\*task offloading and scheduling mechanisms\*\*.



\## ANFIS Architecture



The proposed model combines neural-network learning with fuzzy inference to handle the uncertainty and dynamic conditions of IoV environments.



The ANFIS architecture consists of the following stages:



1\. Fuzzification

2\. Fuzzy rule evaluation

3\. Normalization

4\. Defuzzification

5\. Task priority determination



\## Training



The ANFIS model is trained using a synthetic dataset containing \*\*6000 task samples\*\* generated under different IoV conditions.



The training configuration includes:



\- \*\*Framework:\*\* PyTorch

\- \*\*Optimizer:\*\* Adam

\- \*\*Loss Function:\*\* Mean Squared Error (MSE)

\- \*\*Training Samples:\*\* 6000



After training, the learned model is saved and can be loaded by the IoV simulation environment for task priority determination.



\## Usage



The trained model can be imported into another Python module using:



```python

from FNN import get\_task\_priority

```



The `get\_task\_priority` function receives the required task and vehicle characteristics and returns the corresponding task priority.



\## Requirements



The implementation is developed using Python and the following libraries:



```text

Python 3.x

PyTorch

NumPy

Pandas

Matplotlib

OpenPyXL

```



Install the required dependencies using:



```bash

pip install torch numpy pandas matplotlib openpyxl

```



\## Project Structure



```text

FNN-prioritize-tasks-in-IoV-Environments/

│

├── FNN.py

├── DAG\_Generator.py

├── model/

├── results/

└── README.md

```



\## Research Context



This implementation is developed for research on intelligent task management and computational offloading in \*\*Internet of Vehicles (IoV)\*\* and \*\*Vehicular Edge Computing (VEC)\*\* environments.



The ANFIS-based priority mechanism is designed to provide task priority information for subsequent \*\*offloading, scheduling, and resource allocation\*\* decisions.



\## License



This repository is intended for \*\*academic and research purposes\*\*.

