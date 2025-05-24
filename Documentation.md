
# Documentation

##  Approach Overview

This solution addresses the task of extracting graph structure from images and computing the **minimum fuel cost** from a source to target node using constrained movement rules.

The pipeline is composed of the following major stages:

### 1.  Image-to-Graph Model

I designed a **Convolutional Neural Network (CNN)** that predicts:
- A **6×6 directed adjacency matrix** representing the graph topology
- The **number of active nodes (N)** in the graph

**Input**: PNG image of the graph (from drone/satellite capture)  
**Output**: Flattened 6×6 binary adjacency matrix + predicted node count

This supervised learning model is trained using ground-truth adjacency matrices from a CSV file.

---

### 2. Graph Path Solver

Given the predicted adjacency matrix and number of nodes, I implemented a custom **path-finding algorithm** using Dijkstra’s framework enhanced for the following domain-specific movement rules:

- **Traverse Edge**: If `v → u` exists, move at cost 1
- **Reverse Signals**: Flip entire graph directions (e.g., `v → u` becomes `u → v`), costing `N` fuel units

This models the ability of Phantom Unit-1 to either travel along visible paths or flip entire infrastructure signals.

---


## Performance Observations

With extended training (100 epochs), the model achieves high structural accuracy in adjacency prediction on clean graph images. Minor inconsistencies remain in edge cases with occlusions or very sparse graphs.
