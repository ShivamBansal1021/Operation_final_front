
#  Operation Final Front++

> **Mission**: Predict directed graph adjacency from images, and compute **minimum fuel** required to infiltrate Base N-1 under movement constraints.

---

###  Project Structure

```
.
├── data/
│   ├── graphs_images/           # Input graph images (.png)
│   └── adjacency_matrices.csv   # Flattened adjacency matrices + num_nodes
├── model/
│   ├── train_model.py           # CNN to train on image→adjacency+nodes
│   ├── predict_graph.py         # Predict adjacency + num_nodes for 1 image
│   └── graph_solver.py          # Solve min fuel using movement rules
├── models/                      # Trained model weights (.pth)
├── venv/                        # Python virtual environment (optional)
└── main.py                      # CLI to train/predict/solve
```

---

##  1. Environment Setup

You can use pip or conda. Here's a clean pip-based setup:



###  Create & Activate Virtual Env

```bash
python3 -m venv venv
source venv/bin/activate
```

###  Install Requirements

```bash
pip install -r requirements.txt
```

##  2. Train the Model

```bash
python3 model/train_model.py
```

- Uses images and labels in `data/`
- Saves trained model to: `models/graph_model.pth`

---

##  3. Predict + Solve Fuel Cost for a Single Image

```bash
python3 main.py <image path>
```

Outputs:

```txt
Predicted Adjacency Matrix:
0 1 0 0 0 0
...
Predicted number of nodes: 6
Minimum fuel cost from node 0 to node 5: 3
```


---

## 4. Files & Functions Explained

### `train_model.py`
- Trains a CNN to map images → flattened 6×6 adjacency + node count

### `predict_graph.py`
- Loads saved model
- Predicts adjacency matrix and num_nodes from a `.png` image

### `graph_solver.py`
- Implements movement rules:
  - Traverse → cost 1
  - Reverse signals → cost N
- Computes **minimum fuel cost** from node 0 → N-1

### `main.py`
- for testing

---
