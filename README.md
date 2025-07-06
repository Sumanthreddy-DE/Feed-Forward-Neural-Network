

# Feed-Forward Neural Network

This repository contains my implementation of a Feed-Forward Neural Network as part of the Deep Learning course assignment (DLEX01). The project demonstrates the construction and training of a simple neural network from scratch using Python, without relying on high-level deep learning libraries.

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Project Details](#project-details)
- [References](#references)
- [License](#license)

---

## Overview

The goal of this assignment is to implement the core components of a feed-forward neural network, including layers, activation functions, loss functions, and optimization algorithms. The implementation is modular, making it easy to extend or modify individual components.

---

## File Structure

```
.
├── Base.py                # Base class for layers
├── FullyConnected.py      # Implementation of fully connected (dense) layer
├── Helpers.py             # Utility functions for the network
├── Loss.py                # Loss function implementations (e.g., MSE, Cross-Entropy)
├── NeuralNetwork.py       # Main neural network class
├── NeuralNetworkTests.py  # Unit tests for the neural network
├── Optimizers.py          # Optimizer implementations (e.g., SGD)
├── ReLU.py                # ReLU activation function
├── SoftMax.py             # Softmax activation function
├── Description.pdf        # Assignment description and problem statement
├── submission.zip         # Zipped archive of all files for submission
└── README.md              # This file
```

> For a complete hierarchy and all files, see `submission.zip`.

---

## Requirements

- Python 3.x
- See `requirements.txt` for any additional dependencies (if provided).

No external deep learning libraries (like TensorFlow or PyTorch) are used; only standard Python and NumPy.

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sumanthreddy-DE/Feed-Forward-Neural-Network.git
   cd Feed-Forward-Neural-Network
   ```

2. **(Optional) Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the tests or main script:**
   ```bash
   python NeuralNetworkTests.py
   ```
   or
   ```bash
   python NeuralNetwork.py
   ```

---

## Project Details

- **Layers:** Modular implementation of fully connected layers, ReLU, and Softmax.
- **Loss Functions:** Includes common loss functions for classification and regression.
- **Optimizers:** Basic optimizers such as Stochastic Gradient Descent (SGD).
- **Testing:** Unit tests are provided to verify the correctness of each component.

For a detailed problem statement and assignment requirements, refer to `Description.pdf`.

---

## References

- Deep Learning course materials
- [NumPy documentation](https://numpy.org/doc/)
- Assignment handouts (see `Description.pdf`)

---

## License

This project is for educational purposes as part of a university assignment. Please do not redistribute without permission.

---

If you have any questions, feel free to contact me via GitHub.

---

Let me know if you want to customize any section or add more details!
