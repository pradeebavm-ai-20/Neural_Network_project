# Student Pass/Fail Classification — Neural Network from Scratch (NumPy)

A binary classification neural network built **entirely from scratch using NumPy** — no TensorFlow, PyTorch, Keras, or scikit-learn. Every core operation (matrix multiplication, activations, loss, backpropagation, gradient descent) is implemented manually to demonstrate a clear, ground-up understanding of how neural networks actually work.

##  Project Overview

The model predicts whether a student will **Pass (1)** or **Fail (0)** based on four features:

- Study Hours
- Attendance (%)
- Previous Marks
- Assignment Score

Since no real dataset was available, a **synthetic dataset of 500 student records** is generated with a realistic feature–outcome relationship plus random noise, so the classification task is meaningful but not trivially deterministic.

##  Architecture

```
Input Layer (4 neurons)
        │
        ▼
Hidden Layer (5 neurons) — ReLU activation
        │
        ▼
Output Layer (1 neuron) — Sigmoid activation
```

| Layer | Weights Shape | Bias Shape |
|---|---|---|
| Input → Hidden | (4, 5) | (1, 5) |
| Hidden → Output | (5, 1) | (1, 1) |

##  Features Implemented From Scratch

- ✅ Synthetic dataset generation (NumPy + Pandas)
- ✅ Exploratory Data Analysis (distribution, boxplots, correlation heatmap)
- ✅ Manual train/test split (80/20, fixed seed, no sklearn)
- ✅ Z-score standardization using **training statistics only** (no data leakage)
- ✅ Custom matrix multiplication (no `np.dot` / `np.matmul`)
- ✅ ReLU and Sigmoid activations + their derivatives
- ✅ Forward propagation
- ✅ Binary Cross-Entropy loss (with epsilon clipping)
- ✅ Backpropagation via the chain rule
- ✅ Gradient descent parameter updates
- ✅ Manual evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix
- ✅ `predict_student()` function for real-time new-student predictions

##  Results

**Training:** 2000 epochs, learning rate = 0.05

| Epoch | Loss |
|---|---|
| 1 | 0.681 |
| 200 | 0.478 |
| 1000 | 0.449 |
| 2000 | 0.446 |

**Test Set Performance:**

| Metric | Score |
|---|---|
| Accuracy | 0.78 |
| Precision | 0.7846 |
| Recall | 0.8644 |
| F1-score | 0.8226 |

**Confusion Matrix**

|  | Predicted Fail | Predicted Pass |
|---|---|---|
| **Actual Fail** | 27 (TN) | 14 (FP) |
| **Actual Pass** | 8 (FN) | 51 (TP) |

**Sample New-Student Predictions**

| Study Hrs | Attendance | Prev. Marks | Assignment | Probability | Prediction |
|---|---|---|---|---|---|
| 8.0 | 90 | 75 | 85 | 0.9648 | Pass |
| 2.0 | 55 | 40 | 35 | 0.0324 | Fail |

##  Project Structure

```
student-pass-fail-nn/
├── student_pass_fail_nn.py     # Full implementation (dataset → training → evaluation)
├── EXPLANATION.md               # Concept-by-concept breakdown (what/why/formula/code)
├── outputs/                      # Generated plots (created automatically on run)
│   ├── eda_class_distribution.png
│   ├── eda_feature_boxplots.png
│   ├── eda_correlation_heatmap.png
│   └── training_loss.png
└── README.md
```

##  How to Run

```bash
git clone https://github.com/<your-username>/student-pass-fail-nn.git
cd student-pass-fail-nn
pip install numpy pandas matplotlib
python student_pass_fail_nn.py
```

Or open `student_pass_fail_nn.py` in Google Colab / Jupyter and run all cells. All plots are saved automatically into an `outputs/` folder.

##  Tech Stack

- **Python 3**
- **NumPy** — all mathematical/neural network operations
- **Pandas** — dataset creation and handling
- **Matplotlib** — visualizations

##  Learning Objectives

This project was built to understand, at a fundamental level:

- How forward and backward propagation work mathematically
- How the chain rule drives gradient computation in backpropagation
- Why data leakage matters and how to prevent it
- How activation functions and loss functions shape learning
- How to evaluate a classifier honestly using manually computed metrics

##  License

This project is open-sourced for educational purposes. Feel free to use, modify, and learn from it.
