# The Geometry of Regularization: From L1 to L3

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Portfolio_Project-green)
![Libraries](https://img.shields.io/badge/Libraries-NumPy%20%7C%20SciPy%20%7C%20Sklearn-orange)

## 📖 Overview

In machine learning, **Regularization** is the technique of adding a "penalty" to the model's loss function to prevent overfitting. While **L1 (Lasso)** and **L2 (Ridge)** are industry standards, the mathematical concept generalizes to any **Lp Norm**.

This project explores the behavior of regularization by implementing a custom **L3 Regularizer** (Cubic Penalty) from scratch and comparing it against the standard L1 and L2 methods on synthetic datasets. It visualizes the geometric constraints and the resulting coefficient paths.

## 🧠 The Math: What is "L"?

Regularization changes the learning objective from "minimize error" to "minimize error + keep weights small."

$$J(w) = \text{MSE} + \lambda \cdot ||w||_p$$

Where $||w||_p$ is the p-norm of the weight vector:

$$||w||_p = \left( \sum_{i=1}^{n} |w_i|^p \right)^{1/p}$$

### The Three Contenders

| Method | Norm ($p$) | Formula | Geometric Shape | Key Characteristic |
| :--- | :--- | :--- | :--- | :--- |
| **Lasso** | **L1** | $\sum |w|$ | **Diamond** | **Sparsity:** Forces weak features to exactly 0. Acts as Feature Selection. |
| **Ridge** | **L2** | $\sum w^2$ | **Circle** | **Shrinkage:** Reduces all weights proportionally. Handles correlated features well. |
| **Cubic** | **L3** | $\sum |w|^3$ | **Rounded Square** | **Outlier Suppression:** Penalizes large coefficients extremely heavily, but rarely forces them to exactly 0. |

### Visualizing the Geometry
The "shape" of the regularization determines how the model settles on weights.
* **L1 (Diamond):** The corners touch the axes, allowing coefficients to hit zero.
* **L2 (Circle):** Smooth curvature means coefficients get close to zero but rarely touch.
* **L3 (Super-Gaussian):** As $p$ increases, the shape expands toward a square.



## 🛠️ Implementation Details

### 1. Standard Methods (L1 & L2)
Used `scikit-learn`'s optimized coordinate descent and SVD solvers for Lasso and Ridge regression.

### 2. Custom L3 Implementation
Since L3 is not standard in machine learning libraries, I implemented a custom estimator using `scipy.optimize`.

**The Custom Loss Function:**
```python
def l3_loss(weights, X, y, alpha):
    predictions = X @ weights
    mse = np.mean((y - predictions) ** 2)
    # The L3 Penalty
    penalty = alpha * np.sum(np.abs(weights) ** 3)
    return mse + penalty
