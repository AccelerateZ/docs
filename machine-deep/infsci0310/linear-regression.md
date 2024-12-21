# Linear Regression


## 1. **Introduction to Linear Regression**

- **Definition**: Linear regression is a fundamental approach in data-driven modeling used to predict a numerical target variable based on input features.

- **Example Task**: Predicting house prices based on the size of living areas.

- Types of Regression:
	- Linear (focus of this class)
	- Non-linear (introduced in later courses, e.g., INFSCI 0510)

## 2. **Core Concepts**

### The Model

- Mathematical Representation:
  $$
    f(x) = wx + b
   $$
   where:
   - $f(x)$: Prediction
   - $w$: Weight
   - $b$: Bias

- Predictions are made by adjusting ww and bb to minimize error.

### The Task

- Fit a straight line through data points to accurately predict outcomes.
- Questions to address:
	1. How to define the straight line?
	2. How to adjust ww and bb for accuracy?

## 3. **Loss Functions**

- Purpose: Measure the prediction error.

- Types:

	- Mean Squared Error (MSE): 

        $$
        L = \frac{1}{n} \sum_{i=1}^n (wx_i + b - y_i)^2
        $$

        - Penalizes large errors more than small errors.
	- Mean Absolute Error (MAE):

        $$
        L = \frac{1}{n} \sum_{i=1}^n |wx_i + b - y_i|
        $$

        - Penalizes all errors equally.

- Both have implications for optimization and computational complexity.

## 4. **Optimization: Gradient Descent**

- A method to minimize the loss function.

- Updates weights and bias iteratively:
  $$ 
    w = w - \alpha \frac{\partial L}{\partial w}
  $$
  $$
    b = b - \alpha \frac{\partial L}{\partial b}
  $$
	- $\alpha$: Learning rate

- Gradients depend on the choice of loss function (MSE vs. MAE).

## 5. **Statistical Foundation**

- Assumptions:

	- Data points are **Independent and Identically Distributed (IID)**.
		- Independence: Order of samples doesn't matter.
		- Identically Distributed: All samples come from the same distribution.

- Probabilistic View:

    - Incorporates noise ($\epsilon$) in data: $y_i = wx_i + b + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$
	- Likelihood maximization is equivalent to minimizing MSE.

## 6. **High-Dimensional Linear Regression**

- When features are multi-dimensional

    $$
    \mathbf{X} \in \mathbb{R}^{d \times n}
    $$

    - Prediction: $f(\mathbf{x}_i) = \mathbf{w}^T \mathbf{x}_i + b$
    - Loss: $L = \frac{1}{n} \sum_{i=1}^n (\mathbf{w}^T \mathbf{x}_i + b - y_i)^2$
    - Optimization involves vector calculus.

## 7. **Applications and Further Study**

- Regression is foundational for advanced topics in AI, such as:
	- Supervised Learning (e.g., classification and regression tasks).
	- Probabilistic Modeling and Maximum Likelihood Estimation (MLE).
- Non-linear regression and broader AI applications are discussed in subsequent courses.

This detailed overview provides a framework for understanding and applying linear regression to solve real-world problems.