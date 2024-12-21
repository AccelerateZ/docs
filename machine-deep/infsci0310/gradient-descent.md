# Gradient Descent

## 1 Convex and Non-Convex Optimization

Which value of $\theta$ leads to minimum $f(\theta)$?

### 1. 1 Convex Functions

A convex function is a continuous function whose value at the midpoint of every interval in its domain does not exceed the arithmetic mean of its values at the ends of the interval.

> **Definition**
>
>  A function $f$ is **convex** if
> $$
> f[\lambda x_1+(1-\lambda)x_2]\le \lambda f(x_1)+(1-\lambda)f(x_2)
> $$
> for any $\lambda\in[0,1]$.
>
> [Retrieved from Purdue University](https://engineering.purdue.edu/ChanGroup/ECE302/files/Slide_6_03.pdf)

This means the line segment between any two points on the curve of the function lies above or on the curve itself.

For 1D functions, they are convex if
$$
f''(x)\ge0.
$$
And for a convex function $f(\theta)$, its local minimum will be a global minimum.

If we suppose $\theta^*$ is a local minimum, then for any other point $\theta$ in the domain of $f(\theta)$, we have
$$
f(\theta^*)\le f(\theta).
$$

> Note that $\boldsymbol\theta$ can be high-dimensional.

## 2 Argmin and Argmax

> **Definition**
>
> The input to a function that yields the minimum is called the`argmin`, since it is the argument to the function that gives the minimum. 
>
> Similarly, the `argmax` of a function is the input that gives the function's maximum.
>

And the value of $\theta$ that leads to the minimum $f(\theta)$ denotes $\mathop{\arg\min}\limits_{\theta}$. To find it, we need:
$$
\begin{cases}
\dfrac{\mathrm{d}f(\theta)}{\mathrm{d}\theta}=0,\\
\dfrac{\mathrm{d}^2f(\theta)}{\mathrm{d}\theta^2}>0.
\end{cases}
$$

---

But if function is Non-Convex, we should use gradient descent, the golden rule of non-convex  optimization, widely applied in machine learning / deep learning.

## 3 The Golden Method - Gradient Descent

### 3.1 Definition

First, let's recap some basic descent methods. Take the form
$$
x_{k+1}=x_{k}+\alpha_k p_k,\quad k=0,1,\ldots
$$


The gradients are just first-order derivatives $\dfrac{\mathrm{d}f(\theta)}{\mathrm{d}\theta}$. Since we know that given a position $\theta_0$, we can calculate 
$$
\frac{\mathrm{d}f(\theta_0)}{\mathrm{d}\theta_0}=\lim\limits_{\Delta\theta\to 0}\frac{f(\theta_0+\Delta\theta)-\theta_0}{\Delta\theta}.
$$
Hence, gradient descent, namely, following the negative direction of the gradients of the curve, is defined as follows:
$$
\theta_{t+1}\gets\theta_t-\alpha\frac{\mathrm{d}f(\theta_t)}{\mathrm{d}\theta}.
$$
Where, 

- $𝜃_{t+1}$ is the new / updated variable after single step of gradient descent; 
- 𝛼 is the **step size** or the **learning rate** that controls the stride;
- $\Delta\theta=\alpha\dfrac{\mathrm{d}f(\theta_t)}{\mathrm{d}\theta}$ is the resulted change of variable value.

In this case, we are moving towards the negative direction of the gradients. So if t omitted,
$$
\theta\gets\theta-\alpha\frac{\mathrm{d}f(\theta)}{\mathrm{d}\theta}.
$$
Then looking at the function change after changing the variable, i.e. $\Delta f(\theta)$ after $\Delta\theta=-\alpha\dfrac{\mathrm{d}f(\theta)}{\mathrm{d}\theta}$.

Since $\Delta f(\theta)\approx\Delta\theta\frac{\mathrm{d}f(\theta)}{\mathrm{d}\theta}$,  Substitute, and we find $\Delta f(\theta)\le0$.

If result satisfies accuracy $\varepsilon$, then algorithm ends.

::: danger Question 

1. However, the choice of learning rate $\alpha$ is crucial. What if learning rate is too small, or too large?

	> 1. When the Learning Rate $\alpha$ is too large:
	> 	- Overshooting. A large learning rate may push $\theta$ far past the minimum, and the algorithm may fail to converge.
	> 	- Oscillations or Divergence. If learning rate is excessively large, the algorithm may push $\theta$ away from the minimum. This can lead to oscillations or divergence.
	>	- Loss of Critical Information.In a non-convex landscape, the algorithm might skip over local minima or saddle points because of excessively large steps.
	> 2. When the Learning Rate $\alpha$ is too small:
	> 	- Slow Convergence. A small learning rate may lead to slow convergence, as the algorithm takes tiny steps towards the minimum.
	> 	- Stuck in Local Minima. A small learning rate may cause the algorithm to get stuck in local minima, as it is unable to escape the current position.
	> 	- Sensitive to Initial Conditions. A small learning rate may make the algorithm sensitive to initial conditions, as it may get stuck in a local minimum based on the initial position of $\theta$.
	>	- Vanishing Updates. A small learning rate may cause the updates to become vanishingly small, leading to slow convergence or stagnation.

:::

::: danger Question

2. $\Delta f(\theta)\approx\Delta\theta\frac{\mathrm{d}f(\theta)}{\mathrm{d}\theta}\le0$. Why do we use approximation, and under what condition, the approximation can be changed to equality?

	> **Why Use Approximation**
	>
	> We use approximation because in gradient descent, the change in the parameter $\theta$, denoted as $\Delta \theta$, is typically small. For small changes, the change in the objective function $f(\theta)$ can be approximated by the product of the gradient $\frac{\mathrm{d} f(\theta)}{\mathrm{d} \theta}$ and the change in the parameter $\Delta \theta$. This is a first-order Taylor series approximation.
	>
	> **Condition for Approximation to Become Equality**
	>
	> 1. Infinitesimally Small Size. ($\Delta\theta \to 0$) If the change in the parameter $\Delta \theta$ becomes infinitesimally small, the approximation will become an equality. This is because in the limit as $\Delta \theta$ approaches zero, the approximation becomes exact.
	> 2. Smoothness of the Function. The approximation becomes an equality when the function $f(\theta)$ is smooth and differentiable. In this case, the first-order Taylor series approximation becomes exact.
	> 3. Linear Function. If the function $f(\theta)$ is a linear function, the approximation becomes an equality. This is because the gradient of a linear function is constant, and the change in the function is directly proportional to the change in the parameter.

:::
###  3.2 FBGD

FBGD is short for Full-Batch Gradient Descent. It calculates the gradients of the entire dataset to update the parameters.

1. **Process**:
	1. Calculate the gradients of the entire dataset, by averaging the gradients of each data point.
	2. Rewrite the update rule as $\theta\gets\theta-\alpha\frac{1}{N}\sum_{i=1}^{N}\frac{\mathrm{d}f(\theta)}{\mathrm{d}\theta}$.
	3. Update the parameters $\theta$ using the averaged gradients.
	4. Repeat the process until convergence.
	5. The learning rate $\alpha$ is a hyperparameter that needs to be tuned.

2. **True Gradient** is the average of the gradients of each data point:
$$
\frac{1}{N}\sum_{i=1}^{N}\frac{\mathrm{d}f(\theta)}{\mathrm{d}\theta}
$$

1. **Advantages**:
	- **Convergence**: FBGD can converge to the global minimum of the loss function, given the right learning rate and other hyperparameters.
	- **Stability**: FBGD is stable and can provide consistent updates to the parameters.
	- **Optimal Solution**: FBGD can find the optimal solution for convex functions.

2. **Disadvantages**: 
	- **Computational Cost**: FBGD can be computationally expensive, especially for large datasets, as it requires calculating the gradients for all data points.
	- **Memory Usage**: FBGD requires storing the entire dataset in memory to calculate the gradients, which can be memory-intensive.
	- **Slow Convergence**: FBGD may converge slowly for large datasets, as it updates the parameters based on the average gradients of all data points.

### 3.3 SGD

SGD is short for Stochastic Gradient Descent. It calculates the gradients of a single data point to update the parameters.

1. **Process**:
	1. Calculate the gradient of a single data point.
	2. Update the parameters $\theta$ using the gradient of the single data point.
	3. Repeat the process for all data points in the dataset.
	4. The learning rate $\alpha$ is a hyperparameter that needs to be tuned.
   
2. **Advantages**:
	- **Efficiency**: SGD is computationally efficient, as it updates the parameters based on a single data point at a time.
	- **Memory Usage**: SGD requires less memory compared to FBGD, as it only needs to store a single data point at a time.
	- **Fast Convergence**: SGD can converge faster than FBGD for large datasets, as it updates the parameters more frequently.
  
3. **Disadvantages**: 
	- **Convergence**: SGD may not converge to the global minimum, as it updates the parameters based on a single data point at a time.
	- **Stability**: SGD may provide noisy updates to the parameters, as it updates the parameters based on individual data points.
	- **Optimal Solution**: SGD may not find the optimal solution for convex functions, as it updates the parameters based on individual data points.

### 3.3 MBGD

MBGD is short for Mini-Batch Gradient Descent. It calculates the gradients of a mini-batch of data points to update the parameters.

1. **Process**:
	1. Calculate the gradients of a mini-batch of data points.
	2. Update the parameters $\theta$ using the gradients of the mini-batch.
	3. Repeat the process for all mini-batches in the dataset.
	4. The learning rate $\alpha$ is a hyperparameter that needs to be tuned.

2. **Advantages**:
	- **Efficiency**: MBGD is computationally efficient, as it updates the parameters based on a mini-batch of data points.
	- **Memory Usage**: MBGD requires less memory compared to FBGD, as it only needs to store a mini-batch of data points at a time.
	- **Fast Convergence**: MBGD can converge faster than FBGD for large datasets, as it updates the parameters more frequently.

3. **Disadvantages**: 
	 - **Choice of Mini-Batch Size**: The choice of mini-batch size can affect the convergence and stability of MBGD. A small mini-batch size may lead to noisy updates, while a large mini-batch size may slow down convergence. Usually, the mini-batch size is chosen based on empirical results and computational resources, say the power of 2.

## 5 Reference

1. https://engineering.purdue.edu/ChanGroup/ECE302/files/Slide_6_03.pdf
2. https://courses.grainger.illinois.edu/bioe298b/sp2018/Course%20Notes%20(Text)/Chapter07.pdf

