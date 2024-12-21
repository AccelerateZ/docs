# Statistical Inference

## 1 Populations and Samples

1. **Populations** are the complete set of all possible observations or measurements that are of interest. Its **size** is the number of elements in the population, denoted by $N$.
   
2. **Samples** are subsets of the population selected by sampling procedures. Its **size** is the number of elements in the sample, denoted by $n$.

3. A sampling procedure is called **biased** if it produces inferences that consistently overestimate or underestimate the population parameter of interest.

## 2 Location Measures

1. **Mean** is the sum of all observations divided by the number of observations. It is denoted by $\bar{x}$. It is a measure of central tendency and **sensitive** to outliers.
   
   - For scalar data $x$: $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$

   - For vector data $\boldsymbol{x}_i\in\mathbb{R}^{d\times1}$: $\bar{\boldsymbol{x}} = \frac{1}{n} \sum_{i=1}^{n} \boldsymbol{x}_i$

2. **Median** is the middle value of a dataset. It is denoted by $M$. It is a measure of central tendency and **robust** to outliers.
   
    - For scalar data $x$: 
    $$
    M = \begin{cases} x_{\left(\frac{n+1}{2}\right)} & \text{if } n \text{ is odd} \\ \frac{1}{2} \left( x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)} \right) & \text{if } n \text{ is even} \end{cases}
    $$

    - For vector data $\boldsymbol{x}_i\in\mathbb{R}^{d\times1}$: We compute the geometric median.
    $$
    M = \arg\min_{\boldsymbol{m}} \sum_{i=1}^{n} \|\boldsymbol{x}_i - \boldsymbol{m}\|_2
    $$

3. **Mode** is the value that appears most frequently in a dataset. It is denoted by $Mo$. It is a measure of central tendency and **robust** to outliers.
   
   - For scalar data $x$: 
   $$
    Mo = \arg\max_{x} \sum_{i=1}^{n} \mathbb{I}(x_i = x)
   $$

   - For vector data $\boldsymbol{x}_i\in\mathbb{R}^{d\times1}$: We compute the mode corresponds to the point where all data points, then combine them to form a new mode vector.

   - If samples are less repetitive, we group them into bins and compute the mode of each bin.

4. **Variance** and **Standard Deviation** are measures of dispersion. They quantify the spread of the data around the mean. They are denoted by $s^2$ and $s$ (For sample) and $\sigma^2$ and $\sigma$ (For population).

   - For scalar data $x$: 
   $$
    s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
   $$

   $$
    s = \sqrt{s^2}
   $$

   - For vector data $\boldsymbol{x}_i\in\mathbb{R}^{d\times1}$: 
   $$
    s^2 = \frac{1}{n-1} \sum_{i=1}^{n} \|\boldsymbol{x}_i - \bar{\boldsymbol{x}}\|_2^2
   $$

   $$
   s = \sqrt{s^2}
   $$

   - For population variance and standard deviation, replace $n-1$ by $n$.
   
   	::: danger Why?
   
   	The denominator is $n-1$ instead of $n$ because the sample variance is an **unbiased estimator** of the population variance. The unbiased estimator is the one that gives the correct answer on average over many samples. The sample variance is an unbiased estimator because the expected value of the sample variance is equal to the population variance.
   
   	:::

5. **Boxplot** is a graphical representation of the data based on the five-number summary: minimum, first quartile, median, third quartile, and maximum. It is useful for detecting outliers and comparing distributions.
   1. **Minimum**: The smallest value in the dataset.
   2. **First Quartile (Q1)**: The median of the lower half of the dataset.
   3. **Median (Q2)**: The middle value of the dataset.
   4. **Third Quartile (Q3)**: The median of the upper half of the dataset.
   5. **Maximum**: The largest value in the dataset.
   6. **Interquartile Range (IQR)**: The range between the first and third quartiles, $IQR = Q3 - Q1$.
   7. **Outliers**: Values that fall below $Q1 - 1.5 \times IQR$ or above $Q3 + 1.5 \times IQR$.
   8. **Whiskers**: The lines extending from the box to the minimum and maximum values, excluding outliers.
   
## 3 Correlation Measures

1. **Pearson Correlation Coefficient** measures the linear relationship between two variables. It ranges from -1 to 1, where:
   - 1 indicates a perfect positive linear relationship.
   - -1 indicates a perfect negative linear relationship.
   - 0 indicates no linear relationship.
   
   $$
   \rho = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
   $$
   $$
   \rho = \frac{\text{Cov}(x, y)}{s_x s_y}
   $$

2. **Multi-correlation Coefficient** measures the linear relationship between multiple variables. It ranges from 0 to 1, where:
    - 1 indicates a perfect linear relationship.
    - 0 indicates no linear relationship.

   $$
   R^2 = \frac{\sqrt{\sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2}}{\sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
   $$

3. **Covariance** measures the relationship between two variables. It is the Pearson correlation coefficient multiplied by the standard deviations of the variables.
   -  If $\text{Cov}(x, y) > 0$, the variables are positively correlated. 
   -  If $\text{Cov}(x, y) < 0$, the variables are negatively correlated.

   $$
   \text{Cov}(x, y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
   $$

   ::: warning Note
   If given two variables $\boldsymbol{A}\in\mathbb{R}^{1\times n}$ and $\boldsymbol{B}\in\mathbb{R}^{1\times n}$, the covariance matrix is given by:
   $$
    \text{Cov}(\boldsymbol{A}, \boldsymbol{B}) = \frac{1}{n-1} \sum_{i=1}^{n} (\boldsymbol{A}_i - \bar{\boldsymbol{A}})(\boldsymbol{B}_i - \bar{\boldsymbol{B}})
   $$
   :::

4. **Covariance Matrix**: The covariance matrix is a square matrix that contains the covariance between each pair of variables. It is symmetric and positive semi-definite.
   - The diagonal elements are the variances of the variables.
   - The off-diagonal elements are the covariances between the variables.

   $$
   \text{Cov}(\boldsymbol{X}) = \begin{bmatrix} \text{Cov}(X_1, X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_d) \\ \text{Cov}(X_2, X_1) & \text{Cov}(X_2, X_2) & \cdots & \text{Cov}(X_2, X_d) \\ \vdots & \vdots & \ddots & \vdots \\ \text{Cov}(X_d, X_1) & \text{Cov}(X_d, X_2) & \cdots & \text{Cov}(X_d, X_d) \end{bmatrix}
   $$

   :::warning Note
   Let $\boldsymbol{x_1},\boldsymbol{x_2},\ldots,\boldsymbol{x_n}\in\mathbb{R}^{d\times1}$, (sample), then, its covariance matrix is given by:
   $$
    \Sigma = \frac{1}{n-1} \sum_{i=1}^{n} (\boldsymbol{x}_i - \bar{\boldsymbol{x}})(\boldsymbol{x}_i - \bar{\boldsymbol{x}})^T\in\mathbb{R}^{d\times d}.
   $$
   ::: danger Question
   1. How to understand the covariance matrix?
   
   2. Linear relationship means direction + strength. dose the covaiance matrix here indicates these info?

     > 1. The **covariance matrix** $\Sigma\in\mathbb{R}^{d\times d}$ represents the relationships between the components of the vectors. For diagonal elements, they represent the variance of each component. For off-diagonal elements, they represent the covariance between the components. The covariance matrix is symmetric and positive semi-definite. If its value is positive, it means the two components are positively correlated. If its value is negative, it means the two components are negatively correlated. If its value is zero, it means the two components are nearly independent.
     > 2. A **linear relationship** consists of **direction** and **strength**.
     >   1. Strength of Linear Relationship: Covariance values indicate the degree to which two dimensions are linearly related. However, these values are not normalized, so interpreting the "strength" directly can be difficult. Instead, the correlation coefficient (a normalized version of covariance) is often used to quantify strength. For $\Sigma_{ij}$, the closer its magnitude to $\sqrt{\Sigma_{ii}\Sigma_{jj}}$, the stronger the linear relationship.
     >   2. **Direction of Linear Relationship**: The same to above:
     >
     >   - If $\text{Cov}(x, y) > 0$, the variables are positively correlated. 
     >   - If $\text{Cov}(x, y) < 0$, the variables are negatively correlated.
     >
   
   :::
   
   ::: danger Question
   If we suppose the dataset $\boldsymbol{X}=[\boldsymbol{A}_1, \boldsymbol{A}_2, \ldots, \boldsymbol{A}_d]^T\in\mathbb{R}^{d\times n}$, then, what is the dimension of the covariance matrix $\Sigma$?
   $$
   \Sigma=\begin{bmatrix}Var(\boldsymbol{A}_1, \boldsymbol{A}_1) & Cov(\boldsymbol{A}_1, \boldsymbol{A}_2) & \cdots & Cov(\boldsymbol{A}_1, \boldsymbol{A}_d) \\ Cov(\boldsymbol{A}_2, \boldsymbol{A}_1) & Var(\boldsymbol{A}_2, \boldsymbol{A}_2) & \cdots & Cov(\boldsymbol{A}_2, \boldsymbol{A}_d) \\ \vdots & \vdots & \ddots & \vdots \\ Cov(\boldsymbol{A}_d, \boldsymbol{A}_1) & Cov(\boldsymbol{A}_d, \boldsymbol{A}_2) & \cdots & Var(\boldsymbol{A}_d, \boldsymbol{A}_d) \end{bmatrix}
   $$
   > The covariance matrix $\Sigma\in\mathbb{R}^{d\times d}$.
   
   :::

## 4 Shape Measures

1. **Skewness** measures the asymmetry of the data distributions. It is denoted by $\gamma$.
   - If $\gamma > 0$, the distribution is right-skewed.
   - If $\gamma < 0$, the distribution is left-skewed.
   - If $\gamma = 0$, the distribution is symmetric.
   - If $|\gamma|$ is small, it has a mild skewness.
   - If $|\gamma|$ is large, it has a severe skewness.
   
   $$
   \gamma = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^3}{s^3}
   $$
   
   > From Wikipedia:
   >
   > In probability theory and statistics, skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. The skewness value can be positive, zero, negative, or undefined.
   > $$
   > \gamma_{1}:=\tilde{\mu}_{3}=\mathrm{E}\left[\left(\frac{X-\mu}{\sigma}\right)^{3}\right]=\frac{\mu_{3}}{\sigma^{3}}=\frac{\mathrm{E}\left[(X-\mu)^{3}\right]}{\left(\mathrm{E}\left[(X-\mu)^{2}\right]\right)^{3 / 2}}=\frac{\kappa_{3}}{\kappa_{2}^{3 / 2}}
   > $$
   > 

2. **Kurtosis** measures the tailedness of the data distributions. It is denoted by $\kappa$.
   - If $\kappa > 3$, the distribution has heavier tails than the normal distribution (Leptokurtic).
   - If $\kappa < 3$, the distribution has lighter tails than the normal distribution (Platykurtic).
   - If $\kappa = 3$, the distribution has tails similar to the normal distribution (Mesokurtic).
   
   $$
   \kappa = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^4}{s^4}
   $$
   
   > From Wikipedia:
   >
   > In probability theory and statistics, kurtosis (from Greek: κυρτός, kyrtos or kurtos, meaning "curved, arching") refers to the degree of “tailedness” in the probability distribution of a real-valued random variable. Similar to skewness, kurtosis provides insight into specific characteristics of a distribution. Various methods exist for quantifying kurtosis in theoretical distributions, and corresponding techniques allow estimation based on sample data from a population. It’s important to note that different measures of kurtosis can yield varying interpretations.
   > $$
   > \kappa_{2}:=\tilde{\mu}_{4}=\mathrm{E}\left[\left(\frac{X-\mu}{\sigma}\right)^{4}\right]=\frac{\mu_{4}}{\sigma^{4}}=\frac{\mathrm{E}\left[(X-\mu)^{4}\right]}{\left(\mathrm{E}\left[(X-\mu)^{2}\right]\right)^{2}}=\frac{\kappa_{4}}{\kappa_{2}^{2}}
   > $$