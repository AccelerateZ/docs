# Data Analysis - Supervised Learning

## PCA

1. 使用SVD分解和使用协方差矩阵进行计算的一致性。

	> In the classical implementation of PCA, we explicitly compute the covariance matrix of the data. However, in this task, PCA method is based on SVD, which avoids the need to explicitly compute the covariance matrix.
	>
	> **Proof:**
	>
	> $\boldsymbol{X}\in\mathbb{R}^{d\times n}$ is the data matrix, where $d$ is the number of features and $n$ is the number of samples. The covariance matrix $\boldsymbol{C}$ can be computed as follows:
	>
	> $$
	> \boldsymbol{C}=\frac{1}{n}\boldsymbol{X}\boldsymbol{X}^{T}
	> $$
	>
	> We perform the singular value decomposition of $\boldsymbol{X}$:
	>
	> $$
	> \boldsymbol{X}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{T}
	> $$
	>
	> Where $\boldsymbol{U}\in\mathbb{R}^{d\times d}$ is the left singular vector matrix, $\boldsymbol{\Sigma}\in\mathbb{R}^{d\times n}$ is the diagonal matrix of singular values, and $\boldsymbol{V}\in\mathbb{R}^{n\times n}$ is the right singular vector matrix.
	>
	> Derivation:
	>
	> Expand C using the SVD of X:
	>
	> $$
	> \boldsymbol{C}=\frac{1}{n}\boldsymbol{X}\boldsymbol{X}^{T}=\frac{1}{n}\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{T}\boldsymbol{V}\boldsymbol{\Sigma}^{T}\boldsymbol{U}^{T} = \frac{1}{n}\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{T}\boldsymbol{U}^{T}
	> $$
	>
	> where we used the fact that $\boldsymbol{V}^{T}\boldsymbol{V}=\boldsymbol{I}$, the identity matrix.
	>
	> Now, observe that:
	>
	> - $\boldsymbol{\Sigma}\boldsymbol{\Sigma}^T$ is a diagonal matrix with the squares of the singular values on the diagonal.
	> - Therefore, $\boldsymbol{C}$ can be expressed as:
	>
	> $$
	> \boldsymbol{C}=\boldsymbol{U}\left(\frac{1}{n}\boldsymbol{\Sigma}^2\right)\boldsymbol{U}^{T}
	> $$
	>
	> where,
	>
	> - The columns of U are the eigenvectors of C
	> - The eigenvalues are the scaled squared singular values $\frac{\sigma_i^2}{n}$, which correspond to the varience captured along each principal component.
	>
	> Therefore, SVD-based PCA is mathematically equivalent to the classical PCA approach via covariance matrix eigen-decomposition.
	>
	> > Note: $C$ should be 
	> > $$
	> > \boldsymbol{C} = \frac{1}{n - 1}\boldsymbol{X}\boldsymbol{X}^T.
	> > $$
	> > Hereafter, we will use the covariance matrix as $\boldsymbol{C} = \frac{1}{n - 1}\boldsymbol{X}\boldsymbol{X}^T$ for the PCA implementation.

2. Kernel PCA的详细推导过程。

	Kernel PCA is a combined technique of PCA and the kernel trick, where we are still interested in using the PCA process to find the features $\boldsymbol{Z}\in\mathbb{R}^{k\times n}$. However, such a transformation from $\boldsymbol{X}\in\mathbb{R}^{d\times n}$ to $\boldsymbol{Z}\in\mathbb{R}^{k\times n}$ now becomes non-linear, as a non-linear kernel function can be applied to first transformed $\boldsymbol{X}\in\mathbb{R}^{d\times n}$ to $\phi(\boldsymbol{X})\in\mathbb{R}^{D\times n}$ in a superspace with $D>d$, then, the linear PCA is performed to transform $\phi(\boldsymbol{X})\in\mathbb{R}^{D\times n}$ to $\boldsymbol{Z}\in\mathbb{R}^{k\times n}$. This kernel PCA process brings a major advantage:

	- Since the calculation of $\boldsymbol{Z}$ can be non-linear, and the dimensionality of $\boldsymbol{Z}$ is now $k\in [1, D)$ with $D>d$, these characteristics allow us to search for solutions in a new space (not limited by the original dimentionality $d$), and such solutions may be linear.

	For example, with kernel PCA, for a linearly-inseparable dataset $\boldsymbol{X}\in\mathbb{R}^{d\times n}$ with a low dimensionality, e.g., d = 2, now it may be possible to solve such classification task with linear solutions, while in a new space.

	However, we would like to avoid the explicit computation of the high-dimensional $\phi(\boldsymbol{X})$ for the PCA decomposition, which can

	be done by involving the kernel function $K(\boldsymbol{x}_{i}, \boldsymbol{x}_{j})=<\phi(\boldsymbol{x}_{i}),\phi(\boldsymbol{x}_{j})>$ with the plain PCA, creating the kernel PCA solution. Two different kernel function will be explored here:

	1. Homogeneous Polynomial kernel: $K(\boldsymbol{x}_{i}, \boldsymbol{x}_{j})=(<\boldsymbol{x}_{i}, \boldsymbol{x}_{j}>)^{p}$, where $p>0$ is the polynomial degree.
	2. Radial Basis Function (RBF) kernel: $K(\boldsymbol{x}_{i}, \boldsymbol{x}_{j})=e^{-\gamma ||\boldsymbol{x}_{i}-\boldsymbol{x}_{j}||_{2}^{2}}$, where $\gamma=\frac{1}{2\sigma^{2}}$ and $\sigma$ is the width or scale of a Gaussian distribution centered at $\boldsymbol{x}_{j}$.

	**Mathematically prove howwe can compute the PC Martix?**

	> First things first, we denote:
	>
	> - $\mathbf{X}\in\mathbb{R}^{d \times N}$ is the input matrix, where $d$ and $N$ are the number of the features and samples, respectively.
	> - $\mathbf{x}_i\in\mathbb{R}^{d}$ is the $i$-th column vector for $\mathbf{X}$. Therefore, $\mathbf{X} = \left[\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\right]$.
	> - $\phi(\cdot)$ is a nonlinear transformation. $\phi(\cdot): \mathbb{R}^d \to \mathcal{F}$.
	> - $\phi(\mathbf{X})\in\mathbb{R}^{D\times N}$ is the mapped matrix on a higher or infinity dimensional eigenspeace $\mathcal{F}$.
	> - $\phi(\mathbf{x}_i)\in\mathbb{R}^{D}$ is the $i$-th column vector for $\phi(\mathbf{X})$. Therefore, $\phi(\mathbf{X}) = \left[\phi(\mathbf{x}_1), \phi(\mathbf{x}_2), \ldots, \phi(\mathbf{x}_N)\right]$.
	> - $\mathbf{K}\in\mathbb{R}^{N\times N}$ is the Gram matrix, whose element $k_{ij}$ is given by the kernel function $K(\boldsymbol{x}_{i}, \boldsymbol{x}_{j})=<\phi(\boldsymbol{x}_{i}),\phi(\boldsymbol{x}_{j})>$
	>
	> The first thing is to center the mapped matrix $\phi(\mathbf{X})$ in the feature space $\mathcal{F}$, which is defined as follows:
	> $$
	> \begin{aligned}
	> \tilde\phi(\mathbf{x}_i) &=\phi(\mathbf{x}_i)-\frac{1}{N}\sum_{j=1}^{N}\phi(\mathbf{x}_{j}) \\
	> &=\phi(\mathbf{x}_i)-\frac{1}{N}\phi(\mathbf{X})\mathbf{1}_N \\
	> \end{aligned}
	> $$
	> where $\mathbf{1}_N\in\mathbb{R}^N$ is the vector of ones.
	>
	> Then, centered mapped matrix $\tilde\phi(\mathbf{X})$ can be denoted as:
	> $$
	> \begin{aligned}
	> \tilde\phi(\mathbf{X}) &= \left[\tilde\phi(\mathbf{x}_1), \tilde\phi(\mathbf{x}_2), \ldots, \tilde\phi(\mathbf{x}_N)\right] \\
	> &=\left[\phi(\mathbf{x}_1)-\frac{1}{N}\phi(\mathbf{X})\mathbf{1}_N, \phi(\mathbf{x}_2)-\frac{1}{N}\phi(\mathbf{X})\mathbf{1}_N, \ldots, \phi(\mathbf{x}_N)-\frac{1}{N}\phi(\mathbf{X})\mathbf{1}_N\right] \\
	> &=\phi(\mathbf{X})-\frac{1}{N}\phi(\mathbf{X})\mathbf{1}_N\mathbf{1}_N^T.
	> \end{aligned}
	> $$
	> Similar to the Linear PCA, we have (SVD)
	> $$
	> \tilde\phi(\mathbf{X}) = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T,
	> $$
	> where:
	>
	> - $\mathbf{U}\in\mathbb{R}^{D\times D}$ is the left singular vector (orthonormal) matrix, whose column vectors are eigenvectors of $\tilde\phi(\mathbf{X})\tilde\phi(\mathbf{X})^T$,
	> - $\mathbf{\Sigma}\in\mathbb{R}^{D\times N}$ is the diagonal matrix of singular values, whose elements are ordered from largest to smallest, i.e., $\sigma_1\geq\sigma_2\geq\ldots\geq\sigma_D$, and
	> - $\mathbf{V}\in\mathbb{R}^{N\times N}$ is the right singular vector (orthonormal) matrix, whose column vectors are eigenvectors of $\tilde\phi(\mathbf{X})^T\tilde\phi(\mathbf{X})$.
	>
	> Notice that the covariance matrix $\mathbf{C}$ of $\tilde\phi(\mathbf{X})$ and the Gram matrix $\mathbf{K}$ of $\mathbf{\tilde X}$ are denoted as:
	> $$
	> \mathbf{C} = \frac{1}{n-1}\sum_{i=1}^n\left(\tilde\phi(\mathbf{x}_i)\tilde\phi(\mathbf{x}_i)^T\right) = \frac{1}{n-1}\tilde\phi(\mathbf{X})\tilde\phi(\mathbf{X})^T,
	> $$
	>
	> $$
	> \begin{aligned}
	> \mathbf{K} &= \begin{bmatrix} <\tilde\phi(\boldsymbol{x}_{1}),\tilde\phi(\boldsymbol{x}_{1})> & <\tilde\phi(\boldsymbol{x}_{1}),\tilde\phi(\boldsymbol{x}_{2})> & \cdots & <\tilde\phi(\boldsymbol{x}_{1}),\tilde\phi(\boldsymbol{x}_{N})> \\ <\tilde\phi(\boldsymbol{x}_{2}),\tilde\phi(\boldsymbol{x}_{1})> & <\tilde\phi(\boldsymbol{x}_{2}),\tilde\phi(\boldsymbol{x}_{2})> & \cdots & <\tilde\phi(\boldsymbol{x}_{2}),\tilde\phi(\boldsymbol{x}_{N})> \\ \vdots & \vdots & \ddots & \vdots \\ <\tilde\phi(\boldsymbol{x}_{N}),\tilde\phi(\boldsymbol{x}_{1})> & <\tilde\phi(\boldsymbol{x}_{N}),\tilde\phi(\boldsymbol{x}_{2})> & \cdots & <\tilde\phi(\boldsymbol{x}_{N}),\tilde\phi(\boldsymbol{x}_{N})>\end{bmatrix}\\
	&= \tilde\phi(\mathbf{X})^T\tilde\phi(\mathbf{X}).
	> \end{aligned}
	> $$
	>
	> and both $\mathbf{C}$ and $\mathbf{K}$ are symmetric matrices.
	>
	> Therefore, it is clear that:
	> $$
	> \mathbf{C} = \frac{1}{n-1}\mathbf{U}\mathbf{\Sigma}\mathbf{\Sigma}^T\mathbf{U}^T = \frac{1}{n-1}\mathbf{U}\mathbf{\Sigma}^2\mathbf{U}^T,
	> $$
	> and
	> $$
	> \mathbf{K} = \mathbf{V}\mathbf{\Sigma}\mathbf{\Sigma}^T\mathbf{V}^T = \mathbf{V}\mathbf{\Sigma}^2\mathbf{V}^T.
	> $$
	>
	> If $\phi(\mathbf{X})$ is centered, then $\mathbf{C}$ and $\mathbf{K}$ are both centered. Therefore, it is no need to explicitly  centered $\mathbf{X}$ or $\phi(\mathbf{X})$. The centered Gram matrix $\mathbf{\tilde K}$ can be computed as follows:
	> $$
	> \begin{aligned}
	> \mathbf{\tilde K} &= \tilde\phi(\mathbf{X})^T\tilde\phi(\mathbf{X}) = \left(\phi(\mathbf{X})-\frac{1}{N}\phi(\mathbf{X})\mathbf{1}_N\mathbf{1}_N^T\right)^T\left(\phi(\mathbf{X})-\frac{1}{N}\phi(\mathbf{X})\mathbf{1}_N\mathbf{1}_N^T\right) \\
	> &= \phi(\mathbf{X})^T\phi(\mathbf{X}) - \frac{1}{N}\phi(\mathbf{X})^T\phi(\mathbf{X})\mathbf{1}_N\mathbf{1}_N^T - \frac{1}{N}\mathbf{1}_N^T\mathbf{1}_N\phi(\mathbf{X})^T\phi(\mathbf{X}) + \frac{1}{N^2}\mathbf{1}_N^T\mathbf{1}_N\phi(\mathbf{X})^T\phi(\mathbf{X})\mathbf{1}_N\mathbf{1}_N^T \\
	> & = \mathbf{K} - \frac{1}{N}\mathbf{K}\mathbf{1}_N\mathbf{1}_N^T - \frac{1}{N}\mathbf{1}_N^T\mathbf{1}_N\mathbf{K} + \frac{1}{N^2}\mathbf{1}_N^T\mathbf{1}_N\mathbf{K}\mathbf{1}_N\mathbf{1}_N^T \\
	> \end{aligned}
	> $$
	> If we denote $\mathbf{N} = \frac{1}{n}\mathbf{1}_N\mathbf{1}_N^T$, which is a $N \times N$ matrix, all the elements of $\mathbf{N}$ are equal to $\frac{1}{n}$, then, we have:
	> $$
	> \mathbf{\tilde K} = \mathbf{K} - \mathbf{N}\mathbf{K} - \mathbf{K}\mathbf{N} + \mathbf{N}^T\mathbf{K}\mathbf{N}
	> $$
	>
	> From now, we use $\mathbf{K}$ to refer centere Gram matrix $\mathbf{\tilde K}$.
	>
	> We then still follow the method of finding the first principal component. We know that the PCs are the eigenvectors of $\mathbf{C}$. Notice that the column vectors of $\mathbf{U}$ are eigenvectors of $\mathbf{C}$, therefore, we have:
	>
	> $$
	> \mathbf{C}\mathbf{u}_i = \lambda_{i}\mathbf{u}_i.
	> $$
	>
	> Since the eigenvectors $\mathbf{u}_i$ is a linear combination of $\tilde\phi(\mathbf{X})$, which is
	>
	> $$
	> \mathbf{u}_i = \sum_{i = 1}^{n} \alpha_i\tilde\phi(\mathbf{x}_i) = \tilde\phi(\mathbf{X})\mathbf{\vec \alpha},
	> $$
	>
	> where the $\vec \alpha_i$ is the linear combination factor. We may contrict the mangitude of $\vec \alpha$ is equal to 1, i.e., $\left \| \vec \alpha\right \| = 1$.
	>
	> Therefore,
	>
	> 
	> $$
	> \begin{aligned}
	> \mathbf{C}\mathbf{u}_i &= \lambda_{i}\mathbf{u}_i\\
	> \frac{1}{n-1}\tilde\phi(\mathbf{X})\tilde\phi(\mathbf{X})^T\tilde\phi(\mathbf{X})\vec\alpha_i &= \lambda_{i}\tilde\phi(\mathbf{X})\vec\alpha_i\\
	> \tilde\phi(\mathbf{X})\mathbf{K}\vec \alpha_i &= (n - 1)\lambda_{i}\tilde\phi(\mathbf{X})\vec\alpha_i\\
	> \mathbf{K}\vec \alpha_i &= (n - 1)\lambda_{i}\vec\alpha_i,
	> \end{aligned}
	> $$
	> which means that the eigenvalues of $\mathbf{K}$ are $(n - 1) \lambda_{i}$, and the eigenvectors of $\mathbf{K}$ are $\vec \alpha_i$.
	>
	> Therefore, from the eigen value decomposition, we can derive that
	> $$
	> \mathbf{K} = \mathbf{V}\mathbf{\Sigma}^2\mathbf{V}^T \Rightarrow \mathbf{K}\mathbf{V} = \mathbf{\Sigma}^2\mathbf{V}.
	> $$
	> thus, we know that
	> $$
	> \mathbf{V} = \left[\mathbf{\alpha_1}, \mathbf{\alpha_2}, \cdots, \mathbf{\alpha_n}\right],
	> $$
	> which is $\mathbf{v}_i = \vec{\alpha}_i$, and
	> $$
	> \mathbf{\Sigma}^2 = (n - 1)\mathbf{\Lambda},
	> $$
	> which is $\sigma_i = \sqrt{(n - 1)\lambda_i}$. Therefore, after normalize $\mathbf{u}_i$ to obtain a unit vector, we have:
	> $$
	> \begin{aligned}
	> \mathbf{u}_i &= \frac{\tilde\phi(\mathbf{X}) \mathbf{v}_i}{||\tilde\phi(\mathbf{X}) \mathbf{v}_i||} \\
	> &= \frac{\tilde\phi(\mathbf{X}) \mathbf{v}_i}{\sqrt{\mathbf{v}_i^T\tilde\phi(\mathbf{X})^T\tilde\phi(\mathbf{X})\mathbf{v}_i} } \\
	> &= \frac{\tilde\phi(\mathbf{X}) \mathbf{v}_i}{\sqrt{\mathbf{v}_i^T\mathbf{K}\mathbf{v}_i} } \\
	> &= \frac{\tilde\phi(\mathbf{X}) \mathbf{v}_i}{\sqrt{\sigma_i^2} } = \frac{\tilde\phi(\mathbf{X}) \mathbf{v}_i}{\sigma_i}.
	> \end{aligned}
	> $$
	> Therefore, the final Principal Components are given by:
	> $$
	> \mathbf{U} = \left[\frac{\tilde\phi(\mathbf{X}) \mathbf{v}_1}{\sigma_1}, \frac{\tilde\phi(\mathbf{X}) \mathbf{v}_2}{\sigma_2}, \cdots, \frac{\tilde\phi(\mathbf{X}) \mathbf{v}_k}{\sigma_k}\right],
	> $$
	> where $k$ is the number of the PCs.

	**Mathematically prove how to compute the transformed dataset.**

	> From the subtask 1, we obtain the PC matrix $\mathbf{U}$. We can then compute the transformed dataset $\mathbf{Z}$ as follows:
	> $$
	> \mathbf{Z} = \mathbf{U}^T\tilde\phi(\mathbf{X}) = \left[\mathbf{u}_1, \mathbf{u}_2, \cdots, \mathbf{u}_k\right]^T\tilde\phi(\mathbf{X}).
	> $$
	> Expanding $\mathbf{Z}$, we have:
	> $$
	> \mathbf{Z} = 
	> 
	> \begin{bmatrix}
	> 
	>  \mathbf{u}_1^T\tilde\phi(\mathbf{x}_1) & \mathbf{u}_1^T\tilde\phi(\mathbf{x}_1) & \cdots & \mathbf{u}_1^T\tilde\phi(\mathbf{x}_n)\\
	> 
	>  \mathbf{u}_2^T\tilde\phi(\mathbf{x}_1) & \mathbf{u}_2^T\tilde\phi(\mathbf{x}_2) & \cdots & \mathbf{u}_2^T\tilde\phi(\mathbf{x}_n)\\
	> 
	>  \vdots & \vdots & \ddots & \vdots\\
	> 
	>  \mathbf{u}_k^T\tilde\phi(\mathbf{x}_1) & \mathbf{u}_k^T\tilde\phi(\mathbf{x}_2) & \cdots & \mathbf{u}_k^T\tilde\phi(\mathbf{x}_n)
	> 
	> \end{bmatrix}.
	> $$
	> For $i, j$ in $\mathbf{u}_i^T\tilde\phi(\mathbf{x}_j)$, where $i\in[1, k]$, and $j\in[1, n]$, we have
	> $$
	> \begin{aligned}
	> \mathbf{u}_i^T\tilde\phi(\mathbf{x}_j) &= \left[\frac{\tilde\phi(\mathbf{X}) \mathbf{v}_i}{\sigma_i}\right]^T\tilde\phi(\mathbf{x}_j) = \frac{1}{\sigma_i}\mathbf{v}_i^T\tilde\phi(\mathbf{X})^T\tilde\phi(\mathbf{x}_j)\\
	> 
	> &= \frac{1}{\sigma_i}\mathbf{v}_i^T\mathbf{k}_{\_, j}\\
	> 
	> &= \frac{1}{\sqrt{(n - 1)\lambda_i} }\mathbf{v}_i^T\mathbf{k}_{\_, j}.
	> \end{aligned}
	> $$
	> where $\mathbf{k}_{\_, j}$ is the $j$-th column of the Gram matrix $\mathbf{K}$.
	>
	> Therefore,
	> $$
	> \mathbf{Z} = \frac{1}{\sqrt{n-1}}\text{Diag}(\frac{1}{\sqrt{\lambda_1}}, \frac{1}{\lambda_2}, \cdots, \frac{1}{\sqrt{\lambda_k}})\mathbf{V}_{k}^T\mathbf{K} = \frac{1}{\sqrt{n-1} }\mathbf{\Lambda}_{k}^{-\frac{1}{2}}\mathbf{V}_{k}^T\mathbf{K}.
	> $$
	> where,
	>
	> \- $\mathbf{\Lambda}$ is the diagonal matrix, comprised of $\lambda_i$, the eigenvalues of $\mathbf{K}$.
	>
	> \- $\mathbf{\Lambda}_k^{-\frac{1}{2}}$ is the $k \times k$ diagonal matrix, comprised of $\frac{1}{\sqrt{\lambda_i}}$, the inverse of the square root of $\lambda_i$, where it refers to the top k eigenvalues of $\mathbf{K}$.
	>
	> \- $\mathbf{V}_k$ is the $k \times n$ matrix, comprised of the top k eigenvectors of $\mathbf{K}$.
	>
	> \- $\mathbf{K}$ is the Gram matrix, where $k_{ij} = K(\boldsymbol{x}_{i}, \boldsymbol{x}_{j})=<\phi(\boldsymbol{x}_{i}),\phi(\boldsymbol{x}_{j})>$, which is the kernel function.
	>
	> Therefore, to obtain the tranformed dataset $\mathbf{Z}$, we need to compute the Gram matrix $\mathbf{K}$ first and center it, then, we use a eigen value decomposition to obtain the $\mathbf{V}$ and $\mathbf{\Lambda}$, and finally, using the above equation, we can compute the transformed dataset $\mathbf{Z}$ using the above equation.
	>
	> However, since the direction of the optimization is the same, we sometimes can remove the $\frac{1}{\sqrt{n-1}}$ term. This is how the `KernelPCA` in the Package `Scikit-learn` works. Advantages are:
	>
	> \- Improved Numerical Stability: Omitting the factor prevents transformed coordinates from becoming extremely small, especially for large n. This avoids potential floating-point precision issues, underflow errors, and increased sensitivity to rounding errors in subsequent computations on the reduced-dimensional data.
	>
	> \- Direct Kernel Space Scaling: This scaling is arguably more natural within the kernel context and avoids an arbitrary dependency on n without losing the essential relative geometry between data points.
	>
	> \- Formula Conciseness: The transformation formula is simpler and more directly linked to the kernel matrix eigendecomposition.

## K-Means

1. 在K-Means的收敛性推导中，After the updating step, the sum of squared distance is also ensured to not increasing (≤), if Euclidean distance is used to measure data similarity. 解释一下原因。

	> 在K-means算法中，更新步骤后（即重新计算簇中心后），平方距离之和（Sum of Squared Distances, **SSD**）确保不会增加（$\le$）。这主要是因为K-means算法的本质是一个迭代优化过程，它在每一步都试图最小化SSD。
	>
	> 具体来说，当使用欧几里得距离作为相似性度量时，每次迭代分为两步：
	>
	> 1. 分配步（Assignment Step）
	>
	> 在这一步中，每个数据点 $x_i$ 被分配到离它最近的簇中心 $c_j$。这个操作本身就是为了最小化每个数据点到其所属簇中心的距离平方。因此，在分配步之后，SSD必然会减少或保持不变。因为如果一个数据点可以分配到一个更近的簇中心，那么将其重新分配到那个更近的簇中心必然会降低总体的SSD。
	>
	> 2. 更新步（Update Step）
	>
	> 在这一步中，每个簇的中心被更新为其内部所有数据点的均值。**数学上可以证明，对于一个给定的数据点集合，这些点的均值是使集合内所有点到该均值的平方距离之和最小的那个点。**
	>
	> 假设一个簇 $C_k$ 包含数据点 $x_1, x_2, \ldots, x_m$。我们想找到一个点 $c_k$ 来最小化 $\sum_{i=1}^m \|x_i - c_k\|^2$。对 $c_k$ 求导并令其为零，可以得到 $c_k = \frac{1}{m}\sum_{i=1}^m x_i$，即这些点的均值。这意味着，在更新簇中心后，每个簇内部的平方距离之和达到了局部最小。
	>
	> ---
	>
	> **综合以上两步，每次迭代都会使得总的平方距离之和减小或者保持不变。**
	>
	> * **分配步**确保了每个点到其当前所属簇中心的距离是最小的（对于给定的簇中心）。
	> * **更新步**确保了每个簇的中心是其内部数据点的最优代表（使得簇内平方距离最小）。
	>
	> 这两步的联合作用保证了SSD在一个非递增的序列中。由于SSD是非负的，并且每次迭代都会减小或不变，这个过程最终会收敛到一个局部最优解，即SSD不再发生显著变化。

2. K-Means++.

	> K-Means++ 是一种用于**优化 K-Means 聚类算法初始质心选择**的方法。
	>
	> 
	>
	> 标准的 K-Means 算法有一个显著的缺点：它的聚类结果和收敛速度对**初始簇中心（也称为质心或均值点）的选择非常敏感**。
	>
	> * 如果初始质心选择得不好（例如，所有初始质心都挤在数据点的某一小部分区域），K-Means 算法很容易陷入**局部最优解**，导致最终的聚类效果很差，无法准确地反映数据的真实分布。
	> * 糟糕的初始选择还会导致算法需要更多的迭代才能收敛，从而降低效率。
	>
	> K-Means++ 就是为了解决这个问题而提出的。它的目标是选择一组“更好”的初始质心，使得这些质心在数据空间中尽可能地**分散开来**，从而提高 K-Means 算法的收敛速度和聚类质量。
	>
	> 
	>
	> K-Means++ 的核心思想是：**让选择的下一个初始质心，尽可能地远离已经选择的质心。** 这样可以确保选出的质心能够更好地覆盖整个数据空间，减少初始质心集中于某一区域的可能性。
	>
	> 
	>
	> 假设我们要将数据集聚类成 $K$ 个簇。K-Means++ 选择 $K$ 个初始质心的步骤如下：
	>
	> 1.  **选择第一个质心：**
	>     * 从所有数据点中**随机均匀地选择一个点**作为第一个簇中心 $c_1$。
	>
	> 2.  **选择后续质心（核心步骤）：**
	>     * 对于数据集中的**每一个数据点 $x_i$**，计算它到**目前为止已经选择的所有簇中心中最近那个中心**的距离。我们将这个距离表示为 $D(x_i)$。例如，如果已经选择了 $j$ 个质心 $c_1, c_2, \ldots, c_j$，那么 $D(x_i) = \min_{k=1,\ldots,j} \|x_i - c_k\|^2$（通常使用平方欧几里得距离）。
	>     * 接下来，选择下一个簇中心时，不再是随机均匀地选择，而是采用**带权重的随机抽样**。点 $x_i$ 被选为下一个质心 $c_{j+1}$ 的概率与其 $D(x_i)^2$ 成正比。
	>         具体公式为：$P(x_i) = \frac{D(x_i)^2}{\sum_{x_j \in \text{所有数据点}} D(x_j)^2}$
	>         这意味着，距离当前已选质心越远的点，被选为下一个质心的概率就越大。
	>
	> 3.  **重复步骤 2：**
	>     * 重复执行步骤 2，直到我们成功选择了 $K$ 个簇中心。
	>
	> 4.  **运行标准 K-Means：**
	>     * 一旦 $K$ 个初始质心被选择出来，就将它们作为标准 K-Means 算法的初始点，并按照 K-Means 的迭代过程（分配步和更新步）继续进行聚类，直到收敛。
	>
	> K-Means++ 的优点
	>
	> * **提高聚类质量：** 通过更合理地初始化质心，K-Means++ 显著降低了 K-Means 算法陷入局部最优解的风险，从而得到更优的聚类结果。
	> * **加快收敛速度：** 更好的初始质心通常意味着算法能够更快地找到稳定的聚类结果，减少迭代次数。
	> * **简单易实现：** 尽管比随机初始化复杂一些，但 K-Means++ 的逻辑相对直观，易于实现。
	>
	> 
	>
	> K-Means++ 是一种改进的 K-Means 算法初始化策略。它通过一种“距离最远优先”的随机抽样方法来选择初始簇中心，确保这些中心在数据空间中尽可能分散，从而有效提升了 K-Means 算法的聚类性能和稳定性。

3. Soft K-means.

在探讨 Soft K-means 算法时，其核心在于如何处理数据点对簇的归属这一不确定性。与传统的 K-means 算法中每个数据点被“硬性”地唯一分配给一个簇不同，Soft K-means 引入了概率的概念，允许每个数据点以一定的“责任”（responsibility）或概率属于多个簇。这种不确定性，使得数据点所属的簇成为了一个**隐藏变量（Latent Variable）**。

公式：
$$
r_{ij}=\frac{\exp(-\beta\lVert\mathbf{x}_i-\mathbf{\mu}_j\rVert^2_2)}{\sum_{k=1}^K \exp(-\beta\lVert\mathbf{x}_i-\mathbf{\mu}_k\rVert^2_2)}
$$

$$
\mathbf{\mu}_j=\frac{\sum_{i=1}^n r_{ij}\mathbf{x}_i}{\sum_{i=1}^n r_{ij}}
$$



> 1. 隐藏变量的引入
>
> 在 Soft K-means 中，我们无法直接观测到每个数据点究竟属于哪个簇。例如，一个数据点可能位于两个簇的中间区域，此时对其进行硬性划分会丢失信息。因此，将数据点 $x_i$ 属于哪个簇这一信息视为一个隐藏变量 $z_i$ 是必要的。我们的目标是估计这个隐藏变量的分布，以及模型的其他参数（如簇中心、簇的权重和方差等）。
>
> 2. EM 算法在 Soft K-means 中的应用
>
> 为了解决含有隐藏变量的概率模型的参数估计问题，**期望最大化（Expectation-Maximization, EM）算法**成为了 Soft K-means 的核心优化方法。EM 算法通过迭代的两个步骤，间接地最大化观测数据的似然函数：
>
> * **E 步（期望步 - Expectation Step）：**
> 	在此步骤中，我们利用当前已知的模型参数（例如，上一步迭代得到的簇中心和方差），来计算每个数据点 $x_i$ 属于每个簇 $k$ 的后验概率。这个后验概率就是所谓的“责任”或“软分配概率”，通常表示为 $\gamma(z_{ik})$ 或 $p(z_k|x_i)$。它量化了在当前模型参数下，数据点 $x_i$ 属于簇 $k$ 的“置信度”。这个步骤是对隐藏变量 $z_i$ 的期望进行估计，为后续的 M 步提供依据。
>
> * **M 步（最大化步 - Maximization Step）：**
> 	在 E 步计算出每个数据点对每个簇的责任后，M 步的目标是更新模型参数，以最大化当前观测数据在这些“责任”下的似然函数。例如，新的簇中心是通过所有数据点的加权平均计算得出的，其中权重即为该点对该簇的责任。这个过程本质上是在寻找一组新的模型参数，使得在给定 E 步的隐藏变量估计下，观测数据的概率达到最大。
>
> 3. 最大似然估计（MLE）思想的体现
>
> 整个 Soft K-means 算法，通过 EM 迭代过程，完美地体现了**最大似然估计（Maximum Likelihood Estimation, MLE）**的思想。
>
> EM 算法的目标是最大化观测数据 $\mathbf{X}$ 的边际似然 $P(\mathbf{X}|\mathbf{\theta})$，其中 $\mathbf{\theta}$ 代表所有模型参数。由于存在隐藏变量 $Z$，直接最大化边际似然是困难的。EM 算法通过转而最大化**期望的完全数据对数似然** $E_{Z|\mathbf{X}, \mathbf{\theta}_{\text{old}}}[\log P(\mathbf{X}, Z|\mathbf{\theta})]$ 来实现这一目标。
>
> * **E 步**负责计算这个期望值，即根据当前参数和观测数据，得出隐藏变量的最佳后验分布（即责任）。
> * **M 步**则在此期望值的基础上，选择新的参数来最大化它。
>
> 每一次 EM 迭代都保证似然函数是非递减的，从而确保算法最终收敛到一个局部最优解。因此，**Soft K-means 利用 EM 算法来处理不确定的簇分配这一隐藏变量，正是对最大似然估计原理的实际应用和实现。**

## DBSCAN

1. DBSCAN is short for Density-Based Spatial Clustering of Applications with Noise.

> DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 是一种**基于密度的聚类算法**。与 K-Means 等需要预先指定聚类数量（K值）的算法不同，DBSCAN 能够**发现任意形状的簇**，并且能够**识别出噪声点（异常点）**。
>
> 
>
> DBSCAN 的核心思想是：**一个簇是由密度相连（density-reachable）的点的最大集合**。它认为，如果一个区域的点密度足够高，那么这些点就属于同一个簇。
>
> 
>
> DBSCAN 算法主要依赖于两个核心参数：
>
> 1.  **$\epsilon$ (epsilon) / `eps`**：
>     * **半径**。它定义了一个圆形邻域的半径。对于数据集中的每个点，DBSCAN 会考虑在这个半径内有多少个其他点。
>
> 2.  **MinPts**：
>     * **最小点数**。在一个点的 $\epsilon$ 半径邻域内，如果包含的点的数量达到或超过 MinPts，那么这个点就被认为是**核心点（Core Point）**。
>
> 
>
> 根据这两个参数，DBSCAN 将数据点分为三种类型：
>
> 1.  **核心点（Core Point）**：
>     * 如果一个点在其 $\epsilon$ 邻域内包含至少 MinPts 个其他点（包括它自己），那么它就是一个核心点。核心点是簇的“骨架”。
>
> 2.  **边界点（Border Point）**：
>     * 如果一个点在其 $\epsilon$ 邻域内包含的点数少于 MinPts，但它位于某个核心点的 $\epsilon$ 邻域内（即它是某个核心点的直接密度可达点），那么它就是一个边界点。边界点是簇的“边缘”。
>
> 3.  **噪声点（Noise Point / Outlier）**：
>     * 如果一个点既不是核心点，也不是边界点（即它在其 $\epsilon$ 邻域内包含的点数少于 MinPts，并且它不属于任何核心点的邻域），那么它就是一个噪声点。噪声点不属于任何簇。
>
> 
>
> DBSCAN 算法的聚类过程大致如下：
>
> 1.  从数据集中随机选择一个未被访问的点。
> 2.  检查这个点的 $\epsilon$ 邻域。
>     * 如果其邻域内的点数小于 MinPts，则将该点标记为噪声点（暂时，它之后可能成为边界点）。
>     * 如果其邻域内的点数达到或超过 MinPts，则将该点标记为核心点，并创建一个新的簇。
> 3.  将该核心点邻域内的所有点添加到当前簇中。对于这些新加入的点，如果它们也是核心点，则递归地扩展它们的邻域，将更多密度相连的点加入到当前簇中。
> 4.  重复上述过程，直到所有点都被访问并标记（属于某个簇或被识别为噪声）。
>
> **优点：**
>
> * **无需预设簇的数量 K**：DBSCAN 能够根据数据的密度自动发现簇的数量。
> * **发现任意形状的簇**：它不像 K-Means 那样只擅长发现球状簇，DBSCAN 可以识别出不规则形状的簇，如L形、S形等。
> * **识别噪声点**：DBSCAN 能够明确地区分出哪些点是噪声，不将它们强制分配给任何簇。
> * **对初始点不敏感**：聚类结果对起始点的选择不敏感（除非起始点本身是噪声点）。
>
> **缺点：**
>
> * **参数选择困难**：$\epsilon$ 和 MinPts 这两个参数的选取对聚类结果有很大影响，且通常需要人工经验或多次尝试。
> * **处理密度差异大的簇有困难**：如果数据集中存在密度差异很大的簇（例如，一个非常密集的簇和一个非常稀疏的簇），DBSCAN 很难用同一组参数同时很好地处理它们。
> * **高维数据表现不佳**：在高维空间中，距离度量的有效性降低，选择合适的 $\epsilon$ 变得更加困难，导致“维度灾难”问题。
>
> 总而言之，DBSCAN 是一种强大且灵活的聚类算法，特别适用于处理具有复杂形状簇和包含噪声的数据集。

## MLE to MAP

1. Formula.

	> For Maximum Likelihood Estimation:
	> $$
	> \arg\max_{\theta}P(D|\theta)
	> $$
	> MLE就是求一个参数的取值，使得这个参数在这个样本上面表现的最好。
	>
	> For Maximum *A Posteriori*:
	> $$
	> \arg\max_{\theta} P(\theta|D)
	> $$
	> MAP是已经给定了这个数据，需要结合先验知识来得到一个后验的估计。这个后验的估计要求既能够较好地反映这个数据的分布特征，又要使这个参数在一般的常理之内。

2. 只是Bayes Theorem.

	> $$
	> P(\text{Posterior})=\frac{P(\text{Likelihood})\times P(\text{Prior})}{P(\text{Evidence})}
	> $$

## GMM

1. 在高斯混合模型中，如果我的所有数据都只用一个多元高斯分布来进行刻画，EM算法还有没有使用的必要？

  > 1. **EM算法的目的：** EM算法（期望最大化算法）主要用于**含有隐变量的概率模型**的参数估计。在高斯混合模型中，隐变量是每个数据点所属的高斯分量。当你有多个高斯分量时，你需要EM算法来迭代地估计每个数据点属于哪个分量（E步），然后根据这个估计来更新每个分量的参数（M步）。
  >
  > 2. **单个高斯分布的参数估计：** 如果您的模型只有一个多元高斯分布，那么就没有“混合”的概念，也没有隐变量来指示数据点属于哪个高斯分量（因为它只有一个）。在这种情况下，您可以直接使用**最大似然估计（MLE）**来求解该多元高斯分布的参数：
  >
  > 	- **均值（Mean）**：所有数据点的样本均值。
  > 	- **协方差矩阵（Covariance Matrix）**：所有数据点的样本协方差矩阵。
  >
  > 	这些参数可以直接通过封闭形式的解计算出来，不需要迭代过程。
  >
  > **总结：**
  >
  > - **高斯混合模型（GMM）**：当有**多个**高斯分量时，数据点的所属分量是隐变量，需要EM算法来估计参数。
  > - **单个多元高斯分布**：没有隐变量，可以直接通过最大似然估计（计算样本均值和样本协方差）来估计参数，无需EM算法。

2. 为什么GMM算法没有闭式解？

  > There is NO closed-form solution for them, one obvious reason is the interdependence of $\phi_j$ with $\mu_j$ and $\Sigma_j$. We need another solution to compute $\{\phi_j\}, \{\mu_j\}, \{\Sigma_j\}$, under the perspective of maximizing data log-likelihood $\log p(D|\{\phi_j\}, \{\mu_j\}, \{\Sigma_j\})$
  >
  > The Expectation-Maximization algorithm introduces a way to address this task. 
  >
  > It is very similar to the two-step process in k-means
  >
  > 当我们在 GMM 中最大化似然函数时，我们会遇到一个对数内部包含求和项的表达式：
  > $$
  > L(\theta)=\sum\log\left(\sum \pi_k\mathcal{N}(x_i|\mu_k,\Sigma_k)\right)
  > $$
  > 
  >
  > 这个对数内部的求和项正是症结所在。它意味着我们不知道每个数据点 $x_i$ 究竟是由哪个高斯分量生成的（这就是隐变量）。
  > 如果这个隐变量已知，我们就能将问题分解成多个独立的高斯分布估计，每个都有闭式解。
  > 然而，由于隐变量是未知的，我们无法直接求解对数似然函数的导数并将其设为零来得到解析解。
  >
  > EM 算法正是为了解决这类含有隐变量的问题而生。它通过迭代的方式，先“猜测”隐变量的分布（E 步），然后基于这个猜测更新模型参数（M 步），从而逐步逼近最优解。

3. 模型（GMM）的 E 步中，为什么对于每一个数据点 $x_i$，其对所有高斯分量 $j$ 的责任 $r_{ij}$ 之和 $\sum_{j=1}^K r_{ij}$ 必须等于 1？这个性质与 M 步中混合权重 $\pi_j$ 的更新有什么关系？

    > 1. 为什么 $\sum_{j=1}^K r_{ij} = 1$？
    >
    > 这个等式是基于**概率的完备性原则**。
    >
    > * **$r_{ij}$ 的定义：** $r_{ij}$ 表示给定数据点 $x_i$ 和当前模型参数 $\theta$，数据点 $x_i$ 来自第 $j$ 个高斯分量的后验概率 $p(z_{ij}=1 | x_i, \theta)$。
    > * **隐变量的性质：** 在 GMM 中，我们假设每一个数据点 $x_i$ 都**必然且只**由 $K$ 个高斯分量中的**某一个**生成。也就是说，对于每个数据点 $x_i$，其对应的隐变量 $z_i$（表示它属于哪个分量）必然会是 $1, 2, \ldots, K$ 中的一个确定值。
    > * **概率的归一化：** 由于数据点 $x_i$ 必然属于且仅属于一个分量，那么它来自所有可能分量的后验概率之和必须为 1。这就像任何一个事件在所有可能结果上的概率之和总是 1。
    >
    > 因此，
    > $$
    > \sum_{j=1}^K r_{ij} = \sum_{j=1}^K p(z_{ij}=1 | x_i, \theta) = 1
    > $$
    > 这确保了每个数据点的“责任”在所有分量上是完整分配的，没有遗漏或重复。
    >
    > 2. 与 M 步中混合权重 $\pi_j$ 更新的关系
    >
    > $\sum_{j=1}^K r_{ij} = 1$ 这个特性在 M 步中更新混合权重 $\pi_j$ 时起着**至关重要的作用**，它保证了更新后的权重是合理且规范化的。
    >
    > 在 M 步中，混合权重 $\pi_j$ 的更新公式为：
    > $$
    > \pi_j^{new} = \frac{\sum_{i=1}^N r_{ij}}{N}
    > $$
    >
    > 其关系体现在以下几点：
    >
    > * **分子的含义：** $\sum_{i=1}^N r_{ij}$ 表示第 $j$ 个高斯分量对整个数据集所有数据点所承担的“总责任”。我们可以将其理解为**第 $j$ 个簇所“拥有”的有效数据点数量**（因为 $r_{ij}$ 是软分配）。
    > * **分母的含义：** $N$ 是数据集中的**总数据点数量**。
    > * **$\pi_j$ 的物理意义：** $\pi_j$ 代表第 $j$ 个高斯分量在整个混合模型中所占的比例或先验概率。
    > * **总责任的守恒：** 我们可以验证所有簇的总责任之和等于总数据点数 $N$：
    >     $$
    >     \sum_{j=1}^K \left( \sum_{i=1}^N r_{ij} \right) = \sum_{i=1}^N \left( \sum_{j=1}^K r_{ij} \right)
    >     $$
    >     由于 $\sum_{j=1}^K r_{ij} = 1$（每个数据点的责任之和为 1），上式变为：
    >     $$
    >     \sum_{i=1}^N (1) = N
    >     $$
    >     这意味着，所有分量“分享”的总有效数据点数量恰好等于实际的数据点总数 $N$。
    > * **保证 $\pi_j$ 的合理性：** 将第 $j$ 个簇的有效数据点数量 $\sum_{i=1}^N r_{ij}$ 除以总数据点数 $N$，得到的 $\pi_j^{new}$ 自然地表示了该簇在整个数据集中的比例。由于所有簇的有效数据点总数等于 $N$，这自动保证了所有更新后的混合权重之和为 1：
    >     $$
    >     \sum_{j=1}^K \pi_j^{new} = \sum_{j=1}^K \frac{\sum_{i=1}^N r_{ij}}{N} = \frac{1}{N} \sum_{j=1}^K \sum_{i=1}^N r_{ij} = \frac{1}{N} \sum_{i=1}^N \left( \sum_{j=1}^K r_{ij} \right) = \frac{1}{N} \sum_{i=1}^N (1) = \frac{1}{N} \cdot N = 1
    >     $$
    >
    > 综上所述，$\sum_{j=1}^K r_{ij}=1$ 是 E 步中计算后验概率的**基本性质**，它确保了每个数据点的责任被完全分配。这个性质在 M 步中被巧妙地利用，使得混合权重 $\pi_j$ 的更新公式能够**准确地反映**每个簇所“捕获”的数据点的比例，并且**自动满足**所有混合权重之和为 1 的必要条件。

4. GMM中EM算法的公式：

  > E步：
  > $$
  > r_{ij} = \frac{\pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}{\sum_{k=1}^K \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}
  > $$
  > M步：
  > $$
  > \mu_j^{new} = \frac{\sum_{i=1}^N r_{ij} x_i}{\sum_{i=1}^N r_{ij}}\\
  > \Sigma_j^{new} = \frac{\sum_{i=1}^N r_{ij} (x_i - \mu_j^{new})(x_i - \mu_j^{new})^T}{\sum_{i=1}^N r_{ij}}\\
  > \pi_j^{new} = \frac{\sum_{i=1}^N r_{ij}}{N}
  > $$

5. K-means是如何对GMM进行初始化的？

  > **运行 K-means：** 首先，对数据运行 K-means 算法，得到 $K$ 个硬性（明确划分的）簇和它们的质心。
  >
  > **设置 GMM 初始参数：**
  >
  > - **均值 ($μ_j$)：** 将每个高斯分量的初始均值设置为对应 K-means 簇的质心。
  > - **协方差 ($Σ_j$)：** 将每个高斯分量的初始协方差设置为对应 K-means 簇内数据点的样本协方差。
  > - **混合系数 ($π_j$)：** 将每个高斯分量的初始混合系数设置为对应 K-means 簇中数据点数量占总数据点数量的比例。

6. Summary of a GMMs by the EM Algorithm proof.

    > - We start from representing data likelihood $\log p(D|\theta)$
    > - Then look at sample likelihood $\log p(x_i|\theta)$
    > - Introducing a latent variable $z_i$ and its latent distribution $q(z_i)$
    > - We found out that an alternative way to approximate MLE: maximizing the ELBO of the data likelihood $\log p(x_i|\theta)$
    > - The E-step finds one representation of the ELBO, by equalizing $q(z_i)$ with the posterior $p(z_i|x_i, \theta) \Leftrightarrow$ responsibility $r_{ij}$
    > - The M-step then maximizing this ELBO through zero-derivatives, therefore leading to the solution of parameters $\theta$, namely, $\{\phi_1, \mu_1, \Sigma_1\}, \{\phi_2, \mu_2, \Sigma_2\}, ... \{\phi_k, \mu_k, \Sigma_k\}$

## Ensemble Learning

1. Prove that the bound of training error is
$$
Pr_{i\sim D_1}[H(x_i)\neq y_i]\leq\prod_{t=1}^T\sqrt{1-4\gamma_t^2}=\exp\left(-2\sum_{t=1}^T\gamma_t^2\right)
$$
where $\gamma_t=\frac{1}{2}-\epsilon_t$. Comprehension about it.

> Solution: if $\forall t: \gamma_t \ge |\gamma| > 0$, then training error $\le e^{-2\gamma^2 T}$.
>
> The term $e^{-2\gamma^2 T}$ is related to the time $T$, so: 
>
> - As the training progresses, the upper bound of the training error will reduce exponentially (fast training).
> - Convergence? Two conditions:
> 	1. Training error goes to $0$. 
> 	2. Or, $\gamma_t = 0$, equivalently, $\epsilon_t = 0.5$. Boosting gets stuck: the boosting weights on training set are in such a way that every weak learner has 50% error.

2. What is the meaning of "Bootstrap" in Ensemble Learning?

	> 在集成学习（Ensemble Learning）中，“Bootstrap”通常指的是**自助采样法**，它是**Bagging（Bootstrap Aggregating）** 这种集成学习方法的核心组成部分。
	>
	> 
	>
	> Bootstrap 是一种**有放回的随机抽样**方法。它的基本思想是从原始数据集中反复地、有放回地抽取与原始数据集大小相同（或近似相同）的样本集。
	>
	> 具体来说，对于一个包含 $N$ 个样本的原始数据集 $D$：
	> 1.  **有放回抽样：** 从 $D$ 中随机抽取一个样本，并将其添加到新的样本集 $D'$ 中。
	> 2.  **重复 $N$ 次：** 重复步骤 1 共 $N$ 次。
	> 3.  **生成新的数据集：** 最终得到的 $D'$ 就是一个自助采样集。$D'$ 的大小与 $D$ 相同，但由于是有放回抽样，它可能包含 $D$ 中重复的样本，也可能缺失 $D$ 中的一些样本（估计约有 36.8% 的原始样本不会出现在 $D'$ 中）。
	>
	> 通过重复这个过程多次，我们可以生成多个不同的自助采样集 $D'_1, D'_2, \ldots, D'_M$。