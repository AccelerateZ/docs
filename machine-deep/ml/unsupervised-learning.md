# Data Analysis - Unsupervised Learning

## Perceptron

1. How to interpret the loss function?
$$
	L(w_1,w_2,b)=(w_1x_1+w_2x_2+ b)(y'-y)
$$

该损失函数衡量的是预测结果与真实标签之间的差异以及模型参数的线性组合。具体来说：

- $w_1 x_1 + w_2 x_2 + b$ 是线性分类器的输出（即预测的“激活值”）。
- $y'$ 是模型预测的标签（通常为 +1 或 -1）。
- $y$ 是真实标签。

当 $y' = y$ 时，说明预测正确，损失为零或接近零；当 $y' \neq y$ 时，损失为一个非零值，反映了预测的错误程度。

这个损失函数用于指导权重 $w_1, w_2$ 和偏置 $b$ 的调整，使得预测结果更接近真实标签，从而实现分类性能的提升。

2. What if we are dealing with binary labels $\{-1, +1\}$?

If the binary labels are from the set $\{+1, -1\}$, the perceptron loss function

$$
L(w_1, w_2, b) = (w_1 x_1 + w_2 x_2 + b)(y' - y)
$$

can be interpreted as follows:

- Since $y, y' \in \{+1, -1\}$, the term $(y' - y)$ can take values in $\{-2, 0, 2\}$.
- The predicted label $y'$ is usually defined by the sign of the linear function:

$$
y' = \operatorname{sign}(w_1 x_1 + w_2 x_2 + b)
$$

- The loss encourages correct classification by pushing the decision boundary to separate positive and negative samples.
- Specifically, the perceptron update rule aims to reduce the loss when a sample is misclassified, i.e., when

$$
y (w_1 x_1 + w_2 x_2 + b) \leq 0
$$

which means the prediction $y'$ disagrees with the true label $y$.

In summary, for labels in $\{+1, -1\}$, the perceptron loss is positive when the prediction is wrong and zero or negative when correct, driving the weight updates accordingly.

**标签定义：**  
- $\{0, 1\}$ 表示标签为“否”或“是”，通常用作概率或类别指示。
- $\{+1, -1\}$ 则表示正负类别，更方便用于数学推导，特别是在感知机和支持向量机等算法中。

2. Please give some example of the non-linear decision boundary.

	k-NN.

## Logistic Regression

1. Now we use another new model, which is the logistic model:
$$
f(x)=\frac{1}{1+e^{-x}}.
$$
It depicts the probabilities of the predicted label. Now, we consider a classical setting, optimizing the model to maximize the likelihood of data observations given the current parameters. How to define data likelihood for the whole training set?

把所有样本的**条件概率**连乘起来，得到联合似然（joint likelihood）：
$$
L(\theta) = \prod_{i=1}^{n} P(y^{(i)} | x^{(i)}; \theta)
$$

2. Then we present sample likelihood $p(y_i|x_i.\theta)$. What does the computation mean?
$$
p(y_i|x_i.\theta) = y_{i}^{\prime y_{i}} \cdot\left(1-y_{i}^{\prime}\right)^{\left(1-y_{i}\right)}
$$

由于 $y_i$ 的取值只能是 0 或 1，所以当 $y_i = 1$ 时，该表达式等于 $y_i'$，即模型预测该样本为正类的概率；  当 $y_i = 0$ 时，表达式等于 $1 - y_i'$，即模型预测该样本为负类的概率。  

这个表达式实际上综合了正负两类情况下的概率，为后续在最大似然估计中求解模型参数提供了基础，  使得可以根据所有样本的似然之积来构造目标函数，进而通过优化该目标函数来估计模型参数 $\theta$，  使模型对训练数据的拟合达到最优。
	
3. Why the optimization in LR is better than the one in perceptron? 

LR: Logistic Regression:
$$
\frac{\partial L}{\partial w} = \frac{x_{i}}{n}(y'-y)
$$
Perceptron:
$$
\frac{\partial L}{\partial w} = x_{i}(y'-y)
$$

> 1. 损失函数的差异
>
> - **感知机**使用的是**硬判别误差（0/1损失）**，只在错误分类时才更新参数，且更新量不考虑“错得多严重”。
> - **逻辑回归**使用的是**对数损失（log loss）**，考虑的是**概率距离**，即模型对每个样本的预测信心也纳入优化。
>
> 2. 梯度方向的平滑性
>
> - 感知机的更新是离散的、跳跃的，要么更新、要么不更新，而且没有“错得更远就更新更大的机制”。
> - 逻辑回归的更新是**连续的、光滑的**，即使分类正确，只要概率信心不够大，也会适度更新。**这让优化路径更稳定、更容易收敛。**
>
> 3. 可导性和概率解释
>
> - LR 的损失函数是**可导的、凸函数**，优化问题是**凸优化**，容易收敛到全局最优。
> - 感知机的损失函数是非凸、不可导（或是 piecewise-linear），更容易陷入震荡或无法收敛。
>
> 逻辑回归中参数更新的优势，跟梯度公式中的 n（样本数）有没有关系？
>
> **有关系，但不是核心原因；n 更像是“实现细节”而非“本质区别”。**
> 	

4. Hence, minimizing the cross - entropy loss $L$ is equal to minimizing the summarized KL divergence 
$$
\sum_{i=1}^{n} D_{KL}\left(P(y_i|x_i, \theta)\right\|P\left(y_i^{\prime}|x_i, \theta\right)
$$
over all samples. What does it mean?

我们的模型在训练过程中，通过最小化交叉熵损失，实际上是在让预测分布尽可能接近真实标签分布，即在最小化两者之间的KL散度之和。这从理论上解释了为什么交叉熵损失是用于分类模型训练的一个合理的选择，因为它与衡量分布差异的KL散度紧密相关，而KL散度本身是一个在信息论中有明确意义的概念。

- Relationship between KL divergence and cross-entropy loss:
$$
\sum_{i=1}^{n} D_{\text{KL}}(P(y_i|x_i,\theta)\|P(y_i^{'}|x_i,\theta)) - \sum_{i=1}^{n} \sum_{j=1}^{k} (y_{j i}\log y_{j i}) = nL
$$
$$
L=-\frac{1}{n}\sum_{i=1}^{n}\left(\sum_{j=1}^{k} y_{j i}\log y_{j i}^{'}\right)
$$

## k-NN

1. $\mathcal{l}_p$ norm:
$$
  l_p = \|x\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}\\
$$

1. 当 $p = 0$：伪范数，表示向量中非零元素的个数。

- 注意：这不是严格意义上的范数（不满足齐次性），常用于稀疏建模（如 $L_0$ 最小化）

---

2. 当 $p = 1$：曼哈顿距离（$L_1$ 范数）

- 强调稀疏性，对特征选择友好，常用于 Lasso 回归、稀疏优化

---

3. 当 $p = 2$：欧几里得范数（$L_2$ 范数）

$$
\|x\|_2 = \sqrt{ \sum_{i=1}^{n} x_i^2 }
$$

- 表示“直线距离”，常用于岭回归、标准化、最小二乘问题

---

4. 当 $p \to \infty$：最大值范数（$L_\infty$ 范数）

$$
\|x\|_\infty = \max_i |x_i|
$$

- 强调各维度中的最大偏差常用于最大误差控制、鲁棒优化

2. Hamming距离：对于两个等长的字符串，汉明距离就是算其中有多少个位置的字符不同。
$$
  \text{Hamming}(x,y)=\sum_{i=1}^n\mathbf{1}_{x_i\neq y_i}
$$

3. In the field of k-NN, when k increases or decreases, how the boundary varies?

   - When decreasing $k$, it generates more complex decision boundaries, tends to overfit the training data for small $k$.
   - When increasing $k$, it generates smoother decision boundaries, tends to underfit the training data for large $k$
4. The complexity of k-NN.

对每一个测试样本 $x$，计算它到每个训练样本 $x_i$ 的距离，然后找出最近的 $k$ 个点。

- **距离计算复杂度**：$O(d)$ （比如欧几里得距离需要遍历每个维度）
- **总距离计算**：$O(n \cdot d)$

- **选出前 $k$ 个最近点**：可以使用最小堆（heap），复杂度为 $O(n \log k)$

- **总体复杂度（单个测试点）**：  
  $$
  O(n \cdot d + n \log k) \approx O(n \cdot d) \quad (\text{因为 } \log k \ll d)
  $$

- **若有 $m$ 个测试点**：  
  $$
  O(m \cdot n \cdot d)
  $$

5. Cover和Hart定理：当样本数量趋于无穷大时，1-NN 分类器的误分类率（渐近风险）最多是贝叶斯误差的两倍。
>
> 	For a small $k$ value and a large $n$ value, we have:
> $$
> 	R(\tilde x_i)\leq (1+\sqrt{c/k})R^*(x_i),c\leq8.
> $$
> 	Keep increasing $k$ and $n\to\inf$, we have the universal consistency:
> $$
> \lim_{n\to\infty}R(\tilde x_i)\leq(1+\sqrt{2/k})R^*(x_i).
> $$
>


## Model Evaluation and Selection

1. 分类模型评估指标：准确率、召回率、精确率

    **1. 准确率（Accuracy）**

    - **定义：** 模型正确分类的样本数占总样本数的比例，衡量模型整体的正确性。
    - **公式：**
      $$
      \text{Accuracy} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}}
      $$
    - **说明：**
      
      - 关注所有预测的正确性。
      - 在类别不平衡的数据集中可能具有误导性。

    ---

    **2. 召回率（Recall）/ 真阳性率（TPR）/ 敏感性（Sensitivity）**

    - **定义：** 在所有实际为正类别的样本中，模型正确识别出的正类别样本的比例。衡量模型找出所有正类别的能力（防止漏报）。
    - **公式：**
      $$
      \text{Recall} = \frac{\text{TP}}{\text{TP + FN}}
      $$
    - **说明：**
      
      - 关注真实的正类别样本是否被识别出来。
      - 高召回率意味着更少漏掉正类别。
      - 适用于假阴性（漏报）成本高的场景。

    ---

    **3. 精确率（Precision）/ 查准率（PPV）**

    - **定义：** 在所有模型预测为正类别的样本中，实际是正类别的比例。衡量模型预测正类别的准确性（防止误报）。
    - **公式：**
      $$
      \text{Precision} = \frac{\text{TP}}{\text{TP + FP}}
      $$
    - **说明：**
      - 关注模型预测为正类别的样本有多准确。
      - 高精确率意味着更少误报负类为正类。
      - 适用于假阳性（误报）成本高的场景。

    ---

    **4. F1 分数（F1-Score）**

    - **定义：** 精确率和召回率的调和平均值。是一个综合两者的单一指标。
    - **公式：**
      $$
      \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
      $$
    - **说明：**
      - 平衡了精确率和召回率。
      - 在类别不平衡数据集中，比单独的准确率更能有效反映模型性能。
      - 适用于需要同时兼顾防止漏报和误报的场景。

2. ROC Curve.

    ROC 曲线，全称是 **Receiver Operating Characteristic curve（接收者操作特征曲线）**，是用于评估**二分类模型性能**的重要工具。
    
    横轴（X）: **FPR**（False Positive Rate，也叫False Alarm Rate）
    
    纵轴（Y）: **TPR**（True Positive Rate，也叫召回率 Recall）
    $$
    \text{FPR}=\frac{\text{False Positive}}{\text{False Positive + True Negative}}
	$$
	$$
    \text{TPR}=\frac{\text{True Positive}}{\text{True Positive + False Negative}}\\
    $$
    
    
    
    **完美分类器 (Perfect Classifier)**
    
    * **特点**: 在ROC曲线上表现为$(0.0, 1.0)$的一个点。
    * **含义**:
        * **假正例率 (FPR) 为 $0.0$**: 意味着模型没有产生任何误报。所有实际为负的样本都被正确地识别为负。
        * **真正例率 (TPR) 为 $1.0$**: 意味着模型成功识别了所有实际为正的样本，没有遗漏任何正例。
    * **实际意义**: 完美分类器是一个理想化的概念，在现实世界中几乎不可能达到。它代表了模型性能的上限，即能够完美地将所有正类和负类样本区分开。
    
    **随机分类器 (Random Classifier)**
    
    * **特点**: 在ROC曲线上表现为一条从$(0.0, 0.0)$ 到 $(1.0, 1.0)$的对角线。
    * **含义**:
        * 一个随机分类器，其预测结果与真实标签无关，就像随机猜测一样。
        * 在这种情况下，模型识别出正例的概率大致等于它将负例错误识别为正例的概率，即 **$TPR \approx FPR$**。
    * **实际意义**: 任何一个有实际应用价值的分类器，其ROC曲线都应该位于这条随机分类器曲线的上方。如果模型的ROC曲线与或低于这条对角线，则说明其性能不佳，甚至比随机猜测还要差。它通常作为衡量模型性能的基准线。
    
3. The operation point. What about using performance at fixed operating points? 

  > Very practical metrics, because in real-world scenarios, there are limits regarding how many FPs/FNs that a system can tolerate. For example, detection rate at a fixed false positive rate.
  >

  - **什么是“操作点”？** 在ROC曲线中，一个“操作点”指的是在特定的分类阈值下，模型所对应的**一对 (FPR, TPR)** 值。当你设定一个阈值时，模型会根据这个阈值将样本分类，从而得到一个特定的FPR和TPR。改变阈值，就会得到不同的操作点。
  - **“固定操作点”是什么意思？** 这意味着我们不是看整个ROC曲线的整体性能（比如AUC），而是选择一个或几个特定的FPR或TPR值作为基准，然后比较模型在这些固定基准下的表现。

4. Validation set and K-fold cross-validation

	**1. 验证集 (Validation Set)**
	
	* **概念**: 在训练模型时，将数据集划分为三部分：
	    * **训练集**: 用于模型学习和参数调整。
	    * **验证集**: 用于**超参数调优**和在训练过程中评估模型性能，避免过拟合。
	    * **测试集**: 用于**最终评估**模型在未见过数据上的泛化能力，只在模型开发完成后使用一次。
	* **作用**: 防止过拟合，辅助超参数选择和模型选择。
	* **局限**: 评估结果可能受随机划分影响，数据利用率相对较低。
	
	**2. k-折交叉验证 (k-fold Cross-Validation)**
	
	* **概念**: 一种更鲁棒的评估方法。将数据集（通常是训练数据）分成 $k$ 个子集（折）。
	* **步骤**: 进行 $k$ 次迭代：每次取一个子集作验证集，其余 $k-1$ 个作训练集，训练并评估模型。
	* **结果**: 将 $k$ 次评估结果取平均，得到更稳定、可靠的模型性能估计。
	* **作用**: 提供更可靠的性能估计，充分利用数据，减少评估偏差。
	* **局限**: 计算成本更高（训练 $k$ 次）。

## Decision Trees

1. **决策树 (Decision Tree)**

	决策树是一种直观且强大的监督学习算法，通过构建树状结构进行分类或回归。
	
	* **非线性模型**: 
	    与逻辑回归等广义线性模型不同，决策树通过一系列条件判断将输入空间划分为多个区域，能够捕捉数据中的复杂非线性模式，类似于K-NN这样的非线性模型。
	
	* **自然处理连续和类别数据**: 
	    决策树天生就能很好地处理数值型和类别型特征，无需额外的预处理步骤（如独热编码），即可根据特征值进行分岔判断。
	
	* **训练和测试速度快（高度可并行化）**: 
	    一旦训练完成，决策树的预测过程非常迅速，通过简单的条件判断即可得出结果。在集成学习中，多棵树的训练也可以并行进行，提高效率。
	
	* **生成一套可解释的规则**: 
	    决策树的一大优势是其高可解释性。每条从根节点到叶节点的路径都可以转化为清晰的“如果...那么...”规则，这些规则易于人类理解和解释，对于需要模型透明度的应用场景非常有用。
	
2. Linear and non-linear model.

	**逻辑斯蒂回归 (Logistic Regression)** 是一种**广义线性模型 (Generalized Linear Model)**，通常被认为是**线性模型**。
	虽然它的名字中带有“回归”二字，并且输出的是一个介于0到1之间的概率值，似乎可以处理非线性问题，但它的核心是**线性组合输入特征**。
	
	具体来说，逻辑斯蒂回归的数学表达式如下：
	
	首先，它计算一个线性得分（或称为对数几率）：
$$
	z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
$$
其中，$\beta_0, \beta_1, \dots, \beta_n$ 是模型的系数（权重），$x_1, x_2, \dots, x_n$ 是输入特征。
	
然后，它将这个线性得分通过一个**Sigmoid（S型）激活函数**（也称为逻辑函数）转换成一个概率值：
$$
	P(Y=1|X) = \frac{1}{1 + e^{-z}}
$$

这个Sigmoid函数将任何实数值 $z$ 映射到 $(0, 1)$ 区间内，从而可以解释为概率。

> ​	
> **为什么是线性模型？**
>
> * **决策边界是线性的**：尽管Sigmoid函数引入了非线性变换，但它所作用的输入 $z$ 仍然是特征的线性组合。这意味着，在原始特征空间中，逻辑斯蒂回归的**决策边界是线性的**。例如，对于二维数据，决策边界就是一条直线；对于三维数据，决策边界就是一个平面。
> * **模型本身是线性可分的**：逻辑斯蒂回归尝试找到一个线性组合的权重，使得通过Sigmoid函数转换后，能够最好地区分两个类别。
>
> **与非线性模型的区别：**
>
> * **决策树**和**K-近邻 (K-NN)** 等模型被称为非线性模型，是因为它们的决策边界可以是任意形状的、非线性的，不局限于直线或平面。它们通过对特征空间进行更复杂的划分来学习模式。
> * 尽管逻辑斯蒂回归可以用于处理一些非线性可分的问题（通过特征工程，如引入多项式特征或交互特征，将非线性问题转换为高维空间的线性问题），但其**基本模型结构仍然是线性的**。
>
> 所以，总结来说，逻辑斯蒂回归在数学上属于线性模型范畴，其决策边界是线性的。
>

3. Disadvantages.

	决策树在模型复杂度和泛化能力之间存在一个固有的权衡，这直接导致了它的主要缺点：
	
	* **容易过拟合 (Overfitting)**：
	    * **原因**: 当决策树的节点（规则）过多，树的深度过大时，它会过于关注训练数据中的细节和噪声，为每个样本单独地创建预测规则。
	    * **表现**: 模型在训练集上表现极好（例如，能够为每个训练样本预测正确的标签），但在未见过的新数据（测试集）上泛化能力差，性能急剧下降。
	    * **结果**: 导致模型过于复杂，失去了对普遍模式的捕捉能力，在新数据集上可能无法很好地工作。
	
	* **容易欠拟合 (Underfitting)**：
	    * **原因**: 当决策树的节点（规则）过少，树的深度很浅时，它只能学习到数据中最简单的模式。
	    * **表现**: 如果训练集本身比较复杂，小规模的决策树无法捕捉到足够的模式来正确分类，导致在训练集和测试集上都表现不佳。
	    * **结果**: 训练损失/误差较大，说明模型过于简单，没有充分学习到数据的潜在结构。
	
	* **对数据敏感/不稳定 (Sensitive to Data Variations)**：
	    * 由于决策树通过一系列硬性规则进行划分，训练数据的微小变化（例如，某个特征值略微不同）都可能导致树的结构发生巨大变化，从而影响模型的稳定性和泛化能力。
	    * 这使得单一的决策树模型在实践中通常表现不如集成学习方法（如随机森林、梯度提升树）鲁棒。
	
	总结来说：**“大树过拟合，小树欠拟合”**。决策树的这种特性使得在实际应用中，很难直接构建一个完美平衡泛化能力的单一决策树。因此，在实践中，我们通常会使用决策树的集成学习方法来克服这些缺点。

4. Information Gain and Entropy.

	> 1. Entropy $H(X)$ measures the randomness of  the random variable $X$, calculated as:
	> $$
	> H(X)=-\sum_{x\in X}p(x)\log p(x).
	> $$
	>
	> 2. Higher entropy = more uncertainty = less predictable (less informative)
	>
	> 3. 信息增益 $IG(D,A)$ 用于衡量使用特征 $A$ 对数据集 $D$ 进行划分时，数据集不确定性减少的程度。
	> $$
	> IG(D,A)=H(D)-H(D|A)
	> $$
	>
	> $$
	>   H(D|A)=\sum_{v \in \operatorname{Values}(A)} \frac{\left|D_{v}\right|}{|D|} H\left(D_{v}\right)
	> $$
	>
	> 把总的样本按照特征$A$的取值分成几类，在每一类里面再一次算信息熵。最后把每一类得到的信息熵前面乘以该类的概率即可。

## Naive Bayes Classifier

1. 公式.
$$
P(c|x) = \frac{P(x|c)P(c)}{P(x)}
$$
其中：
* $P(c|x)$ 是给定特征 $x$ 时类别 $c$ 的后验概率（我们希望计算的值）。
* $P(x|c)$ 是给定类别 $c$ 时特征 $x$ 的似然度。在朴素贝叶斯分类器中，我们假设特征之间是条件独立的，因此：
$$
P(x|c) = P(x_1|c) \cdot P(x_2|c) \cdot \dots \cdot P(x_n|c)
$$
其中 $x_1, x_2, \dots, x_n$ 是特征 $x$ 的各个维度。
* $P(c)$ 是类别 $c$ 的先验概率。
* $P(x)$ 是特征 $x$ 的证据或边缘概率，通常作为**归一化常数**，在比较不同类别时可以忽略，因为对于所有类别，它都是相同的。因此，在实际应用中，我们通常比较：
$$
P(c|x) \propto P(x|c)P(c)
$$
并选择使 $P(x|c)P(c)$ 最大的类别。

当然，如果需要去求解的话，使用全概率公式：
$$
P(x)=\sum_c P(x|c)P(c)
$$
2. Laplace Smoothing.

$$
P(\text{Feature|Class})=\frac{a+\beta}{A+k\beta}
$$
$a$ is the number of examples in class with the feature value,
$A$ is the number of examples in class,
$k$ is the number of possible values for the feature,
and $\beta$ is a real number.


3. Is Naive Bayes Classifier parametric or non-parametric?

	这里探讨了朴素贝叶斯分类器是参数模型还是非参数模型的问题。它提出了两种不同的情景，导致了不同的结论。
	
	* **情景一：直接推导（非参数观点）**
	    展示了朴素贝叶斯的核心公式：$p(y_i = c | \vec{x}_i) \propto p(\vec{x}_i | y_i = c) p(y_i = c)$。
	    它指出，先验概率 $p(y_i = c)$ 和似然概率 $p(\vec{x}_i | y_i = c)$ 可以直接从训练数据中估算（推导）出来，而不需要任何优化过程。
	    在这种特定情况下，贝叶斯可以被视为一个**非参数模型**。
	
	* **情景二：使用概率分布（参数观点）**
	    这种情景考虑了如果数据服从特定的概率分布（例如高斯分布）时会发生什么。
	    如果先验 $p(y_i = c)$（更准确地说，是似然 $p(\vec{x}_i | y_i = c)$）被假设服从高斯分布，那么我们就需要估计该高斯分布的参数（例如均值和方差）。
	    估计这些参数可能需要优化技术。
	    在这种情况下，朴素贝叶斯模型就变成了一个**参数模型**。
	
	**为什么它可以分为参数和非参数？它们的划分依据是什么？**
	
	理解朴素贝叶斯为何既可以被视为参数又可以被视为非参数的关键在于对这两个术语的定义以及模型组件的处理方式。
	
	**参数模型（Parametric Models）：**
	
	* **定义：** 参数模型假设一个**固定且有限数量的参数**来描述底层数据分布。模型的结构是预先确定的，学习过程就是从数据中估计这些参数。
	* **特点：**
	    * 它们对数据分布的形式做出强烈的假设（例如，正态分布、伯努利分布）。
	    * 模型的复杂性是固定的，与训练数据的数量无关。
	    * 一旦参数被估计出来，原始的训练数据通常可以被丢弃，因为模型完全由其参数定义。
	* **朴素贝叶斯中的例子（根据幻灯片）：** 当我们假设似然 $p(\vec{x}_i | y_i = c)$ 服从特定分布（如高斯分布）时，我们实际上是说每个类别 $c$ 中的数据可以通过该高斯分布的均值 ($\mu$) 和方差 ($\sigma^2$) 来描述。这些 $\mu$ 和 $\sigma^2$ 就是需要估计的**参数**。这使其成为一种参数方法。幻灯片明确提到了“优化高斯参数 $\theta$”，其中 $\theta$ 就代表这些参数。
	
	**非参数模型（Non-Parametric Models）：**
	
	* **定义：** 非参数模型**不假设固定且有限数量的参数**或底层数据分布的特定函数形式。相反，模型的复杂性通常随数据量的增加而增长，或者它直接使用数据本身进行预测。
	* **特点：**
	    * 它们对数据分布的形式做出的假设较少或没有假设。
	    * 模型的复杂性可以随训练数据的大小而增长。
	    * 它们通常依赖于训练数据本身（或其子集）来做出预测。
	* **朴素贝叶斯中的例子（根据幻灯片）：** 如果我们通过简单地计算训练数据中的出现次数来直接估计概率 $p(y_i = c)$ 和 $p(\vec{x}_i | y_i = c)$（例如，使用经验频率），而不拟合任何预定义的分布，那么我们就不是在估计一个固定数量的参数。例如，如果特征是类别型的，我们可能只是计算每个类别中每个特征值的频率。这里的“参数”实际上是整个观察到的频率表，它随着唯一特征值的数量而增长，并且不符合固定、有限集合（如 $\mu$ 和 $\sigma^2$）的定义。这种直接从训练数据中估计，而不假设底层参数化分布的方式，符合非参数的理念。
	
	**总而言之，关键的划分依据是模型是否假设一个固定、有限的参数集来描述数据分布。**
	
	* 如果你对数据分布的**形式**做出假设（例如，高斯分布、伯努利分布、多项式分布），然后估计该假设分布的参数，那么它就是**参数模型**。
	* 如果你直接根据观察到的数据估计概率或进行预测，而不假设具有固定数量参数的特定分布形式，那么它就被认为是**非参数模型**。
	
	朴素贝叶斯之所以独特，是因为它在处理似然函数 $p(\vec{x}_i | y_i = c)$ 方面具有灵活性，这使得它可以在这两个类别之间切换。如果你为似然函数选择一个特定的参数分布（例如高斯朴素贝叶斯），那么它就变成了参数模型。如果你使用非参数密度估计（例如对分类特征直接进行频率计数，或者对连续特征使用核密度估计而不假设特定形式），那么它就倾向于成为非参数模型。

## Linear Multi-class Classification

1. We assume that we use logistic regression, what about we use perceptron?

	1. 无法直接获得概率输出
	
	- 感知机本身是硬分类器，输出的是离散的分类结果（如 0 或 1）
	- 它不提供样本属于某个类别的概率或置信度
	- 因此，幻灯片中提到的 Softmax 归一化步骤无法直接应用，因为感知机中的 $f_k(x_i)$ 不是一个能直接用于 Softmax 转换的“强度分数”
	
	2. 决策冲突
	
	- 对于一个新的样本 $x_i$，多个 One-VS-All 的感知机可能会同时输出 1（表示 $x_i$ 属于它们各自的类别），或者所有感知机都输出 0（表示 $x_i$ 不属于任何类别）
	- 例如，如果 $x_i$ 靠近类别 A 和 B 之间的边界，那么负责区分 A 和非 A 的感知机可能输出 A，负责区分 B 和非 B 的感知机可能输出 B，这将导致模棱两可的分类结果
	- 在逻辑回归中，即使样本位于决策边界附近，其概率输出也会提供一个平滑的过渡，例如 $P(A|x_i)=0.51$，$P(B|x_i)=0.49$，我们仍然可以基于最高的概率做出决策
	
	3. 需要额外的决策规则
	
	- 由于感知机输出的硬性特性，不能简单地通过比较 $f_k(x_i)$ 来做出最终决策
	- 需要设计额外的投票机制或其他启发式规则来解决多重分类（multiple '1' outputs）或无分类（all '0' outputs）的问题
	- 例如，可以采用“多数投票”机制：如果多个分类器都声称样本属于它们的类别，就统计哪个类别获得的“票数”（即有多少个分类器对其输出 1）最多，但这仍然可能导致平局

2. Softmax函数和普通的（加权）归一化函数相比较的优势：

$$
\text{Softmax}(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K
$$

>	- 拥有指数放大机制
>	
>	- 拥有自然的概率解释
>	
>	- 拥有处理负值的能力

## The Kernel Tricks

1. The Representer Theorem.

	> 对于在再生核希尔伯特空间（Reproducing Kernel Hilbert Space, RKHS）上定义的、由损失函数（Loss Function）和正则化项（Regularization Term）组成的优化问题，其最优解可以被表示为训练数据集中输入点上评估的核函数（Kernel Function）的有限线性组合。
	> $$
	> f(\phi(\mathbf x)) = \mathbf w^T\phi(\mathbf x) \to f(\phi(\mathbf x_j)) = \sum_{i=1}^n \alpha_i y_i \phi(\mathbf x_i)^T\phi(\mathbf x_j)
	> $$

2. A kernel matrix satisfies:

	> 1.  **Symmetry:** $K(\mathbf{x}_j, \mathbf{x}_i) = K(\mathbf{x}_i, \mathbf{x}_j)$
	> 2.  **Positive Semi-Definite:** Any one of the following equivalent statements holds:
	>     * $\mathbf{v}^T \mathbf{K} \mathbf{v} \ge 0$, for all $\mathbf{v} \in \mathbb{R}^n$
	>     * **Mercer's Condition:** $\sum_{i}\sum_{j} c_i c_j K(\mathbf{x}_i, \mathbf{x}_j) \ge 0$, for all $c_i, c_j \in \mathbb{R}$

3. Rules for combining valid kernel functions.

	> * **Sum Rule:** If $K_1$ and $K_2$ are valid kernels on the sample space $\mathcal{X}$, then their sum, $K_1 + K_2$, is also a valid kernel on $\mathcal{X}$.
	>
	> * **Scaling Rule:** If $\lambda > 0$ and $K$ is a valid kernel on $\mathcal{X}$, then scaling it by $\lambda$, resulting in $\lambda K$, also produces a valid kernel on $\mathcal{X}$.
	>
	> * **Product Rule:**
	>     * If $K_1$ and $K_2$ are valid kernels on the same sample space $\mathcal{X}$, then their product, $K_1 K_2$, is a valid kernel on $\mathcal{X}$.
	>     * If $K_1$ is a valid kernel on $\mathcal{X}_1$ and $K_2$ is a valid kernel on $\mathcal{X}_2$, then their product, $K_1 K_2$, is a valid kernel on the Cartesian product space $\mathcal{X}_1 \times \mathcal{X}_2$.

4. 证明下列变换是一个可以被使用到核函数中的从2维到3维的有效变换。
$$
  \phi(x1,x2)=(x_1^2,x_2^2,\sqrt{2}x_1x_2)^T
$$

> 要证明一个变换$\phi(x)$可以被使用到核函数中，最直接的方法是找到一个函数$K(x,z)$，使得对于任意的$x,z\in\mathbb{R}^2$，都有$K(x,z)=\phi(x)^T\phi(z)$。由于
> $$
> \phi_(x)^T\phi(z)=(x^Tz)^2
> $$
> 是多项式核函数在$d=2$的形式，故根据Mercer定理，这个核函数是有效的，并且存在一个特征映射$\phi$使得$K(x,z)=\phi(x)^T\phi(z)$。

## Support Vector Machine

在支持向量机（SVM）中，当引入软间隔（Soft Margin）和核方法（Kernel Methods）后，参数 $C$ 和 $\sigma$（对于高斯核，即RBF核）扮演着至关重要的角色，它们共同影响着模型的复杂性、泛化能力以及对训练数据的拟合程度。

1. 参数 $C$ 的作用 (软间隔)

参数 $C$ 是软间隔 SVM 中引入的一个**正则化参数**，它用于**平衡模型的复杂性和训练误差**。

* **定义：** 在软间隔 SVM 的原始问题中，目标函数通常是最小化 $\frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$。这里的 $C$ 惩罚松弛变量 $\xi_i$（即分类错误的样本或位于间隔内的样本）的总和。
    * $\|\mathbf{w}\|^2$ 项代表模型的复杂性（或决策边界的平滑度，与间隔大小成反比）。
    * $\sum_{i=1}^n \xi_i$ 项代表训练误差（误分类的程度）。

* **$C$ 值的大小对模型的影响：**
    * **$C$ 值小 (正则化强度高)：**
        * 模型会更倾向于容忍一些训练误差，让更多的样本落入间隔之内甚至被错误分类，以换取更大的间隔（更简单的决策边界）。
        * 这通常会导致**欠拟合（underfitting）**的风险，但模型对新数据的泛化能力可能更好，对异常值更鲁棒。
        * 相当于降低了对分类错误的惩罚。
    * **$C$ 值大 (正则化强度低)：**
        * 模型会更严格地惩罚训练误差，试图正确分类所有训练样本，甚至包括一些噪声和异常值。
        * 这会导致模型寻找一个更小的间隔，从而可能产生一个更复杂的决策边界，有**过拟合（overfitting）**的风险。
        * 相当于增加了对分类错误的惩罚。

* **总结：** $C$ 控制着“误分类惩罚”和“最大间隔”之间的权衡。选择一个合适的 $C$ 值是确保 SVM 良好泛化性能的关键。

2. 参数 $\sigma$ 的作用 (高斯核/RBF核)

参数 $\sigma$ 是用于**高斯核（Gaussian Kernel）**或**径向基函数（Radial Basis Function, RBF）核**的特有参数。高斯核的定义通常是：
$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)$

* **定义：** $\sigma$ 决定了高斯核的“宽度”或“作用范围”。它衡量了单个训练样本影响力范围的大小。
    * $\|\mathbf{x}_i - \mathbf{x}_j\|^2$ 是两个样本点之间的欧氏距离的平方。
    * $\sigma^2$ 在分母中，所以 $\sigma$ 越大，指数项的绝对值越小，核函数值越大。

* **$\sigma$ 值的大小对模型的影响：**
    * **$\sigma$ 值小 (核函数影响范围小)：**
        * 核函数的值会随着样本距离的增大而迅速衰减到零，这意味着只有非常靠近的样本才会对彼此产生显著影响。
        * 这使得模型倾向于为每个训练样本创建非常局部的决策边界，从而导致一个**高度复杂且可能高度非线性的决策边界**。
        * 有**过拟合（overfitting）**的风险，因为模型可能过度关注每个数据点的细节，对噪声敏感。
    * **$\sigma$ 值大 (核函数影响范围大)：**
        * 核函数的值衰减较慢，这意味着较远的样本也会对彼此产生影响。
        * 这使得模型倾向于形成一个**更平滑、更全局化的决策边界**，类似于线性分类器。
        * 有**欠拟合（underfitting）**的风险，因为模型可能无法捕捉数据中的复杂非线性关系。

* **总结：** $\sigma$ 控制着数据点之间相似度衰减的速度，从而间接控制了特征映射的非线性程度和模型的复杂性。

3. $C$ 和 $\sigma$ 的联合作用

$C$ 和 $\sigma$ 并非独立作用，它们之间存在相互影响：

* **高 $C$ 和小 $\sigma$：** 这种组合通常会导致模型严重过拟合。模型既严格惩罚错误分类，又使用非常局部化的核，试图精确地拟合每一个数据点，包括噪声。
* **低 $C$ 和大 $\sigma$：** 这种组合可能导致欠拟合。模型对错误分类容忍度高，同时核的影响范围大，使得决策边界过于平滑，无法捕捉数据中的复杂模式。
* **理想情况：** 需要通过交叉验证（Cross-Validation）等技术来同时调整这两个参数，以找到一个最佳组合，使得模型在训练数据上表现良好，并且在新数据上具有优秀的泛化能力。

简而言之，$C$ 权衡了训练误差和模型复杂度，而 $\sigma$ 则控制了非线性决策边界的“平滑度”或“局部性”。它们是 SVM 在实际应用中最重要的两个超参数。 
