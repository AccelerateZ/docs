# 9 聚类

- 聚类是一种经典的**无监督学习**方法，**无监督学习的目标是通过对无标记训练样本的学习，发掘和揭示数据集本身潜在的结构与规律**，即不依赖于训练数据集的类标记信息。聚类则是试图将数据集的样本划分为若干个互不相交的类簇，从而每个簇对应一个潜在的类别。
- 形式化的说，假定样本集$D=\{x_1,x_2,\ldots,x_n\}$包含m个无标记样本， 每个样本都是一个n维特征向量，则聚类算法会将样本集$D$划分为$k$个不相交的簇$\{C_l\mid l\in[1,k]\}$。我们用$\lambda_j\in\{1,2,\ldots,k\}$来表示样本的簇标记(Cluster Label)，即$x_j\in C_l$。于是，聚类的结果可用包含m 个元素的簇标记向量$\mathbf{\lambda}=(\lambda_1;\lambda_2,\ldots,\lambda_n)$。

- 聚类直观上来说是将相似的样本聚在一起，从而形成一个**类簇(Cluster)**。那首先的问题是如何来**度量相似性**(Similarity Measure)呢？这便是**距离度量**，在生活中我们说差别小则相似，对应到多维样本，每个样本可以对应于高维空间中的一个数据点，若它们的距离相近，我们便可以称它们相似。那接着如何来评价聚类结果的好坏呢？这便是**性能度量**，性能度量为评价聚类结果的好坏提供了一系列有效性指标。

## 9.1 无监督学习和聚类

### 无监督学习

无监督学习也称为无监督机器学习，它使用机器学习算法来分析未标记的数据集并进行**聚类**。这些算法无需人工干预，即可发现隐藏的模式或数据分组。

#### 无监督学习的应用

机器学习技术已成为改善产品用户体验和测试系统以保证质量的常用方法。 与手动观察相比，无监督学习提供了查看数据的探索性路径，使企业能够更快地识别大量数据中的模式。 在现实世界中，无监督学习的一些最常见应用如下：

- **新闻栏目：**Google 新闻使用无监督学习，对各种在线新闻媒体关于同一故事的文章进行分类。 例如，可以将总统选举的结果归类到“美国”新闻的标签下。
- **计算机视觉：**无监督学习算法用于视觉感知任务，例如物体识别。  
- **医学成像：**无监督机器学习为医学成像设备提供基本功能，例如图像检测、分类和分割，用于放射学和病理学，可以快速准确地诊断患者病情。
- **异常检测：**无监督学习模型可以梳理大量数据，发现数据集中的非典型数据点。 这些异常现象可以提高人们对故障设备、人为错误或安全违规的认知。
- **客户角色：**通过定义客户角色，可以更轻松地了解共同特征和商业客户的购买习惯。 无监督学习使企业能够建立更完善的买家角色档案，让组织能够更恰当地调整自己的产品讯息传达方式。
- **推荐引擎：**无监督学习可使用过去的购买行为数据，帮助发现相关数据趋势，根据这些趋势可制定出更有效的交叉销售策略。 这用于在线上零售商的结账流程中向客户提供相关的附加建议。

#### 无监督、 有监督与半监督学习的对比

人们经常会将无监督学习和有监督学习一起讨论。 与无监督学习算法不同的是，有监督学习算法使用标记数据。 有监督学习可以通过这些数据来预测未来的结果，或是根据试图解决的回归或分类问题，将数据分配到特定类别。 虽然有监督学习算法往往比无监督学习模型更准确，但有监督学习事先需要人工干预才能恰当地标记数据。 而这些标记数据集能够让有监督学习算法避免计算复杂性，因为不需要大型训练集就能产生预期结果。 常见的回归和分类技术包括线性和逻辑回归、朴素贝叶斯、KNN 算法和随机森林。

如果给定输入数据中只有一部分被标记，就会进行半监督学习。 无监督学习和半监督学习可能是更具吸引力的替代方案，因为依赖领域专业知识为有监督学习恰当标记数据可能既耗时又成本高昂。

可以参见[Supervised vs. unsupervised learning: What's the difference? | IBM](https://www.ibm.com/think/topics/supervised-vs-unsupervised-learning)。

#### 无监督学习面临的难题

虽然无监督学习有很多好处，但在允许机器学习模型无任何人为干预的情况下执行时，可能会遇到一些难题。 其中的一些难题包括：

- 大量训练数据导致的计算复杂性
- 训练时间更长
- 结果不准确的风险较高
- 需要人工干预来验证输出变量
- 数据聚类的基础缺乏透明度

## 9.2 距离度量

### 距离度量的基本性质

对于给定的两个样本点$x_i$和$x_j$，假设它们的特征分别为$x_{i1},x_{i2},\ldots,x_{ip}$和$x_{j1},x_{j2},\ldots,x_{jp}$，则这两个样本点之间的距禽度量可以通过距离函数$d(x_i,x_j)$来度量。距离度量是定义在特征空间中的两个样本点之间的距离，是一个实数，满足下面的性质：

- 非负性：$d(x_i,x_j)\geq 0$；

- 同一性：$d(x_i,x_j)=0$当且仅当$x_i=x_j$；

- 对称性：$d(x_i,x_j)=d(x_j,x_i)$；

- 直递性（三角不等式）：$d(x_i,x_j)\leq d(x_i,x_k)+d(x_k,x_j)$。

### 闵可夫斯基距离
给定样本空间$X$中的两个样本点$x_i$和$x_j$，它们的特征分别为$x_{i1},x_{i2},\ldots,x_{ip}$和$x_{j1},x_{j2},\ldots,x_{jp}$，则这两个样本点之间的闵可夫斯基距离定义为

$$
    d_{mk}(x_i,x_j)=\left(\sum_{u=1}^p|x_{iu}-x_{ju}|^p\right)^{\frac{1}{p}}
$$

对于$p\ge1$时，闵可夫斯基距离满足距离度量的基本性质。

> 上式即为$x_i-x_j$的$L_p$范数$\lVert x_i-x_j\rVert_p$。

- 当$p=1$时，称为曼哈顿距离（Manhattan distance）；

$$
    d_{man}(x_i,x_j)=\sum_{u=1}^p|x_{iu}-x_{ju}|
$$

- 当$p=2$时，称为欧氏距离（Euclidean distance）；

$$
    d_{ed}(x_i,x_j)=\sqrt{\sum_{u=1}^p(x_{iu}-x_{ju})^2}
$$

- 当$p=\infty$时，称为切比雪夫距离（Chebyshev distance）。

$$
    d_{ch}(x_i,x_j)=\max_{u}|x_{iu}-x_{ju}|
$$

> 计算$\lim_{n\to\infty}\sqrt[n]{a_{1}^{n}+a_{2}^{n}+\ldots+a_{k}^{n}}$，其中$a_{i}>0(i=1,2,\ldots,k)$。
>
> 本题考**夹逼定理**的运用
>
> 设$A=\max_{1\leq i\leq k}\left\{ a_{i} \right\}$，则有
>
> 
> $$
>   \sqrt[n]{A^{n}}<\sqrt[n]{a_{1}^{n}+a_{2}^{n}+\ldots+a_{k}^{n}}<\sqrt[n]{kA^{n}}\\
    \sqrt[n]{A^{n}}=A,\lim_{n\to\infty}\sqrt[n]{k}=1
    $$
>
> 由三明治法则可得
> 
> $$
>   \lim_{n\rightarrow\infty}\sqrt[n]{a_{1}^{n}+a_{2}^{n}+\ldots+a_{k}^{n}}=\max_{1\leq i\leq k}\left\{ a_{i} \right\}=A
> $$

### 数据属性和距离

- 我们常常将属性划分为两种类型：连续属性（数值属性，numerical attribute）和离散属性（列名属性，nominal attribute）。

- 对于连续值的属性，一般都可以被学习器所用，有时会根据具体的情形作相应的预处理，例如：归一化等；而对于离散值的属性，需要作下面进一步的处理：

	> （有序属性，ordinal attribute）若属性值之间**存在序关系**，则可以将其转化为连续值，例如：身高属性“高”“中等”“矮”，可转化为{1，0。5，0}。
	> （无序属性，non-ordinal attribute）若属性值之间**不存在序关系**，则通常将其转化为向量的形式，例如：性别属性“男”“女”，可转化为{(1,0)，(0,1)}。

- 在进行距离度量时，易知**连续属性和存在序关系的离散属性都可以直接参与计算**，因为它们都可以反映一种程度，我们称其为“**有序属性**”。这时，可以使用闵可夫斯基距离直接计算。

- 而对于无序属性，我们一般采用VDM(Value Difference Metric)进行距离的计算。

- 令$m_{u, a}$表示在属性$u$上取值为$a$的样本数，$m_{u, a, i}$表示在第$i$个样本簇中在属性$u$上取值为$a$的样本数，$k$为样本簇数，则属性$u$ 上两个离散值$a$和$b$之间的VDM距离为

	$$
	    \text{VDM}_p(a,b)=\sum^k_{i=1}\Bigg|\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}\Bigg|^p
	$$

	样本类别已知时$k$通常设置为类别数。

- 将闵可夫斯基聚类和VDM结合即可处理混合属性。

- 假定有$n_c$个有序属性、$n-n_c$个无序属性，不失一般性，令有序属性排列在无序属性之前，则

	$$
	    \text{MinkovDM}_p(\pmb{x}_i, \pmb{x}_j)=\Bigg(\sum^{n_c}_{u=1}|x_{iu}-x_{ju}|^p+\sum^n_{u=n_c+1}\text{VDM}_p(x_{iu},x_{ju})\Bigg)^\frac{1}{p}
	$$


- 当样本空间中不同属性的重要性不同时，可使用加权距离(weighted distance)。 以加权闵可夫斯基距离为例:

	$$
	    \text{dist}_\text{wmk}(\pmb{x}_i, \pmb{x}_j)=(w_1|x_{iu}-x_{ju}|^p+\ldots+w_n|x_{nu}-x_{nu}|^p)^{\frac{1}{p}}
	$$

	其中权重$w_i\geq0(i=1,2,\ldots，n)$表征不同属性的重要性，通常$\sum^n_{i=1}w_i=1$。

- 通常我们是基于某种形式的距离来定义相似度度量(similarity measure)，距离越大，相似度越小。

- 相似度度量的距离未必一定要满足距离度量的所有基本性质，尤其是直递性。

- 不满足直递性的距离称为非度量距离(non-metric distance)。

- 在现实任务中，也可基于数据样本来确定合适的距离计算式，这可通过距离度量学习(distance metric learning)来实现。

## 9.3 性能度量

* 聚类性能度量亦称聚类有效性指标(validity index)；与监督学习中的性能度量作用类似。

* 对聚类结果，我们需通过某种性能度量来评估其好坏。

* 明确最终将要使用的性能度量，则可直接将其作为聚类过程的优化目标。

* 同一簇的样本尽可能彼此相似，不同簇的样本尽可能不同。 聚类结果的簇内相似度(intra-cluster similarity)高且簇间相似度(inter-cluster similarity)低。

* 聚类性能度量大致有两类。 

	1. 将聚类结果与某个参考模型(reference model)进行比较，称为外部指标(external index)。
		* 可以将领域专家给出的划分结果作为参考模型。
	2. 直接考察聚类结果而不用任何参考模型，称为内部指标(internal index)。

* 对数据集$D=\{\pmb{x}_1,\pmb{x}_2,\ldots,\pmb{x}_m\}$，假定通过聚类给出的簇划分为$C=\{C_1,C_2,\ldots,C_k\}$，参考模型给出的簇划分为$C^*=\{C^*_1,C^*_2,\ldots,C^*_s\}$。

	* 通常$k \neq s$。

	* 令$\pmb\lambda$与$\pmb\lambda^*$分别表示与$C$和$C^*$对应的簇标记向量。

	* 将样本两两配对考虑，定义
  
    $$
		\begin{equation}
			\begin{aligned}
		    a&=|SS|,\quad SS=\{(\pmb{x}_i, \pmb{x}_j)|\lambda_i=\lambda_j, \lambda_i^*=\lambda_j^*,i<j\}, \\
		    b&=|SD|,\quad SD=\{(\pmb{x}_i, \pmb{x}_j)|\lambda_i=\lambda_j, \lambda_i^*\neq\lambda_j^*,i<j\}, \\
		    c&=|DS|,\quad DS=\{(\pmb{x}_i, \pmb{x}_j)|\lambda_i\neq\lambda_j, \lambda_i^*=\lambda_j^*,i<j\}, \\
		    d&=|DD|,\quad DD=\{(\pmb{x}_i, \pmb{x}_j)|\lambda_i\neq\lambda_j, \lambda_i^*\neq\lambda_j^*,i<j\}, \\
			\end{aligned}
		\end{equation}
    $$

其中集合$SS$包含了在$C$中隶属于相同簇且在$C^*$中也隶属于相同簇的样本对，集合$SD$包含了在$C$中隶属于相同簇但在$C^*$中隶属于不同簇的样本对，由于每个样本对$(\pmb{x}_i，\pmb{x}_j)(i<j)$仅能出现在一个集合中，因此有$a+b+c+d=m(m-1)/2$成立。

* 基于上式可以导出常用的聚类性能度量外部指标: 

	* Jaccard系数(Jaccard Coefficient，简称$\text{JC}$)
  
    $$
		JC=\frac{a}{a+b+c}
    $$

	* FM指数(Folkeds and Mallows Index，简称$\text{FMI}$)
  
    $$
		FMI=\sqrt{\frac{a}{a+b}·\frac{a}{a+c}}
    $$

	* Rand指数(Rand Index，简称$\text{RI}$)
  
    $$
		RI=\frac{2(a+d)}{m(m-1)}
    $$

上述性能度量的结果值均在$[0，1]$区间，值越大越好。

* 考虑聚类结果的簇划分为$C=\{C_1,C_2,\ldots,C_k\}$，定义
  
$$
	\begin{equation}
		\begin{aligned}
			\text{avg}(C)&=\frac{2}{|C|(|C|-1)}\sum_\nolimits{{1\leq i<j\leq|C|}}\text{dist}(\pmb{x}_i, \pmb{x}_j), \\
			\text{diam}(C)&=\max_\nolimits{{1\leq i<j\leq|C|}}\text{dist}(\pmb{x}_i, \pmb{x}_j), \\
			\text d_\min(C_i, C_j)&=\min_\nolimits{\pmb{x}_i\in C_i, \pmb{x}_j\in C_j}\text{dist}(\pmb{x}_i, \pmb{x}_j), \\
			\text d_{\text{cen}}(C_i, C_j)&=\text{dist}(\pmb{\mu}_i, \pmb{\mu}_j),
		\end{aligned}
	\end{equation}
$$

$\text{dist}(·,·)$用于计算两个样本之间的距离，距离越大则样本的相似度越低； $\pmb{\mu}$代表簇$C$的中心点$\pmb{\mu}=\frac{1}{|C|}\sum_\nolimits{1\leq i\leq|C|}\pmb{x}_i$。 

* $\text{avg}(C)$对应于簇$C$内样本间的平均距离。

* $\text{diam}(C)$对应于簇$C$内样本间的最远距离。

* $d_\min(C_i,C_j)$对应于簇$C_i$与簇$C_j$最近样本间的距离。

* $d_\text{cen}(C_i,C_j)$对应于簇$C_i$与簇$C_j$中心点间的距离。

	---

	

* 基于上式可导出常用的聚类性能度量的内部指标:

	* DB指数(Davies-Bouldin Index，简称$\text{DBI}$)
  
    $$
		DBI=\frac{1}{k}\sum^k_{i=1}\max_{j\neq i}\Bigg(\frac{\text{avg}(C_i)+\text{avg}(C_j)}{d_{\text{cen}}(C_i, C_j)}\Bigg)
    $$

	* Dunn指数(Dunn Index，简称$\text{DI}$)
  
    $$
		DI=\min_{1\leq i\leq k}\Bigg\{\min_{j\neq i}\Bigg(\frac{d_\min(C_i, C_j)}{\max_\nolimits{1\leq l \leq k}\text{diam}(C_l)}\Bigg)\Bigg\}
    $$

显然，DBI的值越小越好，DI的值越大越好。
        
## 9.4 原型聚类

* 原型聚类亦称基于原型的聚类(prototype-based clustering), 此类算法假设聚类结构能通过一组原型刻画.
* 原型是指样本空间中具有代表性的点.
* 通常算法先对原型进行初始化, 然后对原型进行迭代更新求解.
* 采用不同的原型表示, 不同的求解方式, 将产生不同的算法.

### 9.4.1.k均值算法

* 给定样本集 $D=\{\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_m\}$, $k$ 均值($k$-means)算法针对聚类所得簇划分$C=\{C_1, C_2, ..., C_k\}$最小化平方误差
  
	$$
		E=\sum^k_{i=1}\sum_{x\in C_i}||\pmb{x}-\pmb{\mu}_i||^2_2
	$$

	其中$\pmb{\mu}_i=\frac{1}{|C_i|}\sum_{\pmb{x}\in C_i}\pmb{x}$是簇$C_i$的均值向量.

	* 上式在一定程度上刻画了簇内样本围绕簇均值向量的紧密程度, $E$值越小则簇内样本相似度越高.

* 最小化上式要找到它的最优解需考察样本集$D$所有可能的簇划分, 这是一个$\text{NP}$难问题.

* $k$ 均值算法采用了贪心策略, 通过迭代优化来近似求解上式.

* $k$ 均值算法：

	![](https://img2.imgtp.com/2024/05/29/kYYE9SVd.png)

* 为避免运行时间过长, 通常设置一个最大运行轮数或最小调整幅度阈值, 若达到最大轮数或调整幅度小于阈值, 则停止运行.

### 9.4.2.学习向量量化

* 与$k$均值算法类似, 学习向量量化(Learning Vector Quantization, 简称LVQ)也是试图找到一组原型向量来刻画聚类结构.

* 与一般聚类算法不同的是, LVQ假设数据样本带有类别标记, 学习过程利用样本的这些监督信息来辅助聚类.

* LVQ可看作通过聚类来形成类别子类结构, 每个子类对应一个聚类簇.

* 给定样本集$D=\{(\pmb{x}_1, y_1), (\pmb{x}_2, y_2), ...,(\pmb{x}_m, y_m)\}$, 每个样本$\pmb{x}_j$是由$n$个属性描述的特征向量$(x_{j1}; x_{j2}; ... ;x_{jn})$, $y_j\in \mathcal{Y}$是样本$\pmb{x}_j$的类别标记.

* LVQ的目标是学得一组$n$维原型向量$\{\pmb{p}_1, \pmb{p}_2, ...,\pmb{p}_q\}$, 每个原型向量代表一个聚类簇, 簇标记$t_i\in\mathcal{Y}$.

* LVQ算法描述：

	![](https://img2.imgtp.com/2024/05/29/AVWHXyKK.png)

	* LVQ算法对原型向量进行初始化, 例如对第$q$个簇可从类别标记为$t_q$的样本中随机选取一个作为原型向量.
* 在每一轮迭代中, 算法随机选取一个有标记训练样本, 找出与其最近的原型向量, 并根据两者的类别标记是否一致来对原型向量进行相应的更新.
	* 算法的停止条件可设置为最大运行轮数或原型向量更新幅度很小.

* LVQ的关键是如何更新原型向量.

	* 对样本$\pmb{x}_j$, 若最近的原型向量$\pmb{p}_{i^*}$与$\pmb{x}_j$的类别标记相同, 则令$\pmb{p}_{i^*}$向$\pmb{x}_j$的方向靠拢.
  
	$$
		\pmb{p}'=\pmb{p}_{i^*}+\eta·(\pmb{x}_j-\pmb{p}_{i^*})
	$$

	* $\pmb{p}’$与$\pmb{x}_j$之间的距离为
  
	$$
		\begin{equation}
		\begin{aligned}
			||\pmb{p}’-\pmb{x}_j||_2&=||\pmb{p}_{i^*}+\eta·(\pmb{x}_j-\pmb{p}_{i^*})-\pmb{x}_j||_2\\
			&=(1-\eta)·||\pmb{p}_{i^*}-\pmb{x}_j||_2
		\end{aligned}
		\end{equation}
	$$

	令学习率$\eta\in(0, 1)$, 则原型向量$\pmb{p}_{i^*}$在更新为$\pmb{p}'$之后将更接近$\pmb{x_j}$.

	* 若$\pmb{p}_{i^*}$与$\pmb{x}_j$的类别标记不同, 则更新后的原型向量与$\pmb{x}_j$之间的距离将增大为$(1+\eta)·||\pmb{p}_{i^*}-\pmb{x}_j||_2$从而更远离$\pmb{x}_j$.

* 在学得一组原型向量$\{\pmb{p}_1, \pmb{p}_2, ...,\pmb{p}_q\}$后, 即可实现对样本空间$\mathcal{X}$的簇划分.

* 对任意样本$\pmb{x}$, 它将被划入与其距离最近的原型向量所代表的簇中.

* 每个原型向量$\pmb{p}_i$定义了与之相关的一个区域$R_i$, 该区域中每个样本与$\pmb{p}_i$的距离不大于它与其他原型向量$\pmb{p}_{i'}(i'\neq i)$的距离, 即
  
	$$
		R_i=\{\pmb{x}\in\mathcal{X}|\ ||\pmb{x}-\pmb{p}_i||_2\leqslant||\pmb{x}-\pmb{p}_{i'}||_2, i'\neq i\}
	$$

	* 由此形成了对样本空间$\mathcal{X}$的簇划分$\{R_1, R_2, ..., R_q\}$, 该划分通常称为Voronoi剖分(Voronoi tessellation).
	* 若将$R_i$中样本全用原型向量$\pmb{p}_i$表示, 则可实现数据的有损压缩(lossy compression). 这称为向量量化(vector quantization).

### 9.4.3.高斯混合聚类

* 与$k$均值、LVQ用原型向量来刻画聚类结构不同, 高斯混合(Mixture-of-Gaussian)聚类采用概率模型来表达聚类原型.

* (多元)高斯分布的定义. 对$n$维样本空间$\mathcal{X}$中的随机向量$\pmb{x}$, 若$\pmb{x}$若服从高斯分布, 其概率密度函数为

	$$
		p(\pmb{x})=\frac{1}{(2\pi)^\frac{n}{2}|\pmb{\tiny{\sum}}|^\frac{1}{2}}e^{-\frac{1}{2}(\pmb{x}-\pmb{\mu})^T\pmb{\tiny{\sum}}^{-1}(\pmb{x}-\pmb{\mu})}
	$$

	* 其中$\pmb{\mu}$是$n$维均值向量, $\pmb{\sum}$是的$n\times n$协方差矩阵.
	* 记为$\pmb{x}\sim\mathcal{N}(\pmb{\mu}, \pmb{\sum})$.
	* $\pmb{\sum}$: 对称正定矩阵; $|\pmb{\sum}|$: $\pmb{\sum}$的行列式; $\pmb{\sum}^{-1}$: $\pmb{\sum}$的逆矩阵.
	* 高斯分布完全由均值向量$\pmb{\mu}$和协方差矩阵$\pmb{\sum}$这两个参数确定.

* 为了明确显示高斯分布与相应参数的依赖关系, 将概率密度函数记为$p(\pmb{x}|\pmb{\mu}, \pmb{\tiny{\sum}})$.

* 高斯混合分布的定义
  
	$$
		p_{\mathcal{M}}(\pmb{x})=\sum^k_{i=1}\alpha_i·p(\pmb{x}|\pmb{\mu}_i,\pmb{\tiny{\sum}}_i)
	$$

	* $p_{\mathcal{M}}(·)$也是概率密度函数, $\int p_{\mathcal{M}}(\pmb{x})d\pmb{x}=1$.
	
	* 该分布是由$k$个混合分布组成, 每个混合成分对应一个高斯分布.
	
	* 其中$\pmb{\mu}_i$与$\pmb{\sum}_i$是第$i$个高斯混合分布的参数, 而$\alpha_i>0$为相应的混合系数(mixture coefficient), $\sum^k_{i=1}\alpha_i=1$.
	
	* 假设样本的生成过程由高斯混合分布给出: 首先, 根据$\alpha_1,\alpha_2,..., \alpha_k$定义的先验分布选择高斯混合成分, 其中$\alpha_i$为选择第$i$个混合成分的概率; 然后, 根据被选择的混合成分的概率密度函数进行采样, 从而生成相应的样本.
	
	* 若训练集$D=\{\pmb{x}_1, \pmb{x}_2, ..., \pmb{x}_m\}$由上述过程生成, 令随机变量$z_j\in\{1,2, ..., k\}$表示生成样本$\pmb{x}_j$的高斯混合分布, 其取值未知. $z_j$的先验概率$P(z_j=i)$对应于$\alpha_i(i=1,2,...,k)$.
	
	* 根据贝叶斯定理, $z_j$的后验分布对应于
  
	$$
		\begin{equation}
			\begin{aligned}
				p_\mathcal{M}(z_j=i|\pmb{x}_j)&=\frac{P(z_j=i)·p_\mathcal{M}(\pmb{x}_j|z_j=i)}{p_\mathcal{M}(\pmb{x}_j)}\\
				&=\frac{\alpha_i·p(\pmb{x}_j|\pmb{\mu}_i,\pmb{\sum}_i)}{\sum\limits^k_{l=1}\alpha_l·p(\pmb{x}_j|\pmb{\mu}_l,\pmb{\mathcal{\sum}}_l)}
			\end{aligned}
		\end{equation}
	$$

	换言之, $p_\mathcal{M}(z_j=i|\pmb{x}_j)$给出了样本$\pmb{x}_j$由第$i$个高斯混合成分生成的后验概率. 为方便叙述, 将其简记为$\gamma_{ji}\ (i=1, 2, ..., k)$.
	
	* 当高斯混合分布已知时, 高斯混合聚类将把样本集$D$划分为$k$个簇$C=\{C_1, C_2, ..., C_k\}$, 每个样本$\pmb{x}_j$的簇标记$\lambda_j$如下确定：$\lambda_j=\arg\max_\limits{i\in\{1,2,...,k\}}\ \gamma_{ji}$
  
	从原型聚类的角度来看, 高斯混合聚类是采用概率模型(高斯分布)对原型进行刻画, 簇划分则由原型对应后验概率确定.
	
	* 对于高斯混合分布的定义, 模型参数$\{(\alpha_i, \pmb{\mu}_i, \pmb{\sum}_i)|1\leqslant i\leqslant k\}$, 在给定样本集$D$的求解, 可采用极大似然估计, 即最大化(对数)似然
	
	$$
		\begin{equation}
			\begin{aligned}
  			LL(D)&=\ln\Bigg(\prod^m_{j=1}p_\mathcal{M}(\pmb{x}_j)\Bigg)\\
				&=\sum^m_{j=1}\ln\bigg(\sum^k_{i=1}\alpha_i·p(\pmb{x}_j|\pmb{\mu}_i, \sum_i)\bigg)
			\end{aligned}
		\end{equation}
	$$
	
	采用EM算法进行迭代优化求解.
	
	* 若参数$\{(\alpha_i, \pmb{\mu}_i, \pmb{\sum}_i)|1\leqslant i\leqslant k\}$ 能使上式最大化, 则$\frac{\partial LL(D)}{\partial\pmb{\mu}_i}=0$有

	$$
		\sum^m_{j=1}\frac{\alpha_i·p(\pmb{x}_j|\pmb{\mu}_i,\sum_i)}{
		\sum^k_{l=1}\alpha_l·p(\pmb{x}_j|\pmb{\mu}_l,\sum_l)
  	}(\pmb{x}_j-\pmb{\mu}_i)=0
	$$
	
	* 由$p_\mathcal{M}(z_j=i|\pmb{x}_j)=\frac{\alpha_i·p(\pmb{x}_j|\pmb{\mu}_i,\pmb{\sum}_i)}{\sum\limits^k_{l=1}\alpha_l·p(\pmb{x}_j|\pmb{\mu}_l,\pmb{\mathcal{\sum}}_l)}$以及, $\gamma_{ji}=p_\mathcal{M}(z_j=i|\pmb{x}_j)$, 有
	
	$$
		\pmb{\mu}_i=\frac{\sum\limits^m_{j=1}\gamma_{ji}\pmb{x}_j}{\sum\limits^m_{j=1}\gamma_{ji}}
	$$

	即各混合成分的均值可通过样本加权平均来估计, 样本权重是每个样本属于该成分的后验概率. 
	
	* 类似的, 由$\frac{\partial LL(D)}{\partial\sum_i}=0$可得

	$$
		\sum_\nolimits i=\frac{\sum\limits^m_{j=1}\gamma_{ji}(\pmb{x}_j-\pmb{\mu}_i)(\pmb{x}_j-\pmb{\mu}_i)^T}{\sum\limits^m_{j=1}\gamma_{ji}}
	$$

* 对于混合系数$\alpha_i$, 除了要最大化$LL(D)$, 还需满足$\alpha_i\geqslant 0$, $\sum^k_{i=1}\alpha_i=1$.

* 考虑$LL(D)$的拉格朗日形式：
$$
LL(D)+\lambda\left(\sum_{i=1}^k\ \alpha_i-1\right)
$$

其中$\lambda$为拉格朗日乘子, 由上式对$\alpha_i$的导数为0, 有

$$
\sum^m_{j=1}\frac{p(x_j|\pmb\mu_i,\sum_i)}{\sum\limits^k_{l=1}\alpha_l·p(x_j|\pmb\mu_l,\sum_l)}+\lambda=0
$$

两边同乘以$\alpha_i$, 对所有混合成分求和可知$\lambda=-m$, 有

$$
\alpha_i=\frac{1}{m}\sum^m_{j=1}\gamma_{ji}
$$

即每个高斯成分的混合系数由样本属于该成分的平均后验概率确定.
	
* 即上述推导即可获得高斯混合模型的EM算法: 在每步迭代中, 先根据当前参数来计算每个样本属于每个高斯成分的后验概率$\gamma_{ji}$ (E步), 再根据$\pmb{\mu}_i=\frac{\sum^m_{j=1}\gamma_{ji}\pmb{x}_j}{\sum^m_{j=1}\gamma_{ji}}$, $\sum_i=\frac{\sum^m_{j=1}\gamma_{ji}(\pmb{x}_j-\pmb{\mu}_i)(\pmb{x}_j-\pmb{\mu}_i)^T}{\sum^m_{j=1}\gamma_{ji}}$和$\alpha_i=\frac{1}{m}\sum^m_{j=1}\gamma_{ji}$更新模型参数$\{(\alpha_i,\pmb{\mu}_i,\sum_i)|1\leqslant i\leqslant k\}$ (M步).

* 高斯混合聚类算法描述

    ![19.png](https://i.loli.net/2018/10/18/5bc84fb9c4fa4.png)

    * 第3-5行EM算法的E步, 第6-11行EM算法的M步.
    
* 算法的停止条件可设置为最大迭代轮数或似然函数$LL(D)$增长很少甚至不再增长, 第14-17行根据高斯混合分布确定簇划分.

## 9.5.密度聚类

* 密度聚类亦称基于密度的聚类(density-based clustering), 此类算法假设聚类结构能通过样本分布的紧密程度确定.

* 密度聚类算法从样本密度的角度来考虑样本之间的可连接性, 并基于可连接样本不断扩展聚类簇以获得最终的聚类结果.

* DBSCAN(Density-Based Spatial Clustering of Applications with Noise)是一种著名的密度聚类算法, 它基于一组邻域(neighborhood)参数$(\epsilon, MinPts)$来刻画样本分布的紧密程度.

* 给定数据集$D=\{\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_m\}$, 定义下面这几个概念:

	* $\epsilon$-邻域: 对$\pmb{x}_j\in D$, 其$\epsilon$-邻域包含样本集$D$中与$\pmb{x}_j$的距离不大于$\epsilon$的样本, 即$N_\epsilon(\pmb{x}_j)=\{\pmb{x}_i\in D|\text{dist}(\pmb{x}_i,\pmb{x}_j)\leqslant\epsilon\}$;
	* 核心对象(core object): 若$\pmb{x}_j$的$\epsilon$-领域至少包含$MinPts$个样本, 即$|{N_\epsilon(\pmb{x}_j)}|\geqslant MinPts$, 则是一个核心对象$\pmb{x}_j$;
	* 密度直达(directly density-reachable): 若$\pmb{x}_j$位于$\pmb{x}_i$的$\epsilon$-领域中, 且$\pmb{x}_i$是核心对象, 则称$\pmb{x}_j$由$\pmb{x}_i$密度直达;
		* 密度直达关系通常不满足对称性.
	* 密度可达(density-reachable): 对$\pmb{x}_i$与$\pmb{x}_j$, 若存在样本序列$\pmb{p}_1, \pmb{p}_2, ..., \pmb{p}_n$, 其中$\pmb{p}_1=\pmb{x}_i$, $\pmb{p}_n=\pmb{x}_j$且$\pmb{p}_{i+1}$由$\pmb{p}_i$密度直达, 则称$\pmb{x}_j$由$\pmb{x}_i$密度可达.
		* 密度可达关系满足直递性, 但不满足对称性.
	* 密度相连(density-connected): 对$\pmb{x}_i$与$\pmb{x}_j$, 若存在$\pmb{x}_k$使得$\pmb{x}_i$与$\pmb{x}_j$均由$\pmb{x}_k$密度可达, 则称$\pmb{x}_i$与$\pmb{x}_j$密度相连.
		* 密度相连关系满足对称性.

* DBSCAN将簇定义为: 有密度可达关系导出的最大的密度相连样本集合.

	* $D$中不属于任何簇的样本被认为是噪声(noise)或者异常(anomaly)样本.
	* 给定邻域参数$(\epsilon, MinPts)$, 簇$C\subseteq D$是满足以下性质的非空样本子集:
		* 连接性(connectivity): $\pmb{x}_i\in C$, $\pmb{x}_j\in C\Rightarrow\pmb{x}_i$与$\pmb{x}_j$密度相连
		* 最大性(maximality): $\pmb{x}_i\in C$, $\pmb{x}_j$由$\pmb{x}_i$密度可达 $\Rightarrow\pmb{x}_j\in C$

* 若$\pmb{x}$为核心对象, 由$\pmb{x}$密度可达的所有样本组成的集合记为$X=\{\pmb{x}'\in D|\pmb{x}'$ 由 $\pmb{x}$ 密度可达$\}$, 则可证明$X$即为满足连续性和最大性的簇.

* DBSCAN 算法任选数据集中的一个核心对象为种子(seed), 再由此出发确定相应的聚类簇.

* DBSCAN 算法描述

	![22.png](https://i.loli.net/2018/10/18/5bc8509feb587.png)

## 9.6.层次聚类

* 层次聚类(hierarchical clustering)试图在不同层次对数据集进行划分, 从而形成树形的聚类结构.

* 数据集的划分可采用自底向上的聚合策略, 也可采用自顶向下的拆分策略.

* AGNES(AGglomerative NESting)是一种采用自底向上聚合策略的层次聚类算法.

	* 先将数据集中的每个样本看作一个初始聚类簇, 然后在算法运行的每一步中找出距离最近的两个聚类簇进行合并, 该过程不断重复, 直至达到预设的聚类簇个数.

* AGNES的关键是如何计算聚类簇之间的距离.

	* 每个簇是一个样本集合, 只需采用关于集合的某种距离即可. 
	* 集合间的距离计算常采用豪斯多夫距离(Hausdorff distance).
	* 给定聚类簇$C_i$与$C_j$可通过下面的式子来计算距离:
		* 最小距离: $d_\min(C_i, C_j)=\min\limits_{\pmb{x}\in C_i, \pmb{z}\in C_j}\text{dist}(\pmb{x}, \pmb{z})$
		* 最大距离: $d_\max(C_i, C_j)=\max\limits_{\pmb{x}\in C_i, \pmb{z}\in C_j}\text{dist}(\pmb{x}, \pmb{z})$
		* 平均距离: $d_\text{avg}(C_i, C_j)=\frac{1}{|C_i||C_j|}\sum\limits_{\pmb{x}\in C_i}\sum\limits_{\pmb{z}\in C_j}\text{dist}(\pmb{x}, \pmb{z})$
	* 最小距离由两个簇的最近样本决定, 最大距离由两个簇的最远样本决定, 而平均距离则由两个簇的所有样本共同决定.
	* 当聚类簇距离由$d_\min$、$d_\max$或$d_\text{avg}$计算时, AGNES 算法被相应地称为单链接(single-linkage)、全链接(complete-linkage)或均链接(average-linkage)算法.

* AGNES 算法描述

	![26.png](https://i.loli.net/2018/10/18/5bc8509f9d4a0.png)

	* $d$ 通常使用$d_\min$, $d_\max$, $d_\text{avg}$.
* $i^*<j^*$.