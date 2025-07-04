# P, NP, NP-Completed, NP-Hard

## 1 多项式时间

1. 我们称可以在多项式时间内解决的问题为可以高效解决的问题；把目前没有看起来未来也不太可能有多项式时间算法的问题称为不可以高效解决的问题。

2. 为什么多项式时间被认为是高效的算法？

	> **Moore定律**：
	>
	> 1965 年，计算机芯片行业的先驱Gordo E. Moore注意到，集成电路上的晶体管数量每两年翻一番，于是他预测这一规律将会持续下去。经过适当的修正，如今这一规律被表述为，每18个月集成电路上的晶体管数量将会翻一番，即著名的Moore定律。
	>
	> 表面上看，这一定律对于多项式时间算法应该是产生反向刺激的。
	>
	> 但是，
	>
	> |   实例   | 1975 | 1985 |       1995       | Present |
	> | :------: | :--: | :--: | :--------------: | :-----: |
	> | $O(2^n)$ |  25  |  31  |        38        |   50    |
	> | $O(n^2)$ |  25  | 2500 | $2.5\times 10^5$ | $10^8$  |
	> | $O(n^6)$ |  25  |  50  |       100        |   350   |
	>
	> 多项式时间算法的能力的提升将是指数级的，而指数时间算法的能力只能以多项式的时间进行提升！

## 2 判定问题

1. 定义：判定问题(Decision Problems) 指的是答案只有0或1的问题。

2. 很多问题都有判定版本和优化版本。

	> **顶点染色**
	>
	> 判定问题：顶点染色
	>
	> 输入：无向图G=(V,E)和整数k。问题：是否可以对其顶点进行k染色？
	>
	> ---
	>
	> 优化问题：色数
	>
	> 输入：无向图G=(V,E)。问题：G的色数，即将其顶点染色使得相邻的顶点颜色不同的最小颜色数。

3. 判定问题的算法经常可以用来解决优化问题。

## 3 P类问题

::: tip 确定性算法

设A是求解问题P的一个算法，如果对于问题P的任何一个实例，在整个执行过程中每一步都只有一种选择，那么我们称A是一个确定性算法，也就是说对于同样的输入，A的输出从来不会被改变。

:::

::: tip P类问题

P 类问题由这样的判定问题组成，其解（是/不是）可以用确定性算法在运行多项式步内，比如$O(n^k)$内得到，这里$k$是一个与$n$无关的常数。

:::

P类问题有一个性质：其补问题也在P类问题中。

::: tip 补问题

设$P$是一个判定问题，其补问题$\bar P$定义为：对于$P$的任何一个实例，其补问题的答案是P 的答案的否定。

:::

## 4 NP类问题

有些判定问题我们可能没有高效的算法去解决，但是我们可以高效地验证一个解是否正确。我们把这样的判定问题称为可以被高效验证的问题。事实上，这就是著名的NP类问题。

::: tip 非确定性算法

对于一个输入x，一个非确定型算法由下列两个阶段组成：

- 猜测阶段：这个阶段，算法任意的产生一个字符串y，这个字符串可能对应输入实例的一个解，也可能什么意义也没有。这一阶段唯一的要求就是，y能在多项式步数内生成，即$O(x^k)$步，其中k是一个与x无关的非负整数。
-  验证阶段：在这个阶段，一个确定性算法验证两件事：
	-  y是否是合法的，即y是否是一个合法的字符串，如果不合法则回答NO。
	- y是否是x的一个解，如果是正确的，则回答YES，否则回答NO。

如果存在一个导致回答YES的猜测y，则这个非确定型算法是正确的。如果整个算法的执行时间是多项式的，则称该算法是一个多项式时间的非确定型算法。

:::

[《算法设计与分析》 - 10-NP完全问题(NP-complete Problem)](https://basics.sjtu.edu.cn/~yangqizhe/pdf/algo2024w/slides/AlgoLec10-handout.pdf)
