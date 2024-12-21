# Stirling公式

## Stirling公式

- 斯特林公式（英语：Stirling's formula）是一条用来取n阶乘近似值的数学公式。一般来说，当n很大的时候，n阶乘的计算量十分大，所以斯特林公式十分好用，而且，即使在n很小的时候，斯特林公式的取值已经十分准确。

- 斯特林公式为

$$
   n!\approx\sqrt{2\pi n}\left(\frac{n}{e}\right)^n
$$

- 更加精确地，有

$$
   \begin{aligned}
         \lim_{n\to\infty}\frac{n!}{\sqrt{2\pi n}\left(\dfrac{n}{e}\right)^n}=1\\
   \lim_{n\to\infty}\frac{e^n\cdot n!}{n^n\cdot\sqrt{n}}=\sqrt{2\pi}
   \end{aligned}
$$

- 事实上，在高等数学中，只需要知道下式即可。上述的内容，不需要引入Wallis公式便能得到结论。

$$
   \lim_{n\to\infty}\frac{n!}{\sqrt{n}\cdot\left(\dfrac{n}{e}\right)^n}=c, c\in\mathbb{R}^+
$$

## Stirling公式的推导

参见[Stirling公式的一些思考 - 知乎 ](https://zhuanlan.zhihu.com/p/143992660)

#### 阶乘大小估计——积分的矩形公式

我们先从 n! 大小的估计出发. 阶乘实际为累乘，数学上对累乘处理的工具十分少，一般都会选择取对数转换为累加，然后借助已有工具处理。

$$
   n! = \prod_{k=1}^nk=\exp\left(\sum_{k=1}^n\log k\right).
$$

因此只需要估计 $\sum\limits_{k=1}^n\log k$的值就可以了。而对于这种简单累加，很容易想到积分。因此考虑函数 $\log(x)$ 在$[1,n]$的积分与累加的大小比较。由于$\log(x)$ 是单调递增的函数，所以在任意的 $[k,k+1]$区间上有

$$
   \log(k)<\log(x)<\log(k+1),\quad k<x<k+1.
$$

两边从$k$到$k+1$积分，得到不等式

$$
   \log(k)<\int_{k}^{k+1}\log(x)\mathrm dx<\log(k+1), \quad \forall k\in \mathbb N.
$$

对 k 从 1 到 n 这 n 个不等式进行累加得到

$$
   \sum_{k=1}^{n}\log(k)<\int_{1}^{n+1}\log(x)\mathrm dx<\sum_{k=1}^{n+1}\log(k), \quad \forall n\in \mathbb N.
$$

即

$$
   \sum_{k=1}^{n}\log(k)<(n+1)\log(n+1)-n<\log(n+1)+\sum_{k=1}^{n}\log(k), \quad \forall n\in \mathbb N.
$$

整理后可有

$$
   n\log(n+1)-n<\sum_{k=1}^{n}\log(k)<(n+1)\log(n+1)-n, \quad \forall n\in \mathbb N.
$$

因此

$$
   \dfrac{(n+1)^n}{e^n}<n!<\dfrac{(n+1)^{n+1}}{e^n}, \quad \forall n \in \mathbb N.
$$

这已经是一个初步的比较好看的结果了. 但精度还是不够, 得到这个公式本质上是用矩形公式逼近 $\log(x)$ 的积分. 我们知道, $\log(x)$是一个单调递增的凹函数, 如果改用梯形公式, 则精度会提高很多.

#### 精细化结果——梯形公式

在 $[k,k+1]$ 上, 由于$\log(x)$ 是一个凹函数, 即函数图像在割线之上, 因此

$$
   \log(x)>\dfrac{\log(k+1)-\log(k)}{(k+1)-k}(x-k)+\log(k),\quad k<x<k+1.
$$

两边积分得到

$$
   \int_k^{k+1}\log(x)\mathrm dx > \dfrac12\left( \log(k)+\log(k+1) \right).
$$

对 $k$ 累加得到

$$
   \int_1^n\log(x)\mathrm dx > \sum_{k=1}^n\log(k)-\frac12\log(n),
$$

即

$$
   n\log(n)-n+1 > \sum_{k=1}^n\log(k)-\frac12\log(n),
$$

整理后再取回指数就可以得到

$$
   0<\dfrac{n!}{\sqrt{n}\left( \frac{n}{e} \right)^n}<e.
$$

我们记

$$
   a_n=\dfrac{n!}{\sqrt{n}\left( \frac{n}{e} \right)^n},
$$

则 $a_n$是一个有界数列, 只需要 $a_n$是单调的就可以知道$a_n$极限存在. 下面考察 $a_n$的单调性.

$$
   \begin{aligned} \dfrac{a_{n+1}}{a_n}&=\dfrac{e}{\left( 1+\frac 1n \right)^{n+\frac 12}}\\ &<\exp\left( 1-\left(n+\frac12\right)\log\left( 1+\frac 1n \right) \right)\\ &< \exp\left( 1- \left(n+\frac12\right)\left( \frac 1n-\dfrac{1}{2n^2}+\dfrac{1}{3n^3} \right)\right)\\ &\le e^{-\tfrac{1}{12n^2}}\\ &<1. \end{aligned}
$$

因此 $a_{n+1}<a_n$，$a_n$为单调递减的数列, 极限存在. 接下来只需要证明, 这个极限不是0就可以了. 这里回溯一下我们得到阶乘估计的方法: 积分的梯形公式. 因此需要估计梯形公式的误差.

#### 梯形公式的误差估计——正项级数收敛判别法

设$$a_n\to c$$, 把阶乘表示为

$$
   n!=a_n\sqrt{n}\left( \frac ne \right)^n,
$$

取对数后为

$$
   \sum_{k=1}^n\log(k)=\log(a_n)+n\log(n)-n+\frac 12\log(n),
$$

即

$$
   1-\log(a_n)=\int_1^n\log(x)\mathrm dx-\left(\sum_{k=1}^n\log(k)-\frac 12\log(n)\right),
$$

记 $b_n=1-\log(a_n) , c_n=b_n-b_{n-1} (b_0=0)$ , 则此为梯形公式的误差公式, 那么

$$
   \lim_{n\to\infty}a_n=0\Leftrightarrow \lim_{n\to\infty}b_n=+\infty \Leftrightarrow \sum_{n=1}^\infty c_n=+\infty.
$$

这就回归到了正项级数是否发散的判别上了. 这里

$$
   c_n=\int_{n-1}^n \left( \log(x)-\dfrac{\log(n)-\log(n-1)}{n-(n-1)}(x-n)+\log(n)\right)\mathrm dx,
$$

记

$$
   g_n(x)=\log(x)-\dfrac{\log(n)-\log(n-1)}{n-(n-1)}(x-n)+\log(n),
$$

则$g_n(n-1)=g_n(n)=0$。注意到 $g^{\prime\prime}_n(x)=-\dfrac{1}{x^2}<0$ , 因此$g_n(x)$为凹函数, 函数图象在任意一条切线的下方, 因此

$$
   g_n(x)<g^\prime_n(n)(x-n)=\left( \frac 1n+\log\left( 1-\frac 1n \right) \right)(x-n).
$$

这样我们可以给出梯形公式误差的一个估计:

$$
   c_n=\int_{n-1}^n g_n(x)\mathrm dx <\int_{n-1}^ng^\prime_n(n)(x-n)\mathrm dx=-\frac 12\left(\frac 1n+ \log\left( 1-\frac 1n \right) \right).
$$

由Taylor公式的Lagrange余项,

$$
   \log\left( 1-\frac 1n \right)=-\frac 1n-\frac{1}{2n^2}-\frac{\xi^3}{3},\quad 0<\xi<\frac 1n.
$$

因此

$$
   c_n<-\frac 12\left(\frac 1n+ \log\left( 1-\frac 1n \right) \right)<\frac{1}{4n^2}+\frac{1}{3n^3}<\frac{1}{3n^2}.\quad (n>1)
$$

这说明

$$
   \sum_{n=1}^{\infty} c_n<\sum_{n=1}^n\frac{1}{3n^2}=\frac{\pi^2}{18}
$$

是收敛的. 这也说明

$$
   1-\log(a_n)=b_n=\sum_{k=1}^nc_n<\frac{1}{18\pi^2},
$$

从而给出 $a_n$ 的一个下界

$$
   a_n>\exp\left( 1-\frac{1}{18\pi^2} \right).
$$

这已经足以得到我们 (在这篇文章中) 需要的结论:

$$
   \begin{aligned}
      \lim_{n\to\infty}\dfrac{n!}{\sqrt{n}\left(\dfrac{n}{e}\right)^n} = c, \quad 0<c<+\infty.
   \end{aligned}
$$

甚至给出更精细的范围

$$
   \begin{aligned}
      \lim_{n\to\infty}\dfrac{n!}{\sqrt{n}\left(\dfrac{n}{e}\right)^n} = c, \quad 1-\frac{1}{18\pi^2}<\log(c)<1.
   \end{aligned}
$$

而证明$c=\sqrt{2\pi}$ 一般需要借助Wallis公式, 在一般的数学分析教科书中都可以查阅到.

#### Stirling公式——借助Wallis公式计算极限的值

事实上, 由Wallis公式

$$
   \frac{\pi}{2}=\lim_{n\to\infty}\left(\dfrac{(2n)!!}{(2n-1)!!}\right)^2\dfrac{1}{2n+1}=\lim_{n\to\infty}\left(\dfrac{((2n)!!)^2}{(2n)!}\right)^2\dfrac{1}{2n}=\lim_{n\to\infty}\left(\dfrac{2^{2n}(n!)^2}{(2n)!}\right)^2\dfrac{1}{2n}
$$

我们知道

$$
   \lim_{n\to\infty}\dfrac{2^{2n}(n!)^2}{\sqrt{n}(2n)!}=\sqrt{\pi}.
$$

记

$$
   d_n=\dfrac{2^{2n}(n!)^2}{\sqrt{n}(2n)!},
$$

并把

$$
   n!=a_n\sqrt{n}\left( \frac ne \right)^n
$$

代入 $d_n$ 的表达式可以得到

$$
   d_n=\dfrac{2^{2n}(n!)^2}{\sqrt{n}(2n)!}=\dfrac{2^{2n}(a_n)^2n^{2n+1}e^{-2n}}{\sqrt{n}\sqrt{2n}a_{2n}(2n)^{2n}e^{-2n}}=\dfrac{a_n^2}{\sqrt{2}a_{2n}},
$$

而 $a_n\to c>0, d_n\to\sqrt{\pi}$ , 在上式两端令$n\to\infty$ 得到

$$
   \frac{c^2}{\sqrt{2}c}=\sqrt \pi,
$$

这说明

$$
   c=\sqrt{2\pi} .
$$