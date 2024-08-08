# 排序算法

## 概览

### 分类

| 数据规模、存储特点 | 内部排序（内存即可容纳） | 外部排序（借助外部存储） |
| :----------------: | :----------------------: | :----------------------: |
|      输入形式      |         离线算法         |         在线算法         |
|  所依赖的体系结构  |           串行           |           并行           |
|  是否采用随机策略  |          确定式          |          随机式          |

### 比较树模型和复杂度下界

1. 比较树是一种二叉树，用于表示基于比较的算法，如排序和搜索算法。在比较树中，每个内部节点表示一个比较操作，每个叶节点表示一个可能的输出。
2. 比较树模型可以用来理解和分析基于比较的算法。例如，比较树的深度（最长路径的长度）表示了算法在最坏情况下需要进行的比较次数。这可以用来证明基于比较的排序算法的下界，即无论我们如何改进算法，任何基于比较的排序算法在最坏情况下都必须进行至少n log n次比较，其中n是要排序的元素的数量。
3. 复杂度下界：一般地，任一问题在最坏情况下的最低计算成本，即为该问题的**复杂度下界**。
4. 比较式算法：比较式算法（Comparison-Based Algorithm, CBA）是一类使用比较操作来解决问题的算法。在这类算法中，数据项之间的相对顺序是通过一对一的比较来确定的。排序算法是比较式算法的一个常见例子。

### 两条定理

1. Any algorithm that sorts by exchanging adjacent elements requires $\Omega(n^2)$ average time.

	> In order for a sorting algorithm to run in sub-quadratic time, it must do comparisons and exchanges between elements that are far apart.

2. No sorting algorithm based on key comparisons can possibly be faster than $\Omega(n\log n)$; (Proof Below)

### 估计下界

1. 考查任一CBA式算法A，设`CT(A)`为与之对应的一棵比较树。

2. 根据比较树的性质，算法A每一次运行所需的时间，将取决于其对应叶节点到根节点的距离 （称作叶节点的深度）；而算法A在最坏情况下的运行时间，将取决于比较树中所有叶节点的最大深度（称作该树的高度，记作h(CT(A))）。因此就渐进的意义而言，算法A的时间复杂度应不低于$\Theta\left(h\left(CT(A)\right)\right)$。

3. 则对于任一的CBA算法，其复杂度下界就应为比较树的最小高度。为了计算最小高度，只要考察树中的**叶**节点的数目。对于树高为`h`二叉树（在比较中，无非就是大（小）于或不大（小）于这两个结果，所以这里应为二叉树），其叶子节点的个数不可能多于$2^h$。反过来，有：

  > 若某一问题的输出结果不少于$N$种，则比较树中叶节点也不可能少于$N$个，树高$h$不可能低于$\log_2N$。

4. 对于CBA式的排序算法，就n个元素的排序问题而言，可能的输出共有$n!$种。此时，元素间不止可以判等，而且可以比较大小，故此时，比较树属于三叉树。仿照以上的内容，我们有：

  > 若某一规模为$n$的排序问题的输出结果不少于$N=n!$种，则三叉比较树种的叶子节点也不会少于$N$个，树高$h$不可能低于$\log_3N$。

  这里，使用Stirling逼近公式，有：

  > $$
  > h\ge\left\lceil{\log_3(n!)}\right\rceil=\left\lceil{\log_3\text e\cdot\ln(n!)}\right\rceil=\Omega(n\log n).
  > $$
  >
  > 由此可见，最坏情况下CBA式排序算法至少需要$\Omega(n\log n)$时间，其中n为待排序元素数目。

5. $\Omega(n\log n)$的下界是针对比较树模型而言的。事实上，还有很多不属此类的排序算法（桶排序、基数排序算法），而且其中一些算法在最坏情况下的运行时间，有可能低于这一下界，但与上述结论并不矛盾。
# 排序算法实现
## 适合循秩访问的数据结构的排序算法

### 冒泡排序

1. 思想：每次比较相邻的两个元素，如果顺序不对则交换。

2. 代码实现：

   ```kotlin
    fun bubbleSort(arr: IntArray) {
        val n = arr.size
        for (i in 0 until n) {
            for (j in 0 until n - i - 1) {
                if (arr[j] > arr[j + 1]) {
                    val temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
                }
            }
        }
    }
   ```

   > 冒泡排序的曲线为
   >
   > 
   > $$
   > \left\{(x,y,t)\in[0,1]^3:y\le\frac{x}{x+t},x\le1-t\right\}\cup\left\{(x,x,t):x,t\in[0,1]\right\}
   > $$
   > 
   >
   > [A Rigorous Derivation of the Bubble Sort Curve | Lines That Connect](https://linesthatconnect.github.io/blog/a-rigorous-derivation-of-the-bubble-sort-curve/)

3. 基本版

   ```c++
   template <typename T> void Vector<T>::bubbleSort( Rank lo, Rank hi ) {
       while ( lo < --hi ) //逐趟起泡扫描
       	for ( Rank i = lo; i < hi; i++ ) //逐对检查相邻元素
       		if ( _elem[i] > _elem[i + 1] ) //若逆序
           		swap( _elem[i], _elem[i + 1] ); //则交换
       }
   ```

4. 不过，我们不必需要如此多趟起泡扫描——可能前面的元素依然有序，而我们仍在执行上述操作。我们可以加入判断符，判断上一趟起泡操作完成时，前面是否依然有序。

   ```c++
   template <typename T> void Vector<T>::bubbleSort( Rank lo, Rank hi ) {
       for ( bool sorted = false; sorted = !sorted; hi-- )
           for ( Rank i = lo + 1; i < hi; i++ )
               if ( _elem[i-1] > _elem[i] )
               	swap( _elem[i-1], _elem[i] ), sorted = false;//若本趟未发生交换，则下一趟可以提前退出。
   }
   ```

   >乱序在元素$[0,r)$时，仍要进行$O(n\cdot r)$时间。每一次上述的操作，前面乱序的元素在不断就位，而后面的元素不移动，直到前面的元素全部就位。

5. 上述操作有了明显的改进，但是还可以继续改进：若某一后缀元素已然有序，则……

   ```c++
   //跳跃版，该算法最好时间复杂度是O(n)，最差是O(n^2)
   template <typename T> void Vector<T>::bubbleSort( Rank lo, Rank hi ) {
       for ( Rank last; lo < hi; hi = last )
           for ( Rank i = (last = lo) + 1; i < hi; i++ )
               if ( _elem[i-1] > _elem[i] )
                   swap( _elem[i-1], _elem[i] ), last = i;//可以记录每次扫描交换的位置，内层循环结束后，结果记录为最后一次的扫描交换。
   }
   ```

6. 冒泡排序的算法是稳定的，因为a和b相对位置发生变化，只能是先接近至相邻，后可能因逆序而交换位置。如果if语句内写成$\ge$，则会变得不稳定。

### 归并排序

1. 简介：由 J. von Neumann 于1945年首次编程实现，向量和列表通用。共分三大步：1.序列一分为二；2.子序列递归排序；3.合并有序子序列。时间复杂度T(n)可以表示为：

   
$$
    T(n)=O(1)+2T(n/2)+O(n).
$$


   根据主定理，$a=2, b=2,$ $f(n)=n$. 因为$\Theta(n^{\log_b a})=\Theta(n^{\log_2 2})=\Theta(n)$，根据主定理第二条，有


$$
    T(n)=\Theta(n\log n).
$$

1. 代码实现：

   ```C++
   template <typename T> void Vector<T>::mergeSort( Rank lo, Rank hi ) {
       if ( hi - lo < 2 ) return; //单元素区间自然有序，否则...
       Rank mi = (lo + hi) >> 1; //以中点为界
       mergeSort( lo, mi ); //对前半段排序
       mergeSort( mi, hi ); //对后半段排序
       merge(lo,mi,hi);//归并
   }
   ```

   

3. 二路归并：有序序列合二为一，保持有序。

   1. 若有两个数组a[m], b[n], 和一个合并数组c[m+n], 让i=0, j=0分别为a, b两个数组的下标遍历。哪一个下标对应的元素小，就将其放入c数组中，并将其值自增。重复执行上述操作，直至一方遍历至数组结束。剩下的未遍历的元素必然有序，将其全部放入c数组中。

   2. 代码实现：

      ```c++
      template <typename T> //对各自有序的[lo, mi)和[mi, hi)做归并
      void Vector<T>::merge( Rank lo, Rank mi, Rank hi ) { // lo < mi < hi
         Rank i = 0; T* A = _elem + lo; //合并后的有序向量A[0, hi - lo) = _elem[lo, hi)
         Rank j = 0, lb = mi - lo; T* B = new T[lb]; //前子向量B[0, lb) <-- _elem[lo, mi)
         for ( Rank i = 0; i < lb; i++ ) B[i] = A[i]; //复制出A的前缀
         Rank k = 0, lc = hi - mi; T* C = _elem + mi; //后缀C[0, lc) = _elem[mi, hi)就地
         while ( ( j < lb ) && ( k < lc ) ) //反复地比较B、C的首元素
            A[i++] = ( B[j] <= C[k] ) ? B[j++] : C[k++]; //将更小者归入A中
         while ( j < lb ) //若C先耗尽，则
            A[i++] = B[j++]; //将B残余的后缀归入A中——若B先耗尽呢？
          while (k < lc)
            A[i++]=C[k++];
         delete[] B; //释放临时空间：mergeSort()过程中，如何避免此类反复的new/delete？
      }
      ```

## 适合循位置访问的数据结构的排序算法

### 选择排序(Selection Sort)

#### 思路

1. 在未排序序列中找到最小（或最大）元素，存放到排序序列的起始位置。
2. 再从剩余未排序元素中继续寻找最小（或最大）元素，然后放到已排序序列的末尾。
3. 重复第二步，直到所有元素均排序完毕。

#### 为什么要选择选择排序

> 冒泡排序的效率太低：都是对元素进行比较，而冒泡排序中，最大值就位的过程需要用大量的交换，这样的效率过于低下。使用选择排序，可以大幅度地减少交换的过程。也就是说，选择排序在每一次的迭代中，只会做一次移动。

#### 代码实现

```c++
//c++
template <typename T> //对列表中起始于位置p、宽度为n的区间做选择排序
void List<T>::selectionSort( ListNodePosi<T> p, Rank n ) { // valid(p) && Rank(p) + n <= size
    ListNodePosi<T> head = p->pred, tail = p;
    for ( Rank i = 0; i < n; i++ ) tail = tail->succ; //待排序区间为(head, tail)
    while ( 1 < n ) { //在至少还剩两个节点之前，在待排序区间内
        ListNodePosi<T> max = selectMax ( head->succ, n ); //找出最大者（歧义时后者优先）
    insert( remove( max ), tail ); //将其移至无序区间末尾（作为有序区间新的首元素）
    tail = tail->pred; n--;
    }
}
template <typename T> //从起始于位置p的n个元素中选出最大者
ListNodePosi<T> List<T>::selectMax( ListNodePosi<T> p, Rank n ) {
    ListNodePosi<T> max = p; //最大者暂定为首节点p
    for ( ListNodePosi<T> cur = p; 1 < n; n-- ) //从首节点p出发，将后续节点逐一与max比较
        if ( !lt( ( cur = cur->succ )->data, max->data ) ) //若当前元素不小于max，（画家算法，和画家画油画类似，一处的颜色必是该片区域最后一次上色的颜色）
            max = cur; //更新最大元素位置记录
    return max; //返回最大节点位置
}
```

> 这里，`insert()`和`remove()`都需要使用到动态空间分配，需要尽量少的使用它们。

#### 时间复杂度

一共要迭代n次，且`selectMax()`操作的时间复杂度为$\Theta(n-k)$，`insert()`和`remove()`的操作的时间复杂度为$O(1)$，故总体上的时间复杂度仍为$\Theta(n^2)$。需要注意的是，和冒泡排序相比，改进后的选择排序的移动次数明显变少，所以在性能上还是有较大地提升。

### 插入排序(Insertion Sort)

#### 思想

插入排序是一种简单的排序算法，它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。也就是说，将数组分为两部分，一部分是有序的，一部分是无序的。每次从无序部分取出一个元素，插入到有序部分的合适位置。

实例：玩扑克牌时的整理牌面的动作。

#### 和选择排序的不同之处

1. 有序部分和无序部分的位置的不同。在选择排序中，有序部分是所有元素的靠后的位置，而在插入排序中，元素从前到后依次变为有序，即有序部分是所有元素的靠前的位置。

2. 无序部分的元素大小的问题。在选择排序中，无序部分的元素一定不会超过有序部分元素的最小值（详见算法），而在插入排序中，上述内容并不能得到保证。

#### 代码实现

```cpp
template <typename T> //对列表中起始于位置p、宽度为n的区间做插入排序
void List<T>::insertionSort( ListNodePosi<T> p, Rank n ) { // valid(p) && Rank(p) + n <= size
   for ( Rank r = 0; r < n; r++ ) { //逐一为各节点
      insert( search( p->data, r, p ), p->data ); //查找适当的位置并插入
      p = p->succ; remove( p->pred ); //转向下一节点
   }
}
```

```cpp
void InsertSort(int a[]) {
	int temp;
	int j;
	for (int i = 1; i < a.length; i++) {//从1号元素开始，因为0号元素必有序
		if (a[i] < a[i - 1]) {
			temp = a[i];
			for (j = i - 1; j >= 0 && temp < a[j]; j--) {//比较在有序部分中的大小，插入zhi
				a[j + 1] = a[j];
			}
			a[j + 1] = temp;
		}
	}
}
```

#### 时间复杂度

1. 最好情况下为$O(n)$，最坏情况下为$O(n^2)$，则时间复杂度为$O(n^2)$。
2. 若使用支持**循秩访问**的数据结构，在判断插入的位置时使用时间复杂度为$O(\log n)$的二分查找，则插入排序的时间复杂度仍为$O(n^2)$。

> 这是因为虽然在查找位置时的时间复杂度降低了，但是数据结构变为向量，导致在执行插入新元素至已经由二分查找确定好的位置时候必须把该位置之后的所有元素都要后移一位，而这需要$O(n)$的时间复杂度。所以，时间复杂度还是没有明显的变化。

#### 插入排序的平均性能

采用**后项分析(Backward Analyze)**的方法。

1. 提出假设：每个元素的取值遵守均匀、独立的分布。

2. 考察：当秩为`r`的元素刚插入完成的那一刻，此时秩为$[0,r]$的有序前缀中，哪一个是我们刚刚插入的哪一个元素？

3. 解释：每一个元素均有可能，而且其概率均等于$\dfrac{1}{r+1}$。所以，在刚刚完成的这一次迭代中，为了引入该元素所花费时间的数学期望为

   
   $$
        E(r)=1+\frac{r+(r-1)+ \ldots+3+2+1+0}{r+1}=1+\frac r2.
   $$
   

   于是，总体时间的数学期望为

   
   $$
        E(X)=1+\sum_{r=0}^{n-1}E(r)=O(n^2).
   $$

   

   > 这里使用了数学期望的性质：和的期望等于期望的和，即
   >
   > 
   > $$
   >    E(X_1+X_2+ \ldots+X_n)=E(X_1)+E(X_2)+ \ldots+E(X_n).
   > $$
   > 
   >
   > 证明略。

#### 逆序对

1. 设$A$为一个有$n$个数字的有序集$(n>1)$，其中所有数字各不相同。如果存在正整数$i,j$，使得  $1 \le i<j \le n$ 并且  `A[i]>A[j]`  ，则  $\langle A[i], A[j]\rangle$  这个有序对称为  A  的一个逆序对，也称作逆序数。
2. 逆序对的统计：为了方便起见，我们统一用两者中秩较大的元素进行逆序对的统计工作。记$i(n)$为秩为n的元素的逆序对的数量，即对于秩为n的元素，所有秩比其小而值比其大的元素的个数。
3. 逆序对在**插入排序**中的应用：对于即将要进行插入排序操作的在未排序序列首位的元素`arr[p]`，它进行比较以确定插入位置的次数正好为$i(p)$。$i(p)$就是`p`所对应的查找长度，而$\sum_pi(p)=I$为插入排序总的比较次数。（算法消耗时间的最主要部分）。故插入排序属于输入敏感(Input-Sensitive)的排序算法。

## 其他排序算法

### 快速排序

1. 思想：使用分治法。
   1. 选择一个元素作为"基准"（pivot）。
   2. 将所有**小于或等于**基准的元素移动到基准的左边，所有**大于**基准的元素移动到基准的右边。这个操作称为分区操作（partition）。
   3. 对基准左边和右边的两个子集，重复进行第一步和第二步的操作。
2. 以下是快速排序的Java实现：

 ```java
 public class QuickSort {
     public static void quickSort(int[] arr, int low, int high) {
         if (low < high) {
             int pivot = partition(arr, low, high);
             quickSort(arr, low, pivot - 1);
             quickSort(arr, pivot + 1, high);
         }
     }

     private static int partition(int[] arr, int low, int high) {
         int pivot = arr[high];
         int i = low - 1;
         for (int j = low; j < high; j++) {
             if (arr[j] < pivot) {
                 i++;
                 int temp = arr[i];
                 arr[i] = arr[j];
                 arr[j] = temp;
             }
         }
         int temp = arr[i + 1];
         arr[i + 1] = arr[high];
         arr[high] = temp;
         return i + 1;
     }
 }
 //在这个代码中，`quickSort`函数是主函数，它接受一个数组和两个索引作为参数，然后调用`partition`函数进行分区操作。`partition`函数会选择一个基准，然后将所有小于基准的元素移动到基准的左边，所有大于基准的元素移动到基准的右边，然后返回基准的位置。然后，`quickSort`函数会对基准左边和右边的两个子集进行递归排序。

 ```

3. 时间复杂度：快速排序的递推公式和归并排序的递推公式一致，均为$T(n)=2T(n/2)+cn$。所以其时间复杂度为$O(n\log n)$。最坏情况下是$O(n^2)$。由于快速排序的内部循环可以在大多数实际情况下非常快速地执行，因此它通常比其他$O(n\log n)$算法更快。

   > **平均情况下：**Assume that each of the sub-arrays' size pair is equally likely, and has probability of $\dfrac{1}{n}$. Average cost can be computed in a recurrence way:
   >
   > $$
   > T(n)=\frac{1}{n}\sum_{i=0}^{n-1}\left [ T(i)+T(n-1-i) \right ]+cn
   > $$
   >
   > **对该公式的推导：**
   >
   > 
   >
   > $$
   > \begin{aligned}
   > nT(n)&=\sum_{i=0}^{n-1}\left [ T(i)+T(n-1-i) \right ]+cn^2\\
   > (n-1)T(n-1)&=\sum_{i=0}^{n-2}\left [ T(i)+T(n-2-i) \right ]+c(n-1)^2
   > \end{aligned}
   > $$
   > 
   >
   > （1）-（2），并逐项化简（抵消），有
   >
   > 
   > $$
   > nT(n)=(n+1)T(n-1)+2cn
   > $$
   > 
   >
   > 两边同时除以$n(n+1)$：
   >
   > 
   > $$
   > \frac{T(n)}{n+1}=\frac{T(n-1)}{n}+\frac{2c}{n+1}=\frac{T(n-2)}{n-1}+\frac{2c}{n}+\frac{2c}{n+1}=\ldots
   > $$
   >
   > 
   >
   > 通项公式：
   >
   > 
   > $$
   > \frac{T(n)}{n+1}=\frac{T(n-k)}{n-k+1}+2c\left(\sum_{i=2}^{k+1}{\frac{1}{n-k+i}}\right)
   > $$
   > 
   >
   > 最后，因为$\sum H_n\le\ln n$，我们有：
   >
   > 
   > $$
   > \frac{T(n)}{n+1}=\frac{T(1)}{2}+2c\left(\frac{1}{3}+\frac{1}{4}+ \ldots+\frac{1}{n+1}\right)
   > $$
   > 
   >
   > 则$T(n)$是$O(n+1)+(n+1)O(\ln n)$的，即$T(n)=O(n\log n)$.

- 双路快速排序：双路快速排序算法是随机化快速排序的改进版本，`partition()`过程使用两个索引值`i,j`用来遍历数组，将小于`pivot`的元素放在索引指向位置的左边，将大于`pivot`的元素放在索引指向位置的右边。

### 桶排序(Bucket Sort)

1. 思想：将数组分为若干个桶，每个桶内的元素是有序的，然后将所有桶中的元素合并。

2. 代码实现：

   ```java
   public class BucketSort {
       public static void bucketSort(int[] arr, int bucketSize) {
           if (arr.length == 0) {
               return;
           }
           int minValue = arr[0];
           int maxValue = arr[0];
           for (int value : arr) {
               if (value < minValue) {
                   minValue = value;
               } else if (value > maxValue) {
                   maxValue = value;
               }
           }
           int bucketCount = (maxValue - minValue) / bucketSize + 1;
           List<List<Integer>> buckets = new ArrayList<>(bucketCount);
           for (int i = 0; i < bucketCount; i++) {
               buckets.add(new ArrayList<>());
           }
           for (int value : arr) {
               buckets.get((value - minValue) / bucketSize).add(value);
           }
           int index = 0;
           for (List<Integer> bucket : buckets) {
               Collections.sort(bucket);
               for (int value : bucket) {
                   arr[index++] = value;
               }
           }
       }
   }
   ```

 3. 时间复杂度：桶排序的平均时间复杂度为$O(n^2+n/k+k)$（将值域平均分成n块 + 排序 + 重新合并元素)，当$k\approx n$时为$O(n)$。桶排序的最坏时间复杂度为$O(n^2)$。（n是数组元素数量，k是桶的数量）

 4. 稳定性：如果使用稳定的内层排序，并且将元素插入桶中时不改变元素间的相对顺序，那么桶排序就是一种稳定的排序算法。由于每块元素不多，一般使用插入排序。此时桶排序是一种稳定的排序算法。

    

### 基数排序(Radix Sort)

1. 基数排序也是非比较的排序算法，对每一位进行排序，从最低位开始排序，复杂度为$O(kn)$,n为数组长度，k为数组中的数的最大的位数；

2. 基数排序是按照低位先排序，然后收集；再按照高位排序，然后再收集；依次类推，直到最高位。有时候有些属性是有优先级顺序的，先按低优先级排序，再按高优先级排序。最后的次序就是高优先级高的在前，高优先级相同的低优先级高的在前。基数排序基于分别排序，分别收集，所以是稳定的。

3. 算法描述：

   1. 取得数组中的最大数，并取得位数；
   2. arr为原始数组，从最低位开始取每个位组成radix数组；
   3. 对radix进行计数排序（利用计数排序适用于小范围数的特点）；

4. 代码实现：

   ```java
   public class RadixSort {
       public static void radixSort(int[] arr) {
           if (arr == null || arr.length < 2) {
               return;
           }
           int max = arr[0];
           for (int value : arr) {
               max = Math.max(max, value);
           }
           int maxDigit = 0;
           while (max != 0) {
               max /= 10;
               maxDigit++;
           }
           int mod = 10, div = 1;
           List<List<Integer>> bucketList = new ArrayList<>();
           for (int i = 0; i < 10; i++) {
               bucketList.add(new ArrayList<>());
           }
           for (int i = 0; i < maxDigit; i++, mod *= 10, div *= 10) {
               for (int value : arr) {
                   int num = (value % mod) / div;
                   bucketList.get(num).add(value);
               }
               int index = 0;
               for (List<Integer> bucket : bucketList) {
                   for (int value : bucket) {
                       arr[index++] = value;
                   }
                   bucket.clear();
               }
           }
       }
   }
   ```

5. 时间复杂度：$O(n\cdot k)$

### 堆排序(HeapSort)

### 什么是堆排序

1. **堆排序**是由1991年的计算机先驱奖获得者、斯坦福大学计算机科学系教授罗伯特·弗洛伊德(Robert W. Floyd）和威廉姆斯(J. Williams）在1964年共同发明了的一种排序算法。

2. 该排序算法分为前后两个阶段。第一阶段，是将待排序的n个元素组织为一个优先队列`Q`，然后，不断地将优先队列中最小的元素摘出，使得它们依次构成一个有序的序列。在堆排序中，第一阶段的建堆操作仅需要花费线性时间$O(n)$，第二阶段反复调用`delMin()`方法，不断地取出`Q`中的最小元素，这样，n次调用需要花费$O(n\log n)$时间，总的时间复杂度为$O(n\log n)$。

> 定理 基于比较操作的排序算法具有$\Omega(n\log n)$的时间复杂度下界。证明详见[排序总览 - AccelerateZ](https://acceleratez.github.io/2024/05/06/algorithm/排序总览/)

### 就地堆排序（In-place Heapsort）

1. **就地算法（In-place Algorithm）**：除输入本身外只使用常数辅助空间的算法，称作就地算法。

2. 许多情况下，等待排序的$n$个元素都以一个长度为$n$的数组`S[]`形式给出。为此，除了输入数组以外，我们还需要为堆结构消耗$O(n)$的辅助空间。而实际上，这个空间的规模完全可以降低到$O(1)$。

3. 算法思路：算法的总体思路和策略与**选择排序算法**基本相同：将所有词条分成**未排序**和**已排序**两类，不断从前一类中取出最大者，顺序加至后一类中。算法启动之初，所有词条均属于前一类；此后，后一类不断增长；当所有词条都已转入后一类时，即完成排序。

4. 具体的实现：

   1. 将其划分为**前缀H**和与之互补的后缀S，分别对应于上述**未排序**和已排序部分。与常规选择排序算法一样，在算法启动之初H覆盖所有词条，而S为空。注意：这里，若从小到大进行排序，则`H`应为**大根堆**。

   2. 不同之处：整个排序过程中，不论`H`有多少词条，其始终都组织成一个堆。另外，整个算法过程始终满足如下不变性：`H`中的最大词条不会大于`S`中的最小词条——除非二者之一为空，比如算法的初始和终止时刻。

   3. 首先取出`H`中的堆顶`M`，与末单元词条`X`交换。`M`既是当前堆中的最大者，同时根据不变性，也不大于`S`中的任何词条，故如此交换之后`M`必处于正确的排序位置。此时，可以等效地认为`S`扩大了一个单元，而`H`相应地缩小了一个单元。这里，`H`和`S`仍然满足不变性。

   4. 有很大的可能，`X`不能称为堆顶，否则会破坏堆的堆序性。此时，可以对`X`进行下滤，就能使得`H`的堆序性重新恢复。

      > 对于大根堆的下滤操作，交换时应选择其两个孩子的最大者。

   5. 重复第3，4步，直到整个排序完成。

5. C#代码实现

   ```c#
   public class HeapSort
   {
       public void Sort(int[] array)
       {
           int heapSize = array.Length;
   
           // Build heap
           for (int i = heapSize / 2 - 1; i >= 0; i--)
           {
               Heapify(array, heapSize, i);
           }
   
           // Extract elements from heap one by one
           for (int i = heapSize - 1; i >= 0; i--)
           {
               // Move current root to end
               Swap(array, 0, i);
   
               // Call heapify on the reduced heap
               Heapify(array, i, 0);
           }
       }
   
       private void Heapify(int[] array, int heapSize, int i)
       {
           int smallest = i; // Initialize smallest as root
           int left = 2 * i + 1; // left = 2*i + 1
           int right = 2 * i + 2; // right = 2*i + 2
   
           // If left child is smaller than root
           if (left < heapSize && array[left] < array[smallest])
           {
               smallest = left;
           }
   
           // If right child is smaller than smallest so far
           if (right < heapSize && array[right] < array[smallest])
           {
               smallest = right;
           }
   
           // If smallest is not root
           if (smallest != i)
           {
               Swap(array, i, smallest);
   
               // Recursively heapify the affected sub-tree
               Heapify(array, heapSize, smallest);
           }
       }
   
       private void Swap(int[] array, int i, int j)
       {
           int temp = array[i];
           array[i] = array[j];
           array[j] = temp;
       }
   }
   ```

   

6. 复杂度：在每一步迭代中，交换`M`和`X`只需常数时间，对`X`的下滤调整不超过$O(\log n)$时间。因此，全部$n$步迭代累计耗时不超过$O(n\log n)$。即便使用蛮力算法而不是Floyd算法来完成`H`的初始化， 整个算法的运行时间也不超过$O(n\log n)$。

## 总结

|   Algorithm    |     Time     |                        Notes                         |
| :------------: | :----------: | :--------------------------------------------------: |
| Insertion Sort |   $O(n^2)$   |  Slow (the worse case) In-place For small datasets   |
| Selection Sort |   $O(n^2)$   |           Slow In-place For small datasets           |
|   Merge Sort   | $O(n\log n)$ |         Fast Not in-place For large datasets         |
|   Quick Sort   | $O(n\log n)$ | Fast (the average case) In-place For large datasets  |
|  Bucket Sort   |    $O(n)$    | Fast (the best case) Not in-place For large datasets |
|   Radix Sort   |    $O(n)$    | Fast (the best case) Not in-place For large datasets |
