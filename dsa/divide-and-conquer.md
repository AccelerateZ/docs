# 分治算法

## 二分查找（Binary Search）

1. 二分查找算法，也叫折半查找算法。二分查找的思想非常简单，有点类似分治的思想。二分查找针对的是一个有序的数据集合，每次都通过跟区间的中间元素对比，将待查找的区间缩小为之前的一半，直到找到要查找的元素，或者区间被缩小为 0。
2. 二分查找的时间复杂度：$O(\log n)$
3. 二分查找的局限性：

   1. 二分查找依赖于数组结构；
   2. 二分查找针对的是有序数据；
   3. 数据量过小或过大均不适合二分查找。
4. 常见使用例：

   1. 看见最值中求最值就是二分 ，如：最大值最小， 最小值最大等等。
   2. 不是有序才用二分，有序只是二分的必要条件，所以对于区间求最值，都可以尝试使用二分。

## 代码实现

1. 二分查找的代码实现：二分查找可以使用循环或递归来实现

   ```Java
   private static int bserach(int[] nums, int n, int value) {//循环实现
       int low = 0, high = n - 1;
       while (low <= high) {
           // 找出中间下标 
           int mid = low + ((high - low) >> 1);
           if (nums[mid] > value) {
               high = mid - 1;
           } else if (nums[mid] < value) {
               low = mid + 1;
           } else {
               return mid;
           }
       }
       return -1;
   }
   private static int recursiveBserach(int[] nums, int low, int high, int value){//递归实现
       if (low > high) return -1;
       // 找出中间下标
       int mid = low + ((high - low) >> 1);
       if (nums[mid] == value) {
           return mid;
       } else if (nums[mid] > value) {
           return recursiveBserach(nums, low, mid - 1, value);
       } else {
           return recursiveBserach(nums, mid + 1, high, value);
       }
   }
   ```

   

2. 二分算法的模板：

   1. `C++`中的`std::lower_bound()`函数：在已划分的范围 `[first, last)` 中查找第一个**不**先序于` value `的元素。在标头`<algorithm>`中定义。可以通过`operator<`和`comp()`来确定顺序：

      > 返回 `[first, last)` 中首个使得 `bool(*iter < value) `或`bool(comp(\*iter, value))`是` false `的迭代器 `iter`，或者在不存在这种 `iter` 的情况下返回 `las`t。
      >
      > 如果 `[first, last)` 的元素 `elem` 没有按表达式`bool(elem < value) `或`bool(comp(elem,value))`划分，则行为未定义。
      >
      > 返回值：返回到范围 `[first, last)` 的第一个不先序于 `value `的元素的迭代器，或者在找不到这种元素时返回` last`。
      >
      > 详见[std::lower_bound - cppreference.com](https://zh.cppreference.com/w/cpp/algorithm/lower_bound)

   2. `C++`中的`std::upper_bound()`函数：在已划分的范围 `[first, last)` 中查找第一个后序于` value `的元素。在标头`<algorithm>`中定义。

      > 详见[std::upper_bound - cppreference.com](https://zh.cppreference.com/w/cpp/algorithm/upper_bound)

      需要注意的是，在使用`lower_bound`和`upper_bound`函数的时候，返回结果为一个迭代器，必须使用返回值减去数组的初始内存地址（即数组名称）。如

      ```c++
      int position=lower_bound(v.begin(), v.end(), searchNum) - v;
      int pos=upper_bound(arr, arr+arr.length, searchNum) - arr;
      ```

      > *由迭代器去寻找元素的索引：将返回的迭代器减去容器的起始迭代器来获取元素的索引。*

   3. Python标准库的`bisect_left()`函数：求非降序范围 `[first, last)`内第一个不小于`value`的值的位置，同时适用于区间为空、答案不存在、有重复元素、搜索开/闭的上/下界等情况。

      > Return the index where to insert item x in list a, assuming a is sorted. The return value i is such that all e in a[:i] have e < x, and all e in a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will insert just before the leftmost x already there. Optional args lo (default 0) and hi (default len(a)) bound the slice of a to be searched.

      ```Python
      def lower_bound(array, first, last, value):  # 求非降序范围[first, last)内第一个不小于value的值的位置
          while first < last: # 搜索区间[first, last)不为空
              mid = first + (last - first) // 2  # 防溢出
              if array[mid] < value: first = mid + 1 
              else: last = mid
          return first  # last也行，因为[first, last)为空的时候它们重合
      ```

   4. Python标准库的`bisect_right()`函数：[cpython/Lib/bisect.py at 3.9 · python/cpython (github.com)](https://github.com/python/cpython/blob/3.9/Lib/bisect.py)

      ```python
      def bisect_right(arr, x, low=0, high=None):
          if low<0: raise ValueError('low must be non-negative')
          if high is None: high = len(arr)
          while low < high:
              mid = (low + high) // 2
              if x < arr[mid]: high = mid
              else: low = mid + 1
          return low
      ```

      > 其中，`low`和`high`是可选实参。

   5. 如何用`lower_bound/bisect_left`和`upper_bound/bisect_right`在`[first, last)`完成所有四种binary search (上/下界，开/闭区间)？ 

      1. `lower_bound(value)`本身找的是`x >= value`的下界，若为`last`则不存在；

      2. `upper_bound(value)`本身找的是`x > value`的下界，若为`last`则不存在；

         而区间是离散的，则

      3. `lower_bound(value) - 1` 即为`x < value`的上界，若为`first - 1`则不存在；`

      4. `upper_bound(value) - 1` 即为`x <= value`的上界，若为`first - 1`则不存在。

   6. 按照上述思想，可以实现手写的二分算法模板。

   ## 二分答案

3. 答案有一个区间，在这个区间中二分，直到找到最优答案。

   - 如何判断一个题是不是用二分答案做的呢?
     - 答案在一个区间内（一般情况下，区间会很大，暴力超时）
     - 直接搜索不好搜，但是容易判断一个答案可行不可行
     - 该区间对题目具有单调性，即：在区间中的值越大或越小，题目中的某个量对应增加或减少。
       此外，可能还会有一个典型的特征：求...最大值的最小 、 求...最小值的最大。



## 排序问题

### 合并排序(Merge Sort)

归并排序的基本运算是把两个有序的序列合并成一个有序的序列。代码示例如下：

```cpp
template <typename T> void Vector<T>::mergeSort( Rank lo, Rank hi ) {
    if ( hi - lo < 2 ) return; //单元素区间自然有序，否则...
    Rank mi = (lo + hi) >> 1; //以中点为界
    mergeSort( lo, mi ); //对前半段排序
    mergeSort( mi, hi ); //对后半段排序
    merge(lo,mi,hi);//归并
}
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

其时间复杂度$T(n)$可以表示为：


$$
    T(n)=O(1)+2T(n/2)+O(n).
$$


根据主定理，$a=2, b=2,$ $f(n)=n$. 因为$\Theta(n^{\log_b a})=\Theta(n^{\log_2 2})=\Theta(n)$，根据主定理第二条，有


$$
T(n)=\Theta(n\log n).
$$

### 快速排序(Quick Sort)

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

### 排序算法的时间下界

::: tip 定理

任何一个通过关键字值比较对$n$个元素进行排序的算法，在最坏的情况下，至少需要做$(n/4)\log n$次比较。

:::

> 证明：对$n$个元素进行排序，在最坏的情况下所比较的次数，取决它的二叉判定树(Binary Decision Tree)的高度，二叉判定树上面至少有$n!$个外结点，则其至少有$n!-1$个内结点。其树高至少为$\lceil\log (n+1)\rceil$。（不计外结点的高度）
>
> 当$n>1$，有：
> $$
> n!\ge n(n-1)(n-2)\dots\left(\lceil\frac{n}{2}\rceil\right)\ge\left(\frac{n}{2}\right)^{n/2}
> $$
> 则当$n\ge4$，有：
> $$
> T_C(n)\ge\lceil\log n!\rceil\ge \log n!\ge(n/2)\log (n/2)\ge(n/4)\log n
> $$
> 故$T(n)=O(n\log n)$.

## 选择问题

选择问题是指在一个无序的数组中找到第$k$小的元素。选择问题可以通过排序来解决，但是排序的时间复杂度为$O(n\log n)$，而选择问题可以在$O(n)$的时间复杂度内解决。

对于求最小或最大元的问题，当$1<k\le n/ \log n$时，可以使用堆排序求第k小元素：首先构造一个堆，其时间复杂度为$O(n)$。然后，依次删除堆顶元素$k-1$次，每次删除的时间复杂度为$O(\log n)$。因此，总的时间复杂度为$O(n+k\log n)$。对于$k<n/ \log n$的情况，可以使用快速选择算法。

### 快速选择算法

快速选择算法（Quickselect）是一种在未排序的数组中查找第k小/大元素的算法，时间复杂度为O(n)。它的基本思想是选择一个基准值（pivot），将数组分为两部分，一部分小于等于基准值，一部分大于基准值。然后根据k与基准值的大小关系，选择其中一部分进行递归搜索，直到找到第k小/大元素为止。

快速选择算法和快速排序算法的思路类似，但是快速选择算法只需要对一部分数组进行快速排序，而不需要对整个数组进行排序。因此，快速选择算法的平均时间复杂度为O(n)，最坏时间复杂度为O(n^2)，但是最坏情况出现的概率很小。

**快速选择算法的时间复杂度为$O(n)$，空间复杂度为$O(1)$**。

> 基本步骤：
>
> 1. 选择一个基准值`pivot`，可以选择数组中的任意一个元素。
>
> 2. 将数组分为两部分，一部分小于等于`pivot`，一部分大于`pivot`。
>
> 3. 如果`k`小于等于左边部分的元素个数，那么继续在左边部分中查找第`k`小元素；否则，在右边部分中查找第`k-left`个小元素。
> 4. 重复步骤2和3，直到找到第k小元素为止。
>

```cpp
void search(int left,int right) //递归函数，left和right表示当前搜索的区间
{
    int i=left,j=right,pivot=arr[(left+right)/2]; //取中间值作为基准值
    while(i<=j) //当i<=j时进行循环
    {
        while(arr[j]>pivot) //从右往左找到第一个小于等于基准值的数
            j--;
        while(arr[i]<pivot) //从左往右找到第一个大于等于基准值的数
            i++;
        if(i<=j) //如果i<=j，交换arr[i]和arr[j]的值
        {
            swap(arr[i],arr[j]);
            i++;
            j--;
        }
    }
    //快排后数组被划分为三块： left<=j<=i<=right
    if(k<=j) search(left,j); //如果k在左区间，只需要搜左区间
    else if(i<=k) search(i,right); //如果k在右区间，只需要搜右区间
    else //如果k在中间区间，直接输出arr[j+1]并结束程序
    {
        cout<<arr[j+1];
        exit(0); //任务完成，强制结束！
    }
}
```

上述算法和快速排序有相同的最坏情况的时间复杂度$O(n^2)$，但是，其平均时间复杂度是线性的，即$O(n)$。

如果想要让最坏情况下的时间复杂度仍为线性的，则在对`pivot`的选择上，要采用**二次取中法**(Median of Median Rule)。



## 斯特拉森矩阵算法

不妨设矩阵$A$和$B$均为$n\times n$的矩阵，其中$n=2^k, k\in\mathbb{Z}$，如果$n$不满足条件，可以加入全零行或全零列，来满足条件。分治法的基本思想是将矩阵$A$和$B$分割成四个$n/2\times n/2$的子矩阵，然后通过一系列的加减法运算，计算出矩阵$A$和$B$的乘积。

例如，对于矩阵

$$
A=\begin{bmatrix}A_{11}&A_{12}\\A_{21}&A_{22}\end{bmatrix}, B=\begin{bmatrix}B_{11}&B_{12}\\B_{21}&B_{22}\end{bmatrix}
$$

如果采用传统的矩阵乘法算法，需要进行8次乘法和4次加法，即

$$
\begin{aligned}
c_{11}&=A_{11}B_{11}+A_{12}B_{21}\\
c_{12}&=A_{11}B_{12}+A_{12}B_{22}\\
c_{21}&=A_{21}B_{11}+A_{22}B_{21}\\
c_{22}&=A_{21}B_{12}+A_{22}B_{22}
\end{aligned}
$$

如果，设两个方阵相乘的时间为$T(n)$，则有

$$
T(n)=\begin{cases}b, n \le 2,\\
8T(n/2)+dn^2, n>2.
\end{cases}
$$

有$T(n)=\Theta(n^3)$。

而斯特拉森矩阵乘法通过减少子矩阵的乘法次数，来减少时间复杂度。它先使用7次乘法和10次加法，计算出7个中间矩阵，而后通过加减法运算，计算出矩阵$A$和$B$的乘积。具体的计算过程如下：

$$
\begin{aligned}
P&=(A_{11}+A_{22})(B_{11}+B_{22})\\
Q&=(A_{21}+A_{22})B_{11}\\
R&=A_{11}(B_{12}-B_{22})\\
S&=A_{22}(B_{21}-B_{11})\\
T&=(A_{11}+A_{12})B_{22}\\
U&=(A_{21}-A_{11})(B_{11}+B_{12})\\ 
V&=(A_{12}-A_{22})(B_{21}+B_{22})
\end{aligned}
$$

则有

$$
\begin{aligned}
C_{11}&=P+S-T+V\\
C_{12}&=R+T\\
C_{21}&=Q+S\\
C_{22}&=P+R-Q+U
\end{aligned}
$$

于是，斯特拉森矩阵乘法的时间复杂度为$T(n)=7T(n/2)+O(n^2)$，其时间复杂度为$O(n^{\log_2 7})$。