# 二分查找和二分答案

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