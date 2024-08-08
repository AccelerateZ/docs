# 数组与向量：循秩访问的数据结构

## 数组

### 数组的ADT

- 操作集合：可以表示为$a_i$, 且限定数组元素必须储存在地址连续的内存单元中。

- 方法：

  - `acclocate()`分配所需空间；
  - `getLength()` 取数组的长度；
  - `set(i,x)`存数组元素；
  - `get(i)` 取数组的元素。

- 定义数组：
```java
int a[]=new int[10];//定义一维数组，Java
int b[][]=new int[10][10];//定义二维数组，Java
```

- 对象数组：

```java
public class Position{
    private int x,y;
    public Position(){
        x=y=0;
    }
}
Position pos[]=new Position[4];
```

> 但是，对象数组的每个数组元素都需要通过`new`运算符单独创建。对象数组也支持存操作和取操作，但和基本数据类型不同的是，所有对象名都是**引用**类型，所以，存取操作都是将一个已经创建的对象赋值给另一个对象引用，而不是新创建一个对象并赋值。

- 插入数组元素：

```java
public static void insertion(){
    int[] id=new int[10];
    for(int i=0;i<=5;i++){
        id[i]=i;
    }
    for(int i=5;i>=2;i--){
        id[i]=id[i-1];//向后移动元素，空出位置
    }
    id[1]=114;//New-value
}
//array deletion - similar to array insertion,时间复杂度为O(n)
```

## 从数组到向量

1. 线性数组：元素各由编号唯一指代，并可直接访问。普通的数组、向量类、线性表等。若每个元素占用的空间量为s（已计入padding），则A[i]的物理地址=`A+i*s`;
2. 向量是数组的抽象与泛化，由一组元素按线性次序封装而成。各元素与[0, n)内的秩（zhì，rank）一一对应：`using Rank = unsigned int;`
3. 特点：操作、管理维护更加简化、统一与安全；元素类型可灵活选取，便于定制复杂数据结构。



### 向量的ADT及其实例

1. 在Java中，向量（可变长数组）在`java.util.Vector`类中实现。

   ```Java
   public class Vector<E> extends AbstractList<E> implements List<E>, RandomAccess, Cloneable, Serializable{
       //Vector类的声明
   }
   ```

2. 向量类的接口：

   ```Java
   public interface Vector { 
       //返回向量中元素数目 
       public int getSize(); 
       //判断向量是否为空 
       public boolean isEmpty(); 
       //取秩为r的元素 
       public Object getAtRank(int r) throws ExceptionBoundaryViolation; 
       //将秩为r的元素替换为obj 
       public Object replaceAtRank(int r, Object obj) throws ExceptionBoundaryViolation; 
       //插入obj，作为秩为r的元素；返回该元素 
       public Object insertAtRank(int r, Object obj) throws ExceptionBoundaryViolation; 
       //删除秩为r的元素 
       public Object removeAtRank(int r) throws ExceptionBoundaryViolation; 
   } 
   ```

3.  基于数组实现向量ADT： 

   ```Java
   public class Vector_ExtArray<T> implements Vector<T> { 
       private int N = 8;//数组的容量，可不断增加 
       private int n;//向量的实际规模 
       private Object A[];//对象数组 
       //构造函数 
       public Vector_ExtArray() { A = new Object[N]; n = 0; } 
       //返回向量中元素数目 
       public int getSize() { return n; } 
       //判断向量是否为空 
       public boolean isEmpty() { return (0 == n) ? true : false; } 
       //取秩为r的元素 
       public Object getAtRank(int r) 
       throws ExceptionBoundaryViolation { 
           if (0 > r || r >= n) throw new ExceptionBoundaryViolation("意外：秩越界"); 
           return A[r]; 
       }
   //将秩为r的元素替换为obj 
       public Object replaceAtRank(int r, Object obj) 
       throws ExceptionBoundaryViolation { 
           if (0 > r || r >= n) throw new ExceptionBoundaryViolation("意外：秩越界"); 
           Object bak = A[r]; 
           A[r] = obj; 
           return bak; 
       } 
   //插入obj，作为秩为r的元素；并返回该元素 
       public Object insertAtRank(int r, Object obj) 
       throws ExceptionBoundaryViolation { 
           if (0 > r || r > n) throw new ExceptionBoundaryViolation("意外：秩越界"); 
           if (N <= n) {//空间溢出的处理 
               N *= 2; 
               Object B[] = new Object[N];//开辟一个容量加倍的数组 
               for (int i=0; i<n; i++) B[i] = A[i];//A[]中内容复制至B[] 
                   A = B;//用B替换A（原A[]将被自动回收） 
           } 
           for (int i=n; i>r; i--) A[i] = A[i-1];//后续元素顺次后移 
           A[r] = obj;//插入 
           n++;//更新当前规模 
           return obj; 
       } 
   //删除秩为r的元素 
       public Object removeAtRank(int r) 
       throws ExceptionBoundaryViolation { 
           if (0 > r || r >= n) throw new ExceptionBoundaryViolation("意外：秩越界"); 
           Object bak = A[r]; 
           for (int i=r; i<n-1; i++) A[i] = A[i+1];//后续元素顺次前移 
           n--;//更新当前规模 
           return bak; 
       } 
   }
   ```

   

4. 构造——基于复制的构造：首先在向量内部开辟出足够的空间，后再将区间内的元素逐一复制过来。

   ```Java
   public class Vector<T> {
       private T[] elementData;
       private int size;
       private int capacity;
       private static final int DEFAULT_CAPACITY = 10;
   
       public Vector(T []A, int lo, int hi) {
           copyFrom(A, lo, hi);
       }
   
       public void copyFrom(T[] A, int lo, int hi) {
           capacity = Math.max(DEFAULT_CAPACITY, 2 * (hi - lo));
           elementData = (T[]) new Object[capacity];
           size = 0;
           for (; lo < hi; lo++, size++) {
               elementData[size] = A[lo];
           }
       }
   }
   ```
   

## 再论插入：向量扩充


### 空间管理

1. 若采用静态空间管理策略，容量`capacity`固定，则有明显的不足……

   - 上溢(Overflow)：`elem[]`不足以存放所有元素，尽管此时系统往往仍有足够的空间>
   - 下溢(Underflow): `elem[]`存放的内容过少，装填因子$\lambda=\dfrac{\text{size}}{\text{capacity}}<<50\%$
   - 而且，一般的应用环境中，难以准确预测空间的需求量。所以，我们转向动态空间管理策略……

2. 动态空间管理：和蝉蜕壳时类似，在向量即将上溢时，适当扩大内部数组的容量。扩容算法如下（Doubling方法）：

   ```C++
   template <typename T> void Vector<T>::expand() { //向量空间不足时扩容
    if ( _size < _capacity ) return; //尚未满员时，不必扩容
    _capacity = max( _capacity, DEFAULT_CAPACITY ); //不低于最小容量
    T* oldElem = _elem;
    _elem = new T[ _capacity <<= 1 ]; //容量加倍
    for ( Rank i = 0; i < _size; i++ ) //复制原向量内容
        _elem[i] = oldElem[i]; //T为基本类型，或已重载赋值操作符'='
    delete [] oldElem; //释放原空间
   } //得益于向量的封装，尽管扩容之后数据区的物理地址有所改变，却不致出现野指针
   ```

   ```Java
   private void grow(int minCapacity) { 
       int oldCapacity = elementData.length; //10
       int newCapacity = oldCapacity + ((capacityIncrement > 0) ?
                                        capacityIncrement : oldCapacity);
       if (newCapacity - minCapacity < 0)
           newCapacity = minCapacity;
       if (newCapacity - MAX_ARRAY_SIZE > 0)
           newCapacity = hugeCapacity(minCapacity);
       elementData = Arrays.copyOf(elementData, newCapacity);
   }
   ```

### Doubling & Increment

1. 除了倍增外，还可以每次扩容时追加同样大小的空间(Increment方法)

2. 效率：

   1. Increment递增策略：最坏情况：在初始容量为0的空向量中，连续插入$n=m\cdot I >>2$个元素，不做删除，则在第$kI+1$次插入时，都要扩容，若不急申请空间的操作，各次扩容过程中的复制原向量的时间成本依次为$0,I,2I,3I,\cdots,(m-1)I$，总耗时$O(n^2)$，每次的分摊成本为$O(n)$。

   2. Doubling加倍策略：最坏情况：在初始容量1的满向量中，连续插入$n=2^m >>2$个元素，而无删除操作，则在第$2^k$次插入时，都要扩容。各次扩容过程中复制原向量的时间成本依次为$1,2,4,8,16,\cdots,2^{m-1},(2^m=n)$，总耗时$O(n)$，每次的分摊成本为$O(1)$。

   3. 倍增策略在空间上做了一些损失，而换取了时间上的巨大提升。

      

   ## 平均分析和分摊分析

   |          平均复杂度(Average Complexity)          |          分摊复杂度(Amortized Complexity)          |
   | :----------------------------------------------: | :------------------------------------------------: |
   | 根据各种操作出现概率的分布，将对应的成本加权平均 | 连续实施的足够多次操作，所需总体成本摊还至单次操作 |
   |       各种可能的操作，作为独立事件分别考查       |     从实际可行的角度，对一系列操作做整体的考量     |
   |          割裂了操作之间的相关性和连贯性          |         更加忠实地刻画了可能出现的操作序列         |
   |    往往不能准确地评判数据结构和算法的真实性能    |       更为精准地评判数据结构和算法的真实性能       |

## 集合

集合可以理解成特殊的数组/向量（可变长数组）。

- 集合也是一种循秩访问的数据结构；
- 集合中的元素不能够重复。

集合的抽象数据类型（ADT）：基本上和数组和可变长数组所拥有的ADT类似，不过不同的是，因为集合中没有重复的元素，所以对于集合，还有一些特殊的ADT。

- `contains(), containsAll()`：如果集合包含指定集合的(所有-All)元素，则返回true；
- `hashCode()`：返回哈希码值（集合中元素的地址）。
- 集合的运算ADT

其中，集合的运算ADT包括：
- Union-并集：两个集合`x`和`y`的并集`x.addAll(y)`；
- Intersection-交集：两个集合`x`和`y`的交集`x.retainAll(y)`；
- Subset-子集：判断`x`是否是`y`的子集`y.containsAll(x)`；
- Subtraction-差集：`S-T={e|e is in S but not is in T}`。



## 位图

1. 概念：位图就相当于一个数组，不过每个元素仅仅存放0或1，只占一位（bit）。

   > In computing, a bitmap is a mapping from some domain (for example, a range of integers) to bits. It is also called a bit array or bitmap index.

2. 结构：

   ```c++
   class Bitmap{
       private:
       unsigned char *M;//M for array(byte): M[0,N)
       int N,_size;
       public:
       Bitmap(int n=8){//B for Bit: B[0,n)
           M=new unsigned char[N=(n+7)/8];//ceil
           memset(M,0,N);
       }
       ~Bitmap(){
           delete [] M; M=NULL;
       }
       bool test(int k){expand(k);return M[k>>3] & (0x80>>(k & 0x07));}//1
       void set(int k){if(test(k)) return; expand(k); _size++; M[k>>3] |= (0x80>>(k & 0x07));}//2
       void clear(int k){if(!test(k)) return; expand(k); _size--; M[k>>3] &= ~(0x80>>(k & 0x07));}//3
   };
   ```

   

3. 解释：

   1. 对于`test`函数：先找到待查询的`k`号元素所在的字节`(k/8)`，然后再找到其具体的偏移量`(k%8)，或(k&0x07)`，然后生成`mask(掩码)`，满足在一个字节内部，只有相对应的位是`1`，剩下的位都是`0`。最后，将找到的`M[k>>3]`和这个掩码去做按位与操作，就能判断该值是否为1。`bit_mask`具体地求解过程：先生成一个最高位是1，其余位都是0的字节作为初始的掩码，然后对该掩码进行按位与操作，即可得到上述掩码。
   2. 对于`set`和`clear`函数，只需要改变运算符号就可以实现了。

4. 应用：

   1. 小集合+大数据

   2. 素数：使用`Bitmap`类和其中的`test(), set()`函数

   3. [O(1)快速初始化？带你搞懂J.Hopcroft 校验环！ (qq.com)](https://mp.weixin.qq.com/s?__biz=Mzg2NTY3NzIxMQ==&mid=2247484708&idx=1&sn=32feaf448af825d2eb298f5a69a4b274)
# 无序向量、有序向量

## C++模板类

1. `template`关键字用于定义模板。在C++中，`template`关键字可以用来定义两种主要的模板：函数模板和类模板。

   1. **函数模板**：函数模板是一种特殊的函数，可以处理不同类型的数据。例如，你可以创建一个函数模板来实现一个比较两个元素大小的函数，而不需要为每种数据类型都写一个单独的函数。

      ```c++
      template <typename T>
      T max(T a, T b) {
          return (a > b) ? a : b;
      }
      ```

   2. **类模板**：类模板是一种特殊的类，可以处理不同类型的数据。例如，你可以创建一个类模板来实现一个通用的向量类，而不需要为每种数据类型都写一个单独的向量类。

      ```cpp
      template <typename T>
      class Vector {
      private:
          T* data;
          int size;
      public:
          Vector(int size) : size(size) {
              data = new T[size];
          }
          ~Vector() {
              delete[] data;
          }
          // 其他成员函数...
      };
      ```

   3. 此外，还可以定义模板特化（Template Specialization），它允许你为特定类型定义不同的模板实现。这种好处很多，不仅可以实现泛型编程，而且在未来学习新的DSA时候通过这样的方式定义新的数据结构。如森林的定义：

      ```c++
      template<typename T> class Vector{...;}
      template<typename T> class BinTree{...;}
      Vector<BinTree> forest;//c++
      ```

      ```Java
      class Vector<E> implements List<E>{...;}
      class BinTree<T>{...;}
      ...
      Vector<BinTree> forest;//Java
      ```

## 循秩访问（Call by Rank）

1. 在向量类中，元素可以通过`vec.get(r)`（C++, Java）来访问。当然，在C++中，也可以通过重载下标运算符`[]`来实现循秩访问。（Java中则不可以） 

    ```c++
    template<typename T> T & Vector<T>::operator[](Rank r) { return _elem[ r ]; }//可以作为左值（将某个值赋值给向量中的某个元素）
    template<typename T> T & Vector<T>::operator[](Rank r) const { return _elem[ r ]; }//可以作为左值，但只能作为右值（将向量中的某个元素赋值给其他非向量的同一类型的变量）
    ```

    > 左值返回值之所以可以实现，是因为返回值是一个引用 `&`。

2. 插入和删除（将某些元素左移、右移）：

   ```c++
   template <typename T> Rank Vector<T>::insert( Rank r, T const & e ) {//0<=r<=size
       expand(); //如必要，先扩容
   	for ( Rank i = _size; r < i; i-- ) //O(n-r)：**自后向前**
   	_elem[i] = _elem[i - 1]; //后继元素顺次后移一个单元
       _elem[r] = e; _size++; return r;  //置入新元素，更新容量，返回秩
   }//返回插入的位置（若成功）
   ```

   ```java
    public Object insertAtRank(int r, Object obj) throws ExceptionBoundaryViolation{
        if (0 > r || r > n) throw new ExceptionBoundaryViolation("意外：秩越界"); 
        if (N <= n) {//空间溢出的处理 
            N *= 2; 
            Object B[] = new Object[N];//开辟一个容量加倍的数组 
            for (int i=0; i<n; i++){
                B[i] = A[i];//A[]中内容复制至B[] 
            }
            A = B;//用B替换A（原A[]将被自动回收） 
        } 
        for (int i=n; i>r; i--) 
            A[i] = A[i-1];//后续元素顺次后移 
        A[r] = obj;//插入 
        n++;//更新当前规模 
        return obj; 
    }
   ```

   ```c++
   template <typename T> Rank Vector<T>::remove( Rank lo, Rank hi ) { //0<=lo<=hi<=n
    	if ( lo == hi ) return 0; //出于效率考虑，单独处理退化情况
   	while ( hi < _size ) _elem[ lo++ ] = _elem[ hi++ ]; //后缀[hi,n)前移，自前向后的前移操作
    	_size = lo; shrink(); //更新规模，lo = _size之后的内容无需清零；如必要，则缩容 
   	return hi - lo; //返回被删除元素的数目
   }
   ```

   > 对于单元素删除操作：我们使用区间删除，将单元素视作区间的特例：`[r]=[r,r+1)`。所以可以重载`remove(r,r+1)`方法为`remove(r)`。
   >
   > 若反过来，通过反复调用单元素删除`remove(delElement)`的操作，而实现删除某个区间的元素，会导致整体$O(n^2)$的时间复杂度。每次循环耗时，正比于删除区间的后缀长度$n-hi=O(n)$ 而循环次数等于区间宽度 $hi - lo = O(n)$

3. 查找操作
   1. 该方法的时间复杂度和输入的数据有很大的关系，被称为输入敏感（input-sensitive）：最好O(1)，最差O(n)
   ```c++
   template <typename T> Rank Vector<T>::find( T const & e, Rank lo, Rank hi ) const { //0 <= lo < hi <= _size， //O(hi - lo) = O(n)
    	while ( (lo < hi--) && (e != _elem[hi]) ); //逆向查找
   	return hi; //返回值小于lo即意味着失败；否则即命中者的秩（有多个时，返回最大者）
   }
   ```

4. 唯一化

   1. 思想：当有新元素时，先查找其前驱是否有重复的元素（使用`find`函数），若有，则不放入向量中：若没有，则放入向量中。
   2. 代码实现：

      ```c++
      template <typename T> Rank Vector<T>::dedup() { //剔除相等的元素
      	Rank oldSize = _size;
      	for ( Rank i = 1; i < _size;  )
      	if ( -1 == find( _elem[i], 0, i ) ) //O(i)
      		i++;
      	else remove(i); //O(_size - i)
      	return oldSize - _size;
      } //O(n^2)：对于每一个e，只要find()不是最坏情况（查找成功），则remove()必执行
      ```
   3. 算法的证明：正确性：可以使用数学归纳法证明唯一化；单调性：while循环使得当前元素后缀长度单调下降，且迟早会减至0.
   4. 时间复杂度：主要是find()和remove()操作，每一次循环至多执行n次前述操作，while循环最多为n次，故其时间复杂度是$O(n^2)$的。
   5. 优化：通过`uniquify`方法，至多可以降至$O(n \log n)$.

5. 遍历`traverse()`
   1. 利用指针的机制，只读或做局部的修改。

   2. 利用函数对象的机制，可以进行全局操作。（推荐）

   3. 实现

      ```c++
      //先实现一个可使单个T类型元素加一的类（结构）
      template <typename T> //假设T可直接递增或已重载操作符“++”
       struct Increase{ //函数对象：通过重载操作符“()”实现
           virtual void operator()( T & e ) { e++; } }; //加一
      //再将其作为参数传递给遍历算法
      template <typename T> void increase( Vector<T> & V ){ 
          V.traverse( Increase<T>() ); } //即可以之作为基本操作，遍历向量
      ```



## 何谓有序？

必须让元素按顺序排序。这里就涉及到有序性的甄别操作（从冒泡排序模板改进）

1. 相邻逆序对的数目，可用来度量向量的逆序程度。

	```c++
	template <typename T> void checkOrder ( Vector<T> & V ) { //通过遍历
		int unsorted = 0; V.traverse( CheckOrder<T>(unsorted, V[0]) ); //统计紧邻逆序对
		if ( 0 < unsorted )
			printf ( "Unsorted with %d adjacent inversion(s)\n", unsorted );
		else
			printf ( "Sorted\n" );
	}
	```

2. 算法效率过于低下：因为在有序向量中，重复的元素必然相互紧邻构成一个区间，则每一个区间只需要保留单个元素即可。下面算法的时间复杂度为$O(n^2)$，与无序向量的唯一化操作相同，毫无效率可言。即**逐一移动**。

	```c++
	template <typename T> Rank Vector<T>::uniquify() { //有序向量重复元素剔除算法（低效版）
	   Rank oldSize = _size, i = 1; //当前比对元素的秩，起始于首元素
	   while ( i < _size ) //从前向后，逐一比对各对相邻元素
	      _elem[i - 1] == _elem[i] ? remove ( i ) : i++; //若雷同，则删除后者；否则，转至后一元素
	   return oldSize - _size; //向量规模变化量，即被删除元素总数
	}
	```

3. 改进的高效版本：使用两个指针，一个用来遍历元素，另一个则用来记录所有互异元素的数量。因为0号元素在其和其前驱内必然唯一，先令`i=0`。指针`j`开始后移，当找到第一个与前面互异的元素时，将第`j`号元素赋值给第`i+1`号元素（此时，必然有$i+1\le j$​）。重复上述操作，直至`j`移动至向量末尾，后将返回总删除个数，即`j-i`。即**不移动元素，最后批量删除**。共$n-1$次迭代，而每一次只有常数时间，故时间复杂度为$O(n)$。

	```c++
	template <typename T> Rank Vector<T>::uniquify() { //有序向量重复元素剔除算法（高效版）
	   Rank i = 0, j = 0; //各对互异“相邻”元素的秩
	   while ( ++j < _size ) //逐一扫描，直至末元素
	      if ( _elem[i] != _elem[j] ) //跳过雷同者
	         _elem[++i] = _elem[j]; //发现不同元素时，向前移至紧邻于前者右侧
	   _size = ++i; shrink(); //直接截除尾部多余元素（这一步表明后续重复的元素直接不参与向量的计数，即间接“删除”了他们）
	   return j - i; //向量规模变化量，即被删除元素总数
	}
	```

	> 不用考虑因为互异的元素太多而导致初始的靠前的互异元素被删除的情况，因为在到达这一步之前，已经在他的前面“拷贝”了一份副本。

## 二分查找

1. 语义约定：至少也应该便于向量的维护，以满足某些操作，如`vec.insert(1+vec.search(e),e)`。
2. 约定返回值：在`[lo,hi)`中返回不大于该值的最后一个元素。当有多个命中元素时，必须返回最靠后（秩最大）者；失败时，应返回小于e的最大者（含哨兵`lo-1`）。即：若e在$(-\infty,v[lo])$中，则返回左侧哨兵：若e在$(v[hi-1],\infty)$，则返回hi-1（末位元素是右侧哨兵的左邻）。
3. 二分查找：

	1. 减而治之：
		1.  e < x：则e若存在必属于左侧子区间，故可（减除[mi,hi)并）递归深入[lo, mi)
		2.  x < e：则e若存在必属于右侧子区间，亦可（减除[lo,mi]并）递归深入(mi, hi)
		3.  e = x：已在此处命中，可随即返回 //若有多个，返回何者？


   2. 代码实现（时间复杂度$T(n)=T(n/2)+O(1)=O(\log n)$，成功和失败的平均查找长度均大致为$O(1.5\log n)$，证明使用递推法，见教材第50页）：

	```c++
	//二分查找算法（版本A）：在有序向量的区间[lo, hi)内查找元素e，0 <= lo <= hi <= _size
	template <typename T> static Rank binSearch( T* S, T const& e, Rank lo, Rank hi ) {
	   while ( lo < hi ) { //每步迭代可能要做两次比较判断，有三个分支
	      /*DSA*/ for ( int i = 0; i < lo; i++ ) 
	          printf ( "     " ); 
	      if ( lo >= 0 ) 
	           for ( int i = lo; i < hi; i++ ) 
	               printf ( "....^" );
	      printf ( "\n" );
	      Rank mi = ( lo + hi ) >> 1; //以中点为轴点（区间宽度折半，等效于其数值表示的右移一位）
	      if( e < S[mi] ) hi = mi; //深入前半段[lo, mi)继续查找
	      else if ( S[mi] < e ) lo = mi + 1; //深入后半段(mi, hi)继续查找
	      else return mi; //在mi处命中
	      if ( lo >= hi ) {
	           for ( int i = 0; i < mi; i++ ) 
	               printf ( "     " ); 
	           if ( mi >= 0 ) 
	               printf ( "....|\n" ); 
	           else printf ( "<<<<|\n" ); }
	   } //成功查找可以提前终止
	   return -1; //查找失败
	} //有多个命中元素时，不能保证返回秩最大者；查找失败时，简单地返回-1，而不能指示失败的位置
	```

## Fibonacci查找（二分查找的改进1）

1. 转向左、右分支前的关键码比较次数不等，而递归深度却相同；通过递归深度的不均衡对转向成本的不均衡做补偿，平均查找长度应能进一步缩短！如有$n=fib(k-1)$，则可以取$mi=fib(k-1)-1$，将子向量分为三部分：前$fib(k-1)-1$，命中“1”，后$fib(k-2)-1$。

2. 代码实现：

	```c++
	template <typename T> static Rank fibSearch( T * S, T const & e, Rank lo, Rank hi ) {//0 <= lo <= hi <= _size
		for ( Fib fib(hi - lo); lo < hi; ) { //Fib数列制表备查
	        while ( hi - lo < fib.get() ) 
	            fib.prev(); //自后向前顺序查找轴点（分摊O(1)）
	        Rank mi = lo + fib.get() - 1; //确定形如Fib(k)-1的轴点
	        if ( e < S[mi] ) 
	            hi = mi; //深入前半段[lo, mi)
	        else if ( S[mi] < e ) 
	            lo = mi + 1; //深入后半段(mi, hi)
	        else return mi; //命中
		}
		return -1; //失败
	} //有多个命中元素时，不能保证返回秩最大者；失败时，简单地返回-1，而不能指示失败的位置
	```

3. $\phi=\frac{\sqrt(5)-1}{2}=0.6180339...$

4. 通用策略：在任何区间`A[0,n)`内，总是选取`A[`$\lambda$` n]`作为轴点，

	1. 比如：二分查找对应于$\lambda=0.5$，Fibonacci查找对应于$\lambda=\phi$。
	1. 这类查找算法的渐近复杂度为$\alpha(\lambda)\cdot\log_2 n=O(\log n)$，常系数$\alpha(\lambda)$何时达到最小...
	1. 递推式：（加权平均的思想，并加上成本`->{1,2}`）

$$
\alpha(\lambda) \cdot \log _{2} n=\lambda \cdot\left[1+\alpha(\lambda) \cdot \log _{2}(\lambda n)\right]+(1-\lambda) \cdot\left[2+\alpha(\lambda) \cdot \log _{2}((1-\lambda) n)\right]
$$

   ​整理，有：


$$
\frac{-\ln 2}{\alpha(\lambda)}=\frac{\lambda \cdot \ln \lambda+(1-\lambda) \cdot \ln (1-\lambda)}{2-\lambda}
$$


当  $\lambda=\phi=\dfrac{\sqrt{5}-1}2$  时，  $\alpha(\lambda)=1.440420 \ldots$  达到最小。

## 二分查找的改进2

1. 每次迭代仅使用1次比较，则所有分支只有2个方向，而不再是3个。

2. 代码实现：

	```c++
	//二分查找算法（版本B）：在有序向量的区间[lo, hi)内查找元素e，0 <= lo < hi <= _size
	template <typename T> static Rank binSearch( T * S, T const & e, Rank lo, Rank hi ) {
	   while ( 1 < hi - lo ) { //有效查找区间的宽度缩短至1时，算法才终止
	      Rank mi = (lo + hi) >> 1; //以中点为轴点，经比较后确定深入[lo, mi)或[mi, hi)
	      e < S[mi]  ?  hi = mi  :  lo = mi;
	   } //出口时hi = lo + 1
	   return e == S[lo]  ?  lo : -1 ; //有多个命中元素时，不能保证返回秩最大者；查找失败时，简单地返回-1，而不能指示失败的位置
	}
	```

3. 上述的所有操作，均没有实现第2点中的“语义约定”。

## 二分查找的改进3

1. 代码实现：

	```c++
	//版本C
	template <typename T> static Rank binSearch( T * S, T const & e, Rank lo, Rank hi ) {
	   while ( lo < hi ) { //不变性：[0, lo) <= e < [hi, n)
	      Rank mi = (lo + hi) >> 1;
	      e < S[mi]  ?  hi = mi  :  lo = mi + 1; //[lo, mi)或(mi, hi)，[mi]或被遗漏？
	   } //出口时，区间宽度缩短至0，且必有[lo = hi] = M
	   return lo - 1; //至此，[lo]为大于e的最小者，故[lo-1] = m即为不大于e的最大者
	} //留意与版本B的差异
	```

	> 也可参考Python中的`bisect_left()`函数。

2. 与版本B的差异：

	1. 待查找区间宽度缩短至0而非1时，算法才结束
	2. 转入右侧子向量时，左边界取作`mi+1`而非`mi`——`A[mi]是否会被遗漏？`不会
	3. 无论成功与否，返回的秩严格符合接口的语义约定。

3. 版本C的正确性：

	1. 单调性显而易见。
	2. 不变性：`A[0,lo)<=e<A[hi,n)`（其中，e是欲查找的元素）
		1. 初始时，lo=0,  hi=n, 自然成立。
		2. 数学归纳：假设不变性一直保持至某一次执行完成，对于下一次执行，以下分为两种情况：
			- 当`e<A[mi]`时，由于之前有`e<A[hi]`成立，且向量有序，则可以将`hi`的范围拓展成下界为`mi`,而保证初始条件不变。即`e<A[hi]`和`e<(A[mi]~A[hi])`$\Rightarrow$`e<[mi]`。
			- 当`A[mi]≤e`时，与上述同理，所以可以将`lo`的上界拓展成`mi+1`，以保证在`[0,lo-1)`(`[0,lo)`)中，均满足`A[0,lo)≤e`。

## 此外……

还有插值查找等内容，本质上是通过猜测轴点`mi`，提高收敛速度。而在实际上的算法实现中，往往是“算法接力”。首先通过插值查找将范围迅速缩小，然后再使用二分查找，进一步缩小范围，最后使用顺序查找。