# 列表基础与ADT

## 列表：循位置访问

1. 根据是否修改数据结构，所有操作大致分为两类方式：静态和动态。
   1. 静态： 仅读取，数据结构的内容及组成一般不变：get$O(1)$、search$O(\log n)$;
   2. 动态： 需写入，数据结构的局部或整体将改变：put、insert、remove$O(n)$.
2. 与操作方式相对应地，数据元素的存储与组织方式也分为两种，即静态和动态。
   1. 静态： 数据空间整体创建或销毁，数据元素的物理次序与其逻辑次序严格一致；可支持高效的静态操作。比如向量，元素的物理地址与其逻辑次序线性对应。
   2. 动态： 为各数据元素动态地分配和回收的物理空间。相邻元素记录彼此的物理地址，在逻辑上形成一个整体；可支持高效的动态操作。
3. 列表（list是采用动态储存策略的典型结构。
   1. 其中的元素称作节点（node），通过指针或引用彼此联接。
   2. 在逻辑上构成一个线性序列$L={a_0,a_1,\ldots,a_n}$.
   3. 相邻节点彼此互称前驱（predecessor）或后继（successor）。
   4. 没有前驱/后继的节点称作首（first/front）/末（last/rear）节点。
4. Call-By-Position:列表的循位置访问
   1. 列表中各元素的物理地址将不再决定于逻辑次序，动态操作可以在局部完成，复杂度有望控制在$O(1)$
   2. Call-By-Position：利用节点之间的相互引用，找到特定的节点。（如：我的...朋友A的...亲戚B的...同事C的...战友D的...同学Z）
   3. 如果是按逻辑次序的连续访问，单次也是$O(1)$

## 列表ADT

1. 列表节点：作为列表的基本元素，列表节点首先需要独立地“封装”实现。

   > list~node~data类似于列车~车厢~货物

2. 列表节点的ADT：

   ```c++
   template <typename T> using ListNodePosi = ListNode<T>*; //列表节点位置（C++.0x）
   template <typename T> struct ListNode { //简洁起见，完全开放而不再严格封装
       T data; //数值
       ListNodePosi<T> pred; //前驱
       ListNodePosi<T> succ; //后继
       ListNode() {} //针对header和trailer的构造
       ListNode(T e, ListNodePosi<T> p = NULL, ListNodePosi<T> s = NULL)
       : data(e), pred(p), succ(s) {} //默认构造器
       ListNodePosi<T> insertAsPred( T const & e ); //前插入
       ListNodePosi<T> insertAsSucc( T const & e ); //后插入
   };
   ```

3. 列表模板类：

   ```c++
   #include "listNode.h" //引入列表节点类
   template <typename T> class List { //列表模板类
       private: Rank _size; ListNodePosi<T> header, trailer; //哨兵
       protected: /* ... 内部函数 */
       public: //构造函数、析构函数、只读接口、可写接口、遍历接口
   };
   ```

   > 这里，我们把节点分成以下几种：头结点、尾结点、首结点、末结点、中间结点。其中，头、尾结点为哨兵结点，它们具有`private`属性，对外部不可见。首结点认为是对外部可见的结点中的前驱为头结点的结点，末结点认为是对外部可见的结点中的后继为尾结点的结点。一个列表中不一定有首、末结点，也不一定首、末结点同时出现。头、首、末、尾节点的秩，可分别理解为-1、0、n-1、n。

4. 构造：

   ```c++
   template <typename T> void List<T>::init() { //初始化，创建列表对象时统一调用
       header  = new ListNode<T>;
       trailer = new ListNode<T>;
       header->succ = trailer; header->pred = NULL;
       trailer->pred = header; trailer->succ = NULL;
       _size = 0;
   }
   ```

5. 此时可以重载下标运算符，模仿向量的Call-By-Rank访问方式。时间复杂度$O(r)$，当均匀分布时，期望复杂度为$(1+2+3+\ldots+n)/n=O(n)$。

## 线性表

### 单链表
1. 单链表(Singly-linked Lists)是具体的数据结构，包含一系列的结点序列：每一个结点储存`Data`和`*Next`。数据域：存储元素数值数据；指针域：存储直接后继的存储位置。

   > 这里把Node定义为私有静态内部类(private static class)，原因：1.杜绝被别的外部类调度或实例化； 2.只为其依附的外部类服务。[Java基础-私有静态内部类](https://blog.csdn.net/weixin_44228952/article/details/108482040)
2. 单链表操作举例：

   - 插入(Insertion)：
     - 若在头位置插入：插入的结点应指向current.head；插入的结点应为头结点。若在tail后插入结点，tail.next=current, current.next=null。插入后，size++. 时间复杂度：$$O(1)$$. 
     - 若在中间位置插入，两个步骤：1. build links for your insertion code, 2. change links on the original linked list. O(n)

```java
public void insert(int i, Object obj) throws Exception {//把新结点插入到第i个结点前，新结点data域的值为obj.
    if(i < 0 || i > size){
        throw new Exception("参数错误！");
    }
    index(i - 1);//先定位到第i-1个结点处，此时，current=第i-1个结点；
    current.setNext(new Node(obj,current.next));//创建新的结点，新结点的data=obj(T=Node)，next域为(current.next，实际上是指向第i个结点)
    // 加入新结点，set current由i-1.next变为新创建的结点
    size++;//第28行通过构造方法使得新结点和后继结点链接；setNext方法使得前序结点和新结点链接。
}
```
3. Singly-linked Lists versus ArrayLists
   - 相似：
     - Both two data structures store collections of elements, which can be accessed, added, removed, and modified.
     - Both two data structures can grow and shrink dynamically as elements are added or removed.
   - 不同：
     - Contiguous memory
     - Data access and manipulation
     - Dynamic sizing

### 单链表和顺序表的对比

|  项目  |                             优点                             |                             缺点                             |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 顺序表 |             主要优点是算法简单，空间单元利用率高             | 主要缺点是需要预先确定数据元素的最大个数，插入和删除时需要移动较多的数据元素。 |
| 单链表 | 主要优点是不需要预先确定数据元素的最大个数，插入和删除操作不需要移动数据元素； | 主要缺点是查找数据元素时需要顺序进行，不能像顺序表那样随机查找任意一个数据元素。另外，每个结点中要有一个指针域，因此空间单元利用率不高。而且单链表操作的算法也较复杂。 |

### 循环链表

结构特点是链表中最后一个结点的指针域指向整个链表的第一个结点，从而使链表形成一个环。它的优点是从链尾到链头比较方便。循环单链表也有带头结点和不带头结点两种结构。

```java
1.构造函数新增head.next=head的语句，把初始时的带头结点的循环单链表重新设计
2.在index(i)成员函数中，while(current!=null)换为while(current!=head) A=[3,5,6] B=[2,5,10,15] A union B=[2,3,5,6,10,15]
    a<b a.isLess(a,S)->A.remove(A.first()); b<a b.isLess(b,S)->B.remove(B.first()); a=b bothAreEqual(a,b,S)->A/B.remove(A/B.first());
```



### 双向链表

双向链表是每个结点除后继指针域外还有一个前驱指针域，它有带头结点和不带头结点，循环和非循环结构，双向链表是解决查找前驱结点问题的有效途径。
` A=[3,5,6] B=[2,5,10,15] A union B=[2,3,5,6,10,15]`

 `a<b a.isLess(a,S)->A.remove(A.first()); `

`b<a b.isLess(b,S)->B.remove(B.first()); `

`a=b bothAreEqual(a,b,S)->A/B.remove(A/B.first());`

```java
public class Node{
    Object data;
    Node prior;
    Node next;
}
pubilc class DoubleLink{
    Node head;
    Node current;
}
```

我们有

```java
p.next.prior=p.prior.next=p;
```

构建DoubleLink类时，需要Override insert/delete等成员函数，加上反向的。

> 单链表 头部插入 O(1) 尾部插入若有头结点O(1),无O(n) 头部删除O(1) 尾部删除 若有头结点O(n) 无O(n)
>
> 双链表 O(1)

### 仿真链表

可以用二维数组/Pair/Map等实现：

| index | element | next |
| ----- | ------- | ---- |
| 0     | A       | 1    |
| 1     | B       | 2    |
| 2     | C       | -1   |
# 无序、有序列表

操作仍是下面的几种：增、删、改、查。

## 插入与删除

1. 因为我们引入了两个哨兵结点，所以我们的插入算法和删除算法变得十分简单。

   ```c++
   //前插入（在this前插入）算法，后插入对称即可。：
   template<typename T> ListNodePosi<T> ListNode<T>::insertAsPred(T const& e){//O(1)
       ListNodePosi<T> x = new ListNode( e, pred, this ); //创建新结点，其前驱是pred，后继是this
       pred->succ = x; pred = x; //次序不可颠倒
       return x; //建立链接，返回新节点的位置
   } //得益于哨兵，即便this为首节点亦不必特殊处理——此时等效于insertAsFirst(e)
   ```

   

   ```c++
   template <typename T> T List<T>::remove( ListNodePosi<T> p ) { //删除合法节点p
       T e = p->data; //备份待删除节点存放的数值（设类型T可直接赋值）
       p->pred->succ = p->succ;   
       p->succ->pred = p->pred;  //短路联接
       delete p; _size--; return e; //返回备份的数值
   } //O(1)
   ```

   时间复杂度：$O(1)$。

   > （C++）析构函数`List<T>::~List()`：大致可以分两步：1.清空列表；2.释放头尾哨兵节点。
   >
   > 其中，清空列表的操作是反复删除首节点，使得列表变空；
   >
   > ```c++
   > int List<T>::clear(){
   > 	int oldSize = _size;
   > 	while(0 < _size)
   > 		remove(header->succ);
   > 	return oldSize;
   > }
   > ```
   >
   > 待上述操作完成之后，执行`delete header; delete trailer`即可。
   >
   > 析构函数的时间复杂度：$O(n)$。

## 查找算法

1. 这里将查找算法定义为在`p`的$n$ 个前驱中，找到某一元素。函数语句：`find(T const &e, Rank n, ListNodePosi<T> p);`。当然，我们也可以实现在它的$n$个后继中，查找到某一元素。

2. 算法的基本思路是从`p`出发，自后向前逐个比对，若匹配成功，则返回该节点；否则，则返回`NULL`。

3. 时间复杂度：$O(n)$。

## 去重算法

和在向量中的去重算法实现类似：对于任何一个节点，在保证前序的所有节点所存放的元素没有重复的前提下（使用从前到后依次查找即可实现该步骤），查找其前驱是否有重复的元素（使用`find`函数），若有，则使用`remove()`方法删除该节点，若无，则秩加一。（此时，该节点成为了唯一性的节点，也同时成为了下一个待去重节点的前驱节点。）返回列表规模的变化量，即被删除元素的总数。

## 有序列表

1. 唯一化：和向量的基本思路大致相同，不同的是，在列表中，可以直接进行删除操作，而不必和向量一样间接地进行删除的操作。时间复杂度：$O(n)$。

   算法过程：只保留每个相等元素区间的第一个元素。

   ```c++
   template <typename T> Rank List<T>::uniquify() { //成批剔除重复元素，效率更高
       if ( _size < 2 ) return 0; //平凡列表，自然不含相等元素
       Rank oldSize = _size; //记录原规模
       ListNodePosi<T> p = first(); ListNodePosi<T> q; //p为各区段起点，q为其后继
       while ( trailer != ( q = p->succ ) ) //反复考查紧邻的节点对(p, q)
           if ( p->data != q->data ) p = q; //若互异，则转向下一区段
           else remove( q ); //否则（相等）直接删除后者，不必如向量那样间接地完成删除
       return oldSize - _size; //列表规模变化量，即被删除元素总数
   }
   ```

2. 查找：和无序列表、有序向量的查找思路相似：从后向前地查找每一个节点，直至某一个节点的值大于或小于（查找失败）或等于（查找成功）。时间复杂度：最好情况下$O(1)$，最坏情况下为$O(n)$——总体上，仍为$O(n)$。这里，我们不能在有序列表中用二分查找使得时间复杂度降低至$\Theta(\log n)$，因为列表不能高效地循秩访问。例如，二分查找的第一步——定位到中点，访问模式为循位置访问的数据结构（如列表）所用的时间就为$\Theta(n)$。

3. 列表和向量的不同之处：

   | 比较对象 | 访问特点 | 应用实例 |
   | :------: | :------: | :------: |
   |  Vector  |   Rank   |   RAM    |
   |   List   | Position |    TM    |

   > 注：RAM模型属于循秩访问的典型模型：给出一个秩（下标），它可以在$O(1)$的时间内访问；而TM模型属于循位置访问的模型，它只能选择向左或向右访问相邻的内容，而对更远的内容，只能一步一步的前往并访问。
   >
   > Call-by-Rank和Call-by-Position这两种访问模式均存在自己的优势和不足之处，要根据不同的问题，选择不同的访问模式和相对应的数据结构。