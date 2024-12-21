# 树基础和基本实现

## 树的基本知识

### 树的优点：

|       项目       |                             优点                             |                             缺点                             |
| :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 基于数组实现的DS | 在**常数**的时间内找到目标对象，并读取或更新其内容。（通过下标或秩） | 一旦需要对这类结构进行修改（插入、删除），都需要耗费**线性**的时间。 |
| 基于链表实现的DS | 允许我们借助引用或位置对象，在**常数**的时间内插入或删除元素； | 为了找出居于特定次序的元素，我们不得不花费**线性**的时间对整个结构进行遍历查找。 |

> 树：每一次查找、更新、插入或删除操作都可以在$O(\log n)$内完成，而每次遍历都可以在$O(n)$的时间内完成。（平衡二分查找树）

> 不难证明，对任意的$c>0$，都有$\log n = O(n^c)$。也就是说，就多项式而言$O(\log n)$与$O(1)$可以无限接近，而$O(\log n)$与$O(n)$相比几乎提高了一个线性因子。因此就这一点而言，树结构的确能够将数组和链表的优点结合起来。

### 树的某些特征：

1. **半线性结构(Semi-linear Structures)**: 树结构的元素间不存在天然的直接后继或直接前驱关系，因此属于非线性结构(Non-linear structures)。而只要附加上某种约束（比如遍历），也可以在树结构中的元素之间确定某种线性次序，因此也有人称之为“半线性结构”。
2. **分层结构**：层次化。例子：文件系统、数据库系统、URL等。
3. **综合性**：兼具`Vector`和`List`的优点；兼顾高效的查找、插入、删除。
4. 树和图的相似性：树 = 无环连通图 = 极小连通图 = 极大无环图。

### 树的有关术语和性质：

1. 节点的深度、树的深度与高度：**树中所有节点的最大深度，称作树的深度或高度。**

   1. 树中的元素也称为节点`Node`。树中的每个节点`v`都被赋予了一个特殊的指标⎯⎯深度，记作`depth(v)`。
   2. 每个节点的深度都是一个非负整数；
   3. 深度为0的节点有且仅有一个，称作树根（Root）；
   4. 对于深度为$k\ (k\ge 1)$的每个节点`u`，都有且仅有一个深度为k-1的节点`v`与之对应，称作`u`的父亲（Parent）或父节点。
   5. 若节点`v`是节点`u`的父亲，则`u`称作`v`的孩子（Child），并在二者之间建立一条树边（Edge）。边的方向是从父亲指向孩子。
   6. 尽管每个节点至多只有一个父亲，但却可能有多个孩子。同一节点的孩子互称“兄弟”（Sibling）。
   7. 树中节点的数目，总是等于边数加一。

2. 度、内部节点与外部节点：

   1. **度（Degree）**：取某一节点，该节点的**孩子**数目。注意：节点的父亲不计入度中。

   2. **“内部节点”（Internal node）**：至少拥有一个孩子的节点称作内部节点。

   3. **“外部节点”（External node）**：没有任何孩子的节点则称作外部节点，或“叶子”(Leaf)。

      > 一个节点为叶子当且仅当它的度为`0`。

3. 路径（Path）：

   1. 由树中$k+1$节点通过树边首尾衔接而构成的序列$\{(v_0, v_1), (v_1, v_2), \ldots, (v_{k-1}, v_k) | k \ge  0\}$，称作树中长度为$k$的一条路径。下面的树，{(1, 2), (2, 5), (5,8), (8, 7), (7, 0)}为一条路径。注意，当讨论路径时，不需要考虑方向。

      ```
      								5
              +-----------------------+------------------+
      		2										   8
          +---+-------------+                   +--------+--------+
        	1           	  3					  6 				7
                   +--------+               +---+         		+---+
                   9 						  4 						0
      ```

   2. 树中任何两个节点之间都存在唯一的一条路径。

   3. 若`v`是`u`的父亲，则`depth(v) + 1 = depth(u)`。

      推论：从树根通往任一节点的路径长度，恰好等于该节点的深度。

4. 祖先、后代、子树和节点的**高度**：

   1. 每个节点都是自己的“祖先”（Ancestor），也是自己的“后代”（Descendant）；  

   2. 若`v`是`u`的父节点的祖先，则`v`也是`u`的祖先；若`u`的父节点是`v`的后代，则`u`也是`v`的后代。

   3. **真祖先（后代）**：除节点本身以外的祖先（后代），称作真（proper）祖先（后代）。

   4. 任一节点`v`的深度，等于其真祖先的数目。 

   5. 任一节点`v`的祖先，在每一深度上最多只有一个。（否则路径不唯一，与上面矛盾）

   6. 树`T`中每一节点`v`的所有后代也构成一棵树，称作`T`的“以`v`为根的**子树（Subtree）**”。

      > 空节点（`null`）本身也构成一棵树，称作“空树”（Empty tree）。空树虽然不含任何节点，但却是任何树的（平凡）子树。 特别地，空树的高度为$-1$。

      > 我们可将“以`v`为根的子树”直接称作“子树`v`”。

   7. 若子树`v`的深度（高度）为`h`，则称`v`的高度为`h`，记作height(v) = h。

   8. > 对于叶子节点`u`的任何祖先`v`，必有`depth(v) + height(v) \ge  depth(u)`。

5. 共同祖先及最低共同祖先：

   1. **共同祖先（Common Ancestor）**： 在树`T`中，若节点`u`和`v`都是节点`a`的后代，则称节点`a`为节点`u`和`v`的共同祖先。

   2. 每一对节点至少存在一个共同祖先。

   3. **最低共同祖先（Lowermost Common Ancestor, LCA）**：在一对节点u和v的所有共同祖先中，深度最大者称为它们的最低共同祖先，记作 `lca(u, v)`。LCA必存在且唯一。

## 树抽象数据类型及其实现

**“父亲——长子——弟弟”模型**：根据树的定义，每个节点的所有后代均构成了一棵子树，故从数据类型的角度来看，树、子树 以及树节点都是等同的。这里，将它们统一为一个类：`Tree`.

基于列表实现树：

   ```Java
   public class TreeLinkedList implements Tree { 
       private Object element;//树根节点 
       private TreeLinkedList parent, firstChild, nextSibling;//父亲、长子及最大的弟弟 
       //（单节点树）构造方法 
       public TreeLinkedList() 
       	{ this(null, null, null, null); } 
       //构造方法 
       public TreeLinkedList(Object e, TreeLinkedList p, TreeLinkedList c, TreeLinkedList 
       s) { 
           element = e; 
           parent = p; 
           firstChild = c; 
           nextSibling = s; 
       } 
       /*---------- Tree接口中各方法的实现 ----------*/ 
       //返回当前节点中存放的对象 
       public Object getElem() 
       	{ return element; } 
       //将对象obj存入当前节点，并返回此前的内容 
       public Object setElem(Object obj) 
       	{ Object bak = element; element = obj; return bak; } 
       //返回当前节点的父节点；对于根节点，返回null 
       public TreeLinkedList getParent() 
       	{ return parent; } 
       //返回当前节点的长子；若没有孩子，则返回null 
       public TreeLinkedList getFirstChild() 
       	{ return firstChild; } 
       //返回当前节点的最大弟弟；若没有弟弟，则返回null 
       public TreeLinkedList getNextSibling() 
      	 	{ return nextSibling; } 
       //返回当前节点后代元素的数目，即以当前节点为根的子树的规模 
       public int getSize() { 
           int size = 1;//当前节点也是自己的后代 
           TreeLinkedList subtree = firstChild;//从长子开始 
           while (null != subtree) {//依次 
               size += subtree.getSize();//累加 
               subtree = subtree.getNextSibling();//所有孩子的后代数目 
       	} 
   		return size;//即可得到当前节点的后代总数 
       } 
       //返回当前节点的高度 
       public int getHeight() { 
           int height = -1; 
           TreeLinkedList subtree = firstChild;//从长子开始 
           while (null != subtree) {//依次 
               height = Math.max(height, subtree.getHeight());//在所有孩子中取最大高度 
               subtree = subtree.getNextSibling(); 
           } 
           return height+1;//即可得到当前节点的高度 
       } 
       //返回当前节点的深度 
       public int getDepth() { 
           int depth = 0; 
           TreeLinkedList p = parent;//从父亲开始 
           while (null != p) {//依次 
           	depth++; p = p.getParent();//访问各个真祖先 
           } 
           return depth;//真祖先的数目，即为当前节点的深度 
       } 
   } 
   ```

## 统计（子）树的规模——`getSize()`

1. 结论： 一棵树的规模，等于根节点下所有子树规模之和再加一，也等于根节点的后代总数。

2. 操作（递归）：先使用`firstChild`找出根节点的长子，并沿着`nextSibling` 引用顺次找到其余的孩子，递归地统计出各子树的规模。最后，只要将所有子树的规模累加起来，再计入根节点本身，就得到了整棵树的规模。当遇到没有任何孩子的节点（即原树的叶子）时，递归终止。

   ```java
   public int getSize() { 
       int size = 1;//当前节点也是自己的后代 
       TreeLinkedList subtree = firstChild;//从长子开始 
       while (null != subtree) {//依次 
           size += subtree.getSize();//累加 
           subtree = subtree.getNextSibling();//所有孩子的后代数目 
       } 
       return size;//即可得到当前节点的后代总数 
   }
   ```

3. 时间复杂度：$O(n)$。

   

## 计算节点的高度——`getHeight()`

1. 结论：若`u`是`v`的孩子，则`[1]height(v) ≥ height(u) + 1; [2]height(v)=1+max(height(u))` 。

2. 操作（递归）：仍是先使用`firstChild`找出根节点的长子，并沿着`nextSibling` 引用顺次找到其余的孩子，递归地统计出各子树的高度。然后找到所有子树的最大高度并计入根节点本身，就得到了根节点的高度（即树高）。当该节点为叶子时，递归中止。

   ```java
   public int getHeight() { 
       int height = -1; 
       TreeLinkedList subtree = firstChild;//从长子开始 
       while (null != subtree) {//依次 
           height = Math.max(height, subtree.getHeight());//在所有孩子中取最大高度 
           subtree = subtree.getNextSibling(); 
       } 
       return height+1;//即可得到当前节点的高度 
   } 
   ```

3. 时间复杂度：$O(n)$。



## 计算节点的深度——`getDepth()`

1. 结论：若`u`是`v`的孩子，则`depth(u) = depth(v) + 1`。 

2. 操作（尾递归或迭代）：`getDepth(v)`将从`v`的父亲开始，沿着`parent`指针不断上移，直到深度为0的树根。在这个过程中所遇到的每个节点，都是`v`的真祖先；反之，在这一过程中，v的每一真祖先迟早都会被找到。因此，根据总共上移的层数，就可以得到`v`在整棵树中的深度。

   ```java
   public int getDepth() { 
       int depth = 0; 
       TreeLinkedList p = parent;//从父亲开始 
       while (null != p) {//依次 
           depth++; p = p.getParent();//访问各个真祖先 
       } 
       return depth;//真祖先的数目，即为当前节点的深度 
   } 
   ```

3. 时间复杂度：$O(n)$。

   

## 树的遍历

### 遍历(Traversal)

按照某种次序访问树中各节点，每个节点被访问恰好一次。按照被访问的次序，可以得到由树中所有节点排列成的一个序列。$T=\text V\cup\text L\cup\text R$。遵照不同的遍历规则，我们可以得到不同的遍历方法。遍历：结果 ~ 过程 ~ 次序 ~ 策略。

### 先序遍历(Pre-order Traversal)

1. 首先访问其根节点，然后再递归地做左子树的先序遍历，最后做右子树的先序遍历。

2. 应用：打印结构化文件。

3. 递归实现：

   ```cpp
   //c++
   template <typename T, typename VST> 
   void traverse( BinNodePosi<T> x, VST & visit ) {
       if ( ! x ) return;
       visit( x->data );
       traverse( x->lc, visit );//尾递归，可以较为简单地化为迭代形式
       traverse( x->rc, visit );//只需要引入一个栈，即可实现上述操作
   } //O(n)
   ```

   ```java
   //java
   public void PreorderTraversal(TreeLinkedList v){
       if(null!=v){
           visit(v.element);
           for(TreeLinkedList u=v.getFirstChild(); null!=u; u=u.getNextSibling()){
               PreorderTraversal(u);
           }
       }
   }
   ```

   

   > 制约：使用默认的`Call Stack`，允许递归的深度有限。我们需要不依赖于递归机制，即使用迭代算法。

4. 对于先序遍历（尾递归）的迭代形式的实现：

   ```cpp
   //cpp
   template <typename T, typename VST> //元素类型、操作器 
   void travPre_I1 ( BinNodePosi(T) x, VST& visit ) { //二叉树先序遍历算法（迭代版#1） 
       Stack<BinNodePosi(T)> S; //辅助栈 
       if ( x ) S.push ( x ); //根节点入栈 
       while ( !S.empty() ) { //在栈变空之前反复循环 
           x = S.pop(); visit ( x->data ); //弹出并访问当前节点，其非空孩子的入栈次序为先右后左 
           if ( HasRChild ( *x ) ) S.push ( x->rc ); 
           if ( HasLChild ( *x ) ) S.push ( x->lc ); 
       }
   }
   ```

   > 缺陷：对于中序和后序遍历，即非尾递归的场合，则无法实现。我们需要考虑一种新的思路。

5. **“藤缠树”**：仔细观察可以发现，沿着左侧藤，整个遍历过程可分解为：**自上而下**访问藤上节点，再自下而上遍历各右子树。这里，各右子树的遍历彼此独立，自成一个子任务。

   > 我们如何进行上述的操作？首先，访问根节点；然后，访问根节点的左孩子；再然后，访问根节点的左孩子的左孩子……即首先我们沿着**左侧链**一直进行访问并向着树的更深层走下去。如此下去，直到某一个节点没有左孩子。此时，我们进行一次转移操作，回退到上一次出现右孩子的节点，然后继续重复上述的操作。

   思路如下：我们只将左侧链画出：在左侧链上面，我们仅仅画出根节点和链上各节点的左孩子，而每个节点右孩子和其下面的所有内容，我们抽象地绘制成一棵右子树，标记在其原先的右孩子的位置。第一次访问，一定会沿着左侧链上的节点进行访问，直到访问到没有左孩子的节点`L_d`。然后，我们进行回溯，自下而上地遍历从`L_d`到`L_0`（根节点）上面的全部的右子树。

   ```
   preorder(T)={
       visit(L0); visit(L1); ...; visit(L_end-1); visit(L_end); 
       preorder(T_end);//(preorder就是从该位置重新执行一系列的visit操作，直到叶子节点)
       preorder(T_end-1); ...; preorder(T1); preorder(T0);
   }
   ```

   4. 迭代实现

   ```c++
   //从当前节点出发，沿左分支不断深入，直至没有左分支的节点；沿途节点遇到后立即访问
   template <typename T, typename VST> //元素类型、操作器
   static void visitAlongVine( BinNodePosi<T> x, VST& visit, Stack<BinNodePosi<T>>& S ) {
      while ( x ) {
         visit( x->data ); //访问当前节点
         S.push( x->rc ); //右孩子入栈暂存（可优化：通过判断，避免空的右孩子入栈）
         x = x->lc; //沿左分支深入一层
      }
   }//使用栈，正是利用其LIFO的特性。
   
   template <typename T, typename VST> //元素类型、操作器
   void travPre_I2( BinNodePosi<T> x, VST& visit ) { //二叉树先序遍历算法（迭代版#2）
      Stack<BinNodePosi<T>> S; //辅助栈
      while ( true ) {
         visitAlongVine( x, visit, S ); //从当前节点出发，逐批访问
         if ( S.empty() ) break; //直到栈空
         x = S.pop(); //弹出下一批的起点
      }
   }
   ```

   

### 中序遍历(Inorder Traversal)

1. 在中序遍历中，我们首先递归地做左子树的中序遍历，然后访问根节点，最后递归地做右子树的中序遍历。

2. 应用：使用中序遍历打印与二叉树相关的算术表达式。

3. 递归实现：

   ```c++
   //c++
   template <typename T, typename VST> //元素类型、操作器
   void travIn_R ( BinNodePosi<T> x, VST& visit ) { //二叉树中序遍历算法（递归版）
       if ( !x ) return;
       travIn_R ( x->lc, visit );//注意到，这里已经不是尾递归了
       visit ( x->data );
       travIn_R ( x->rc, visit );
   }
   ```

   ```javascript
   //javascript
   class TreeNode {
       constructor(value, left = null, right = null) {
           this.value = value;
           this.left = left;
           this.right = right;
       }
   }
   
   function inOrderTraversal(root) {
       if (root === null) {
           return;
       }
   
       inOrderTraversal(root.left); // 遍历左子树
       console.log(root.value); // 访问根节点
       inOrderTraversal(root.right); // 遍历右子树
   }
   ```

   

4. 对中序遍历的思考和观察：

   > 1. **第一个访问的节点究竟是哪一个？**
   >
   >    在中序遍历中，第一个访问的节点是整个二叉树**最左侧**的节点，也就是**所有节点中深度最大的左子节点。**和先序遍历刚开始时的情况类似：在中序遍历中，先是转让“控制权”，直至转让到没有左孩子的节点。在先序遍历中，也是先向左向下移动，直到找到第一个没有左孩子的节点。
   2. “藤缠树”的思想：顺着最左侧通路，自底而上依次访问沿途各节点及其右子树。算法沿最左侧通路自底而上，以沿途各节点为界，中序遍历序列可分解为$d + 1$段，各段彼此独立，且均包括访问来自最左侧通路的某一节点$L_k$，以及遍历其对应的右子树$T_k$​。 

      ```
      inorder(T) = visit(Ld), inorder(Td);
      				visit(Ld-1),inorder(Td-1);
      					..., ...;
      						visit(L1), inorder(T1);
      							visit(L0),inorder(T0)
      ```

   3. 迭代实现：（版本1）

      ```c++
      template <typename T> //从当前节点出发，沿左分支不断深入，直至没有左分支的节点
      static void goAlongVine( BinNodePosi<T> x, Stack<BinNodePosi<T>>& S ) {
         while ( x ) { S.push( x ); x = x->lc; } //当前节点入栈后随即向左侧分支深入，迭代直到无左孩子
      }
      //需要LIFO的数据结构——栈，开口向下。
      template <typename T, typename VST> //元素类型、操作器
      void travIn_I1( BinNodePosi<T> x, VST& visit ) { //二叉树中序遍历算法（迭代版#1）
         Stack<BinNodePosi<T>> S; //辅助栈
         while ( true ) {
            goAlongVine( x, S ); //从当前节点出发，逐批入栈
            if ( S.empty() ) break; //直至所有节点处理完毕
            x = S.pop(); visit( x->data ); //弹出栈顶节点并访问之
            x = x->rc; //转向右子树（此句较为重要）
         }
      }
      ```

   4. 该算法的效率——分摊分析：

      1. `goAlongVine()`最多要调用$\Omega(n)$次，单次调用，最多需要做$\Omega(n)$次`push()`，但是，这个算法最多仅仅是$O(n)$。
      2. 每一次调用`goAlongVine()`后都恰好有一次`pop()`操作，全程不超过$O(n)$次。`push()`的次数不确定，但是，累计应该与`pop()`一样多。
      3. 在`goAlongVine()`操作中，所有的`push()`操作，就恰好对应着左侧链上的长度。这个次数不会超过所有的节点数。

   5. 延伸：对于二叉搜索树来说，中序遍历的作用至关重要。相关算法必需的一项基本操作，就是定位任一节点在中序遍历序列中的直接后继。在中序遍历意义下的直接后继就是最靠左的右后代（如果有右孩子）或者最低的左祖先（如果没有右孩子）。这样，可以实现相关的`succ()`接口：

      ```c++
      template <typename T> BinNodePosi<T> BinNode<T>::succ() { //定位节点v的直接后继
          BinNodePosi<T> s = this; //记录后继的临时变量
          if ( rc ) { //若有右孩子，则直接后继必在右子树中，具体地就是
              s = rc; //右子树中
              while ( HasLChild( *s ) ) s = s->lc; //最靠左（最小）的节点
          } else { //否则，直接后继应是“将当前节点包含于其左子树中的最低祖先”，具体地就是
              while ( IsRChild( *s ) ) s = s->parent; //逆向地沿右向分支，不断朝左上方移动
              s = s->parent; //最后再朝右上方移动一步，即抵达直接后继（如果存在）
          }
          return s;
      }
      ```

      > 1. 若当前节点有右孩子，则其直接后继必然存在，且属于其右子树。此时只需转入右子树，再沿该子树的最左侧通路朝左下方深入，直到抵达子树中最靠左（最小）的节点。
      > 2. 反之，若当前节点没有右子树，则若其直接后继存在，必为该节点的某一祖先，且是将当前节点纳入其左子树的最低祖先。于是首先沿右侧通路朝左上方上升，当不能继续前进时，再朝右上方移动一步即可。
      > 3.   作为后一情况的特例，出口时`s`可能为`NULL`。这意味着此前沿着右侧通路向上的回溯，抵达了树根。也就是说，当前节点全树右侧通路的终点——它也是中序遍历的终点，没有后继。

      对上述迭代版本的代码，使用`succ()`接口进行改写，得到新的版本。我们可以借助该版本，实现需要空间更少的算法。

      版本3：以上的迭代式遍历算法都需使用辅助栈，尽管这对遍历算法的渐进时间复杂度没有实质影响，但所需辅助空间的规模将线性正比于二叉树的高度，在最坏情况下与节点总数相当。为此，可对代码继续改进，借助`BinNode`对象内部的`parent`指针。该版本无需使用任何结构，总体仅需O(1)的辅助空间，属于就地算法。当然，因需要反复调用`succ()`，时间效率有所倒退。

      ```c++
      template <typename T, typename VST> //元素类型、操作器
      void travIn_I3( BinNodePosi<T> x, VST& visit ) { //二叉树中序遍历算法（迭代版#3，无需辅助栈）
         bool backtrack = false; //前一步是否刚从左子树回溯——省去栈，仅O(1)辅助空间
         while ( true ){
            if ( !backtrack && HasLChild( *x ) ) //若有左子树且不是刚刚回溯，则
               x = x->lc; //深入遍历左子树
            else { //否则——无左子树或刚刚回溯（相当于无左子树）
               visit( x->data ); //访问该节点
               if ( HasRChild( *x ) ) { //若其右子树非空，则
                  x = x->rc; //深入右子树继续遍历
                  backtrack = false; //并关闭回溯标志
               } else { //若右子树空，则
                  if ( !( x = x->succ() ) ) break; //回溯（含抵达末节点时的退出返回）
                  backtrack = true; //并设置回溯标志
               }
            }
         } 
      }
      ```
      
      

### 后序遍历(Postorder Traversal)

1. 在后序遍历中，我们首先递归地做左子树和右子树的后序遍历，然后访问根节点。

2. 应用：计算目录及其子目录中的文件所用空间。

3. 递归实现：

   ```c++
   //c++
   template <typename T, typename VST> //元素类型、操作器
   void travIn_R ( BinNodePosi<T> x, VST& visit ) { //二叉树中序遍历算法（递归版）
       if ( !x ) return;
       travIn_R ( x->lc, visit );
       travIn_R ( x->rc, visit );//两个都不是尾递归
       visit ( x->data );
   }
   ```

   ```python
   # python
   class TreeNode:
       def __init__(self, x):
           self.val = x
           self.left = None
           self.right = None
   
   def postorder_traversal(root):
       if root is None:
           return
   
       postorder_traversal(root.left)  # 遍历左子树
       postorder_traversal(root.right)  # 遍历右子树
       print(root.val)  # 访问根节点
   ```

   

4. **“藤缠树”**的思想：从根出发下行，对于**每一个节点**，如果其有左孩子，就继续向左分支；如果其没有左孩子，则向右侧分支。注意：向左分支是**尽可能**的行为。如果对于某一个节点来说，其既没有左孩子，又没有右孩子，则`visit()`开始执行。所以，在后序遍历的迭代版本中，最**靠左的叶子节点**（中序遍历中次序最靠左者）是**第一个被访问**的节点。这里，树的藤上依然会挂上某些右子树，但是不是所有的节点都挂上了右子树，因为对于某些只有右孩子的节点来说，它的右子树在藤上面。

5. **最高左侧可见叶节点(Highest Leaf Visible From Left, HLVFL)**：如果我们将树`T`画在一个二维平面上，并假设所有节点和边均不透明。于是从左侧水平向右看去，未被遮挡的最高叶节点`v`，称作最高左侧可见叶节点。注意，该节点既可能是左孩子，又可能是右孩子。

   ```
       1
      / \
     2   3
    / \ / \
   4  5 6  7
        / \
       8   9
   ```

   入栈示意图，左侧是栈顶：`1->45231->5231->231->896731->96731->6731->731->31->1->Null`

   其后序遍历序列为`4, 5, 2, 8, 9, 6, 7, 3, 1`。

6. 迭代实现：

   ```c++
   template <typename T> //在以S栈顶节点为根的子树中，找到最高左侧可见叶节点
   static void gotoLeftmostLeaf( Stack<BinNodePosi<T>>& S ) { //沿途所遇节点依次入栈
      while ( BinNodePosi<T> x = S.top() ) //自顶而下，反复检查当前节点（即栈顶）
         if ( HasLChild( *x ) ) { //尽可能向左
            if ( HasRChild( *x ) ) S.push( x->rc ); //若有右孩子，优先入栈
            S.push( x->lc ); //然后才转至左孩子
         } else //实不得已
            S.push( x->rc ); //才向右
      S.pop(); //返回之前，弹出栈顶的空节点
   }
   
   template <typename T, typename VST>
   void travPost_I( BinNodePosi<T> x, VST& visit ) { //二叉树的后序遍历（迭代版）
      Stack<BinNodePosi<T>> S; //辅助栈
      if ( x ) S.push( x ); //根节点入栈
      while ( !S.empty() ) { // x始终为当前节点
         if ( S.top() != x->parent ) ////若栈顶非x之父（而为右兄）
            gotoLeftmostLeaf( S ); //则在其右兄子树中找到HLVFL（相当于递归深入）
         x = S.pop(); 
         visit( x->data ); //弹出栈顶（即前一节点之后继），并访问之
      }
   }
   ```

7. 时间复杂度：仍然是线性的时间复杂度$O(n)$。使用摊还分析中的**聚集分析**和**记账法**可以进行分析。 

   > 对于后序遍历二叉树，我们可以将其看作是三个操作的集合：
   >
   > 访问左子树、访问右子树、访问当前节点。
   >
   > 每个操作都会被执行 n 次，其中 n 是二叉树的节点数量。因此，整个后序遍历的时间复杂度是 $O(n)$。
   >
   > 这是因为无论二叉树的形状如何（即使是最坏的情况，也就是所有的节点都只有左子节点或只有右子节点，形成一条线），后序遍历都需要访问每个节点一次。因此，总的操作次数是 n，所以时间复杂度是$ O(n)$。

8. 后序遍历的实际应用：表达式树、逆波兰表达式

   1. 在遇到一个中缀表达式的时候，我们先通过添加适当的括号来直接表明各个运算符的优先级。
   2. 按照左括号向下，右括号向上的原则（这样做可以区分出层次结构，便于直观地进行计算和构建树），对该中缀表达式进行排列。
   3. 保留数字和运算符的位置不变，使用恰当的连接线，将树构建出来。这里，每一个数字都是叶子节点，而运算符则为内部节点。
   4. 然后进行对构建好的表达式树的后序遍历。注意，双目运算符下面必有左孩子和右孩子，而单目运算符下面只有左孩子。
   5. 按照运算规则进行后序遍历，即可得到最终答案。（后序遍历的序列就是RPN）

### 层次遍历(Traversal by Level)

1. 它按照**树的层次**从上到下，同一层次从左到右的顺序访问每一个节点。具体来说，首先访问根节点，然后访问所有的第二层节点（即根节点的直接子节点），接着访问所有的第三层节点（即第二层节点的直接子节点），以此类推，直到所有的节点都被访问过。这种遍历方式常常用于**广度优先搜索（Breadth-First Search，BFS）算法**，因为它首先访问的是离根节点最近的节点。
2. 使用**队列**进行层次遍历：

```c++
template <typename T> template <typename VST> //元素类型、操作器
void BinNode <T>::travLevel( VST& visit ) { //二叉树层次遍历算法
   Queue<BinNodePosi<T>> Q; 
   Q.enqueue( this ); //引入辅助队列，根节点入队
   while ( !Q.empty() ) { //在队列再次变空之前，反复迭代
      BinNodePosi<T> x = Q.dequeue(); visit( x->data ); //取出队首节点并访问之
      if ( HasLChild( *x ) ) Q.enqueue( x->lc ); //左孩子入队
      if ( HasRChild( *x ) ) Q.enqueue( x->rc ); //右孩子入队
   }
}
```
# 二叉树基础和基本实现

## 有序树、m叉树

1. 在树T中，若在每个节点的所有孩子之间都可以定义某一线性次序，则称T为一棵“有序树（Ordered tree）”。 
2. 每个内部节点均不超过m度的有序树，称作m叉树。

## 二叉树

1. **二叉树**：每个节点均不超过2度的有序树，称作二叉树（Binary tree）。 在二叉树中，每个节点的孩子可以用左、右区分，分别称作左孩子和右孩子。如果左、右孩子同时存在，则左孩子的次序优先于右孩子。
2. **真二叉树**：不含 1 度节点的二叉树，称作真二叉树（Proper binary tree），否则称作非真二叉树 （Improper binary tree）。
3. 二叉树的相关结论：
4. 在二叉树中，深度为$k$的节点不超过$2^k$个。
      1. 推论 1.高度为$h$的二叉树最多包含$2^{h+1}-1$个节点。
      2. 推论 2.由$n$个节点构成的二叉树，高度至少为$\left\lfloor\log _{2} n\right\rfloor$。
5. 在二叉树中，叶子总是比2度节点多一个。

> 当二叉树是完全二叉树时，结论显然成立。若在第$n-k$层的某个2度节点去除其孩子：若去除一个孩子，则该节点变为1度节点，叶子数目和2度节点的数目均减$2^{k-1}$，成立；若去除两个孩子，则该节点退化为叶子，叶子的数目先减$2^k$后加$1$，2度节点的数目减$2^k-1$，结论仍然成立。

## 完全二叉树和满二叉树

   1. **满二叉树(Full Binary  Tree)**：
      1. 国际定义：在国际上，满二叉树通常被定义为一个所有节点都有0个或2个子节点的二叉树。这意味着没有节点只有一个子节点。这种定义下的满二叉树不一定是完全二叉树。这种定义下，又称为真二叉树。（不含 1 度节点的二叉树，称作真二叉树。）
      2. 国内定义：在中国，满二叉树通常被定义为一个所有层都被完全填满的二叉树，也就是说，除了最后一层，每一层的节点数都达到最大值，且最后一层的节点都靠左排列。这种定义下的满二叉树实际上是完全二叉树。
   2. **完全二叉树(Complete Binary Tree)**：对于一个有n个节点的二叉树，按层序编号（从第一层到第n层，每层从左到右），如果编号为i（1≤i≤n）的节点与同样深度的满二叉树中编号为i的节点在二叉树中位置完全相同，则这个二叉树称为完全二叉树。

   > 简单来说，满二叉树是每个节点都有0个或2个子节点的二叉树，而完全二叉树则是除了最后一层外，每一层都是满的，且最后一层的节点都靠左排列。

   推论：在由固定数目的节点所组成的所有二叉树中，完全二叉树的高度最低。

4. 二叉树的具体实现：和树类似，不同的是每一个节点至多有两个孩子，分别使用`leftChild`和`rightChild`来表示。

5. 基数：设度数为0、1和2的节点，各有$n_0$、$n_1$和$n_2$个。

   1. 边数$E=n-1=n_1+2n_2$（1/2度节点各对应于1/2条入边）

   2. 叶节点数$n_0=n_2+1$，与$n_1$无关

   3. 节点数$n=n_0+n_1+n_2=1+n_1+2n_2$

      > 当$n_1=0$，节点度数均为偶数，不含单分支节点

   4. 真二叉树：通过引入$n_1 + 2n_0$个外部节点，可使原有节点度数统一为2。

      1. 如此，即可将任一二叉树转化为真二叉树（proper binary tree）。
      2. 如此转换之后， 全树自身的复杂度并未实质增加。
      3. 对于红黑树之类的结构，真二叉树可以简化描述、理解、实现和分析。

## 二叉树的某些方法

大部分方法和树的方法相似，如`getSize(), getHeight(), getDepth()`等。这里只列出一些特殊的方法。

### 更新规模记录——`updateSize()`

1. 若当前节点的孩子发生变化，比如原有的某个孩子被删除或者有新的孩子插入，就需要更新当前节点及其祖先的规模记录，以便后续的查询。

2. 原理：若节点`v`的左、右孩子分别为`lc`和`rc`，则`size(v) = 1 + size(lc) + size(rc)`。

3. 结论：一旦左右的子树规模已经确定，我们就可以在$O(1)$​时间内得到以节点v为根的子树规模。然后逆行向上，依次更新各个祖先的规模。

4. 算法实现：

```
   算法：updateSize(v)
   输入：二叉树中任一节点v
   输出：更新v的后代规模记录
   {
       令size(v) = 1 + size(lc) + size(rc);
       若v的父亲p存在，则调用updateSize(p)，递归地更新父亲的规模记录;//尾递归，可改写为迭代形式
   }
```

   ```java
   public void updateSize() {
       size = 1;//当前节点
       if (hasLChild())	size += getLChild().getSize();//左子树的规模
       if (hasRChild())	size += getRChild().getSize();//右子树的规模
       if (hasParent())	getParent().updateSize();//递归更新各个真祖先的规模记录
   }
   ```

5. 算法的效率：若节点`v`的深度为`depth(v)`，则总共新要修改`depth(v)+1`个节点的高度记录。而更新每一个节点的高度记录，只需要执行两次`getHeight()`操作、两次加法和两次取最大操作，不过常数时间，故该算法的总体运行时间为$O(\text{depth(v)}+1)$。

### 更新高度记录——`updateHeight()`

1. 同样地，在孩子发生变化后，也有必要更新当前节点的高度记录。

2. 原理：若`u`是`v`的孩子，则`height(v) >= height(u) + 1`。 且`height(v)=1+max(height(u))`。

3. 结论：只需读出左、右孩子的高度，取二者中的大者，再计入当前节点本身，就 得到了当前节点v的新高度。当然，此后也需要从v出发沿parent引用逆行向上，依次更新各个祖先 的高度记录。

4. 算法实现：

   ```
   算法：updateHeight(v)
   输入：二叉树中任一节点v
   输出：更新v的高度记录 
   { 
       height(v) = 0;//先假设没有左、右孩子 
       若v有左孩子lc，则令：height(v) = Max(height(v), 1 + height(lc)); 
       若v有右孩子lc，则令：height(v) = Max(height(v), 1 + height(rc)); 
       若v的父亲p存在，则调用updateHeight(p)，递归地更新父亲的高度记录;
   }
   ```

   ```java
   public void updateHeight() {
   		height = 0;//先假设没有左、右孩子
   		if (hasLChild())	height = Math.max(height, 1+getLChild().getHeight());//左孩子
   		if (hasRChild())	height = Math.max(height, 1+getRChild().getHeight());//右孩子
   
   		if (hasParent())	getParent().updateHeight();//递归更新各个真祖先的高度记录
   	}
   ```

5. 算法的效率：若节点`v`的深度为`depth(v)`，则总共新要修改`depth(v)+1`个节点的高度记录。而更新每一个节点的高度记录，只需要执行两次`getHeight()`操作、两次加法和两次取最大操作，不过常数时间，故该算法的总体运行时间为$O(\text{depth(v)}+1)$。

### 更新深度记录——`updateDepth()`

1. 在父亲节点发生变化后，也有必要更新当前节点的深度记录。

2. 原理：若`u`是`v`的孩子，则`depth(u) = depth(v) + 1`。

3. 结论：只需读出新的父亲节点的深度，再加上一即得到当前节点新的深度。当然，此后还需要沿着`lChild`和`rChild`引用，逐层向下递归地更新每一后代的深度记录。

4. 算法实现：

   ```
   算法：updateDepth(v)
   输入：二叉树中任一节点v
   输出：更新v的深度记录
   {
       若v的父亲节点p存在，则令depth(v) = depth(p)+1;
       否则，令depth(v) = 0;
       若v的左孩子lc存在，则调用updateDepth(lc);//沿孩子引用逐层向下，
       若v的右孩子rc存在，则调用updateDepth(rc);//递归地更新所有后代的深度记录
   }
   ```

   ```java
   public void updateDepth() {
       depth = hasParent() ? 1+getParent().getDepth() : 0;//当前节点
       if (hasLChild())	getLChild().updateDepth();//沿孩子引用逐层向下，
       if (hasRChild())	getRChild().updateDepth();//递归地更新所有后代的深度记录
   }
   ```

5. 算法的效率：若节点v的后代规模为`size(v)`，则总共需要修改`size(v)`个节点的深度记录。鉴于单个节点的深度记录可以在常数时间内得到更新，故 `updateDepth()`算法的总体运行时间为 $O(\text{size}(v))$。

### 分离子树——`secede()`

1. 为了简化二叉树动态操作的实现，这里专门设计了一个`secede()`方法。该方法的功能是，将以某一节点为根的子树从母树中分离出来。

2. 算法实现：

   ```
   算法：secede(v)
   输入：二叉树中任一节点v
   输出：将以v为根的子树丛母树中分离出来
   {
       若v有父亲 {
           切断父亲指向v的引用; 
           调用updateSize(v)和updateHeight(v)，更新v及其祖先的规模记录和高度记录;
           切断v指向父亲的引用; 
           调用updateDepth(v)，更新v及其后代的深度记录; 
   	} 
   } 
   ```

   ```java
   //断绝当前节点与其父亲的父子关系
   //返回当前节点
   public BinTreePosition secede() {
       if (null != parent)	{
           if (isLChild())	parent.setLChild(null);//切断父亲指向当前节点的引用
           else parent.setRChild(null);
   
           parent.updateSize();//更新当前节点及其祖先的规模
           parent.updateHeight();//更新当前节点及其祖先的高度
   
           parent = null;//切断当前节点指向原父亲的引用
           updateDepth();//更新节点及其后代节点的深度
       }
       return this;//返回当前节点
   }
   ```

3. 算法的效率：这一算法无非是对节点`v`各执行了一次上述的三种`update`方法。因此，`secede(v)`的算法的运行时间为$O(\text{depth}(v)+\text{size}(v)+1)$。

### 连接父子——`attachL()`和`attachR()`

1. 这一个方法的功能是，将节点`c`作为左或右孩子与节点`v`联接起来。

2. 算法实现：

   ```
   算法：attachL(p, c) 
   输入：两个二叉树节点p与c 
   输出：将c作为左孩子，与p联接起来 
   { 
       若p已经有左孩子lc，则首先调用secede(lc)将其摘除;  
       调用secede(c)，使c及其后代脱离原属母树; 
       设置相应的引用，在p和c之间建立父子关系; 
       调用updateSize(p)和updateHeight(p)，更新节点p及其祖先的规模和高度; 
       调用updateDepth(c)，更新c及其后代节点的深度;
   }
   ```

   ```java
   public BinTreePosition attachR(BinTreePosition c) {
       if (hasRChild())	getRChild().secede();//摘除当前节点原先的右孩子
       if (null != c) {
           c.secede();//c脱离原父亲
           rChild = c;	c.setParent(this);//确立新的父子关系
           updateSize();//更新当前节点及其祖先的规模
           updateHeight();//更新当前节点及其祖先的高度
           c.updateDepth();//更新c及其后代节点的深度
       }
       return this;
   }
   ```

   另外一边的思想和具体代码实现基本一致。

### 直接前驱/后继的定位算法——`getPrev()`和`getNext()`

1. 若规定：“左（右）子树必须完全居于根节点的（左）右侧”，则所有节点在水平轴上投影的自左向右次序，恰好与中序遍历序列的次序吻合。这表明：中序遍历就是按照自左向右的次序访问各个节点。

2. 根据其中序遍历序列S(T)，我们都可以在其中所有节点之间定义出一个线性次序。所以，除首、末节点外，每一节点都有唯一的直接前驱（后继）。如何快速实现定位直接前驱（后继）的算法？

3. 注意到以下事实，我们实际上可以使用该事实来实现直接查找前驱（后继）的定位算法。

   > 二叉树中，除中序遍历序列中的首节点外，任一节点`v`的直接前驱`u`不外乎三种可能： 
   >
   > 1. `v` 没有左孩子，同时`v`是右孩子：此时，`u`就是`v`的父亲节点；  
   > 2. `v`没有左孩子，同时v是左孩子：此时，从`v`出发沿`parent`引用逆行向上，直到第一个是右孩子 的节点`w`，则`u`就是`w`的父亲节点；  
   > 3. `v`有左孩子：此时，从`v`的左孩子出发，沿`rChild`引用不断下行，最后一个（没有右孩子的）节 点就是`u`。

4. 代码实现：根据上述结论，我们可以实现`getPrev()`和`getNext()`算法：

   ```java
   public BinTreePosition getPrev() {
       //若左子树非空，则其中的最大者即为当前节点的直接前驱
       if (hasLChild()) return findMaxDescendant(getLChild());
       //至此，当前节点没有左孩子
       if (isRChild())	return getParent();//若当前节点是右孩子，则父亲即为其直接前驱
       //至此，当前节点没有左孩子，而且是左孩子
       BinTreePosition v = this;//从当前节点出发
       while (v.isLChild()) v = v.getParent();//沿左孩子链一直上升
       //至此，v或者没有父亲，或者是父亲的右孩子
       return v.getParent();
   }
   
   protected static BinTreePosition findMaxDescendant(BinTreePosition v) {
       if (null != v)
       while (v.hasRChild()) v = v.getRChild();//从v出发，沿右孩子链一直下降
       //至此，v或者为空，或者没有右孩子
       return v;
   }
   ```
   
# 二叉搜索树（二叉排序树）

## 循关键码访问
1. 二叉搜索树（Binary Search Tree），也称二叉排序树（Binary Sort Tree）。它或者是一棵空树，或者是具有下列性质的二叉树：
> 1. 若左子树不空，则左子树上所有结点的关键码均小于根结点的关键码；
> 2. 若右子树不空，则右子树上所有结点的关键码均大于根结点的关键码；
> 3. 左、右子树本身又各是一棵二叉搜索树。

2. 循关键码访问（Call-by-key）
    1. 向量、列表并不能兼顾静态查找与动态修改。能否兼顾二者的优点？
    2. 使用键值对（Key-Value）：各数据项依所持关键码而彼此区分。当然，关键码之间也必须同时地支持比较大小和比对是否相等。数据集中的数据项，统一地表示和实现为词条（Entry）形式。
    3. 这里再次强调，为了与有序词典结构的定义一致，这里并不要求二分查找树中各节点的关键码互异。

3. BST的性质：

    1. 顺序性：任一节点均不小于/不大于其左/右后代。（其只是一种局部性的特征，但却可导出BST的整体特征）
    2. 单调性：二叉树T为二分查找树，当且仅当其中序遍历序列是单调非降的。
    3. 证明：考查二叉搜索树中的任一节点r。按照中序遍历的约定，r左（右）子树中的节点（若存在）均应先于（后于）r接受访问。按照二叉搜索树的定义，r左（右）子树中的节点（若存在）均不大于（不小于）r，故中序遍历序列必然在r处单调非降；反之亦然。鉴于以上所取r的任意性，题中命题应在二叉搜索树中处处成立。

4. 思考：BST的定义能否被修改？

    > 二叉搜索树的定义不能将原定义中的“**左（右）后代**”，替换为“**左（右）孩子**”。因为这样的定义会导致二叉搜索树的性质丧失，例如，下图中的二叉树是二叉搜索树。如按照修改后的定义，把图中的3改成30，虽然满足定义2，但已经不是二叉搜索树了。
    >
    > ![](https://blog.nanpuyue.com/usr/uploads/2019/05/487643383.svg)

## BST类的实现

1. BST的实现主要由两部分组成：树节点类BinNode和二叉搜索树类BST。

3. Java实现：

   ```java
   class Node {
       int key;
       Node left, right;
   
       public Node(int item) {
           key = item;
           left = right = null;
       }
   }
   
   class BST {
       Node root;
   
       BST() {
           root = null;
       }
   
       void insert(int key) {
           root = insertRec(root, key);
       }
   
       Node insertRec(Node root, int key) {
           if (root == null) {
               root = new Node(key);
               return root;
           }
           if (key < root.key)
               root.left = insertRec(root.left, key);
           else if (key > root.key)
               root.right = insertRec(root.right, key);
           return root;
       }
   
       void inorder()  {
           inorderRec(root);
       }
   
       void inorderRec(Node root) {
           if (root != null) {
               inorderRec(root.left);
               System.out.println(root.key);
               inorderRec(root.right);
           }
       }
   }
   ```
   
4. C++实现：

   ```cpp
   #include <iostream>
   
   struct Node {
       int key;
       struct Node *left, *right;
   };
   
   struct Node *newNode(int item) {
       struct Node *temp = new Node;
       temp->key = item;
       temp->left = temp->right = NULL;
       return temp;
   }
   
   void inorder(struct Node *root) {
       if (root != NULL) {
           inorder(root->left);
           std::cout << root->key << std::endl;
           inorder(root->right);
       }
   }
   
   struct Node* insert(struct Node* Node, int key) {
       if (Node == NULL) return newNode(key);
       if (key < Node->key)
           Node->left  = insert(Node->left, key);
       else if (key > Node->key)
           Node->right = insert(Node->right, key);
       return Node;
   }
   ```
## BST的方法
BST具有独特的访问数据的方法，即循关键码访问(Call-By-Key)。这主要是通过三种接口完成的，即静态的查找、动态的插入和删除。这三种接口的实现方法是BST的基本方法。

### 查找
- BST的查找方法：从根节点出发，逐步地缩小查找范围，直到发现目标（成功），或抵达空树（失败）。对照中序遍历序列可见，整个过程可视作是在仿效有序向量的二分查找。（减治）算法的时间复杂度：$O(h)$，其中$h\in[\log n,n]$是BST的深度。

- 算法实现：

   ```
   算法：binSearch(v, key)
   输入：二叉树中的节点v，一个关键码key
   输出：在以v为根节点的（子）树中，找出关键码为key的节点；若不存在这样的节点，则返回最后被访问的节点
   要求：首次调用时，v为树根节点
   {
       置当前节点u = v;
       不断地迭代 {
           将当前节点u与目标关键码key做比较;
           若目标关键码更小，则
           若u有左孩子，则令u = u.lChild;
           否则（查找失败），直接返回u;
           否则，若若目标关键码更大，则
           若u有右孩子，则令u = u.rChild
           否则（查找失败），直接返回u;
           否则（查找命中），直接返回u;
   	}
   }
   ```

- 算法的有效性证明：
   - 正确性：在`binSearch()`算法中每次深入左（右）子树时，被忽略的右（左）子树必然不含目标节点。

   - 确定性：BST允许多个节点拥有相等的关键码。`binSearch()`算法总能返回最靠近根节点的节点。

      > 引理：
      >
      > 在任一二分查找树T中，若至少存在一个关键码为key的节点，则这些节点中深度最小者必然唯一，而`binSearch()`算法找出的正是这一节点。
      >
      > 证明：
      >
      > - 首先证明，若存在关键码相同的节点，则深度最小者必唯一。（反证法）
      >
      >    否则，任取深度最小的两个节点`u`和`v`，令`c`为它们的最低共同祖先。显 然，`u ≠ c ≠ v`，而且`u`和`v`不可能处于`c`的同一侧子树（如果是在同一侧的情况下，命题直接得证）。不失一般性，设`u`、`v`分别属于`c`的左、 右子树。于是便有`key = key (u) ≤  key(c)  ≤  key(v)  =  key`。这与`u`、`v`是关键码为`k ey`的深度最小节点矛盾。
      >
      > - 接下来，我们注意到以下事实：在对二分查找树的查找过程中，接受比较的各个节点的深度比然不断递增。因此，在关键码为key的所有节点中，唯一的那个深度最小的节点必然首先接收比较，并使得算法以成功告终。

- 代码实现：

    ```java
    class Node {
        int key;
        Node left, right;
    
        public Node(int item) {
            key = item;
            left = right = null;
        }
    }
    
    class BinarySearchTree {
        Node root;
        Node search(Node root, int key) {
            if (root == null || root.key == key)
                return root;
    
            if (root.key < key)
                return search(root.right, key);
    
            return search(root.left, key);
        }
    }
    ```

- 算法的语义：对于返回的引用值：查找成功时，指向一个关键码为e且真实存在的节点；失败时，指向最后一次试图转向的空节点NULL——随后可视需要进行修改。此时，不妨假想地将该空节点转换为一个数值为e的哨兵节点。

### 插入

- 为了在二分查找树中插入一个节点，我们需要根据其关键码key，利用查找算法`binSearch()`确定插入的位置及方向，然后才将新节点作为叶子插入。

- 算法实现：

   ```
   算法：insert(v, key, value) 
   输入：以v为根节点的子树，关键码key以及数据value 
   输出：将条目(key, value)插入二分查找树中，并返回该条目 
   { 
       若当前的树为空，则生成并返回一棵包含单节点(key, value)的二分查找树; 
       p = 树根v; 
       while (true) { 
           调用binSearch(p, key), 在以p为根节点的子树中查找关键码为key的最高节点 
           若key ≠ p.key，则根据二者的大小关系将新节点作为p的右或左孩子插入，并返回新节点 
           若key = p.key，则有两种可能： 
           若p没有左或右孩子，则将新节点作为p的左或右孩子插入，并返回新节点; 
           若p已有两个孩子，则取p = p.lChild; 
       } 
   }
   ```

- 算法的正确性和确定性容易证明。因为该算法和查找算法大体上十分相似。
- 插入算法的时间复杂度：$O(h)$，其中$h\in[\log n,n]$是BST的深度。

### 删除

- 为了从二分查找树中删除关键码为key的节点，我们首先也需要通过算法`binSearch()`判断树中是否的确存在这样的节点；如果存在，我们要确定其位置，然后才能将其摘除。

- 算法实现：

   ```
   算法：remove(r, key) 
   输入：子树r，关键码key 
   输出：若在以r为根节点的子树中存在关键码为key的节点，则删除它，并返回其中存放的条目 
   { 
       调用binSearch(r, key), 在子树r中查找关键码为key的最高节点 
       若key ≠ v.key，则说明目标节不存在，故返回null; 
       若v没有左孩子，则 
       	摘除v，代之以v的右孩子; 
       否则 { 
           在v的左子树中找出其直接前驱w; //请注意，w必然没有右孩子 
           将v与w交换位置; 
           摘除v，代之以v的左孩子; 
       } 
   } 
   ```

- 删除算法的时间复杂度：$O(h)$，其中$h\in[\log n,n]$是BST的深度。

>  简而言之，有：
>
> 1. The node to be deleted is a node without non-leaf child: just delete it directly.
> 2. The node to be deleted has a single non-leaf child: Connect the child with the parent of the node-to-delete
> 3. The node to be deleted has a two non-leaf children:
> 	1. The node to be deleted has a two non-leaf children;
> 	2. Delete the node with the max key from its left subtree.

> 在含n个节点的二叉搜索树（BST）中进行查找的最坏时间复杂度为O(n)。
>
> 这种情况发生在二叉搜索树退化为链表的情况下，即所有节点都只有左子节点或只有右子节点。在这种情况下，查找操作需要遍历所有的节点，因此时间复杂度为O(n)。
> 

# BST的平衡与等价

## 二分查找树的平均性能

1. BST在最坏情况下，都需要$O(n)$时间才能完成，就这一点而言，似乎和字典结构无甚差别。现在我们主要来考察二分查找的平均性能。我们将针对两种随机统计的口径（随机生成、随机组成），给出BST的平均性能。

2. **随机生成**：

   1. 考虑关键码互异的$n$个条目，对于这些条目的任何一种全排列$\sigma$，若从空树开始，依次调用`insert()`算法，将$\sigma$中的关键码插入，都可以得到这n个条目的一棵二叉搜索树`T(σ)`。我们有如下定义：

      > 与随机排列 $\sigma$ 相对应的二分查找树`T(σ)`，称作由 $\sigma$ 生成的二分查找树。

   2. 对于任意 $n$ 个互异关键码，总共有 $n!$ 个全排列。如果假定这 $n!$ 个排列作为输入的概率均等，则只要将它们各自生成的二分查找树的平均查找长度进行平均，所得到的总体平均查找长度将能反映二分查找树的平均查找性能。

      > 由 $n$ 个互异条目随机生成的BST，平均查找长度为 $O(\log n)$ 。

3. **随机组成**：

   1. 一点思考：同一组条目的不同排列所生成二分查找树有可能雷同。所以，对所有 $n!$ 个随机排列进行平均，并不能反映二分查找树的平均查找性能。根据以上的分析，$n$ 个条目组成的BST的数目，将远远小于 $n!$ 。

   2. 一点改进：假定树中的n个节点是给定的，然后在中序遍历次序保持一致的前提下，统计它们能够构成的二分查找树的数目。

      > 观察结论：在保持中序遍历次序的前提下，由n个互异节点构成的每棵二叉树，都是一棵二分查找树（称作由这些节点组成的一棵二分查找树）。

   3. 那么，这些拓扑结构互异的BST有多少？

      > 定理：由n个互异节点组成的二分查找树，总共有 $\dfrac{(2n)!}{n!(n+1)!}$ 棵。

      

      > 推导：将n个互异节点组成的二分查找树的总数记为 $T(n)$ 。尽管由同一组节点组成的二叉搜索树不尽相同，但它们的中序遍历序列却必然相同，不妨记作：
      >
      > 
      > $$
      > \left[x_0,x_1,\ldots,x_{k-1}\right],x_k,\left[x_{k+1},\ldots,x_{n-1}\right]
      > $$
      > 
      >
      > 根据所取的树根节点不同，所有的搜索树可以分为 $n$ 类。如果以 $x_k$ 为根节点，则其左右子树见上。
      >
      > 对应的递推关系如下：
      >
      > 
      > $$
      > T(0)=T(1)=1,\quad T(n)=\sum\limits_{k=0}^{n-1}T(k)\cdot T(n-k-1)
      > $$
      > 
      >
      > （边界条件很好理解，而递推式可以以减治的方式进行理解：对于某一棵BST，其由左、右两个子树和根节点组成。因为中序序列确定，所以它们三者的排序方式不会改变，即依然是左子树——根节点——右子树组成。于是，问题转化成求 $T(L),T(R)$ ，其中，$L,R$ 分别是构成这两棵子树的元素总个数，且 $n=L+R+1$。同时，根据乘法原理，可以推出递推式。）
      >
      > 这是典型的Catalan数式递推关系，解之即得题中结论。

   4. 若假定这些树出现的概率相等，则通过对它们各自的平均查找长度进行平均，也可以得到一个总体的平均指标。

      > 由 $n$ 个互异条目随机生成的BST，平均查找长度为 $O(\sqrt n)$ 。

4. 两种衡量方式的比较：同一组关键码的不同排列所生成的二叉搜索树，未必不同。第一种情况下，某些BST被统计的次数较多。而实际上，越是平衡的树，被统计的次数亦越多。从这个角度讲，前一种平均的方式，在无形中高估了二叉搜索树的平均性能。因此，相对而言，第二种更加可信。

## BST：理想平衡与渐进平衡

1. 理想平衡：节点数目固定时，兄弟子树的高度越接近（平衡），全树也将倾向于更低。由n个节点组成的理想平衡二叉树，高度的下界为$\lfloor\log_2 n\rfloor$。当达到该下界时，称作理想平衡。这大致相当于完全树甚至满树：叶节点只能出现于最底部的两层——条件过于苛刻。
2. 渐进平衡（适度平衡）：理想平衡出现的概率极低，因此我们更多地关注渐进平衡。对于一棵BST，如果其高度渐进地不超过为$O(\log n)$，则称作渐进平衡。这种情况也在我们可以接受的范围内。
3. 平衡二叉搜索树（Balanced BST）：对于一棵BST，如果其满足渐进平衡，我们称之为平衡二叉搜索树。

## 中序歧义和等价变换

1. 中序歧义：对于两棵拓扑结构不同的二叉树，它们的中序遍历序列可能相同。这种现象称作中序歧义。中序歧义的存在，使得我们无法通过中序遍历序列来唯一确定一棵二叉树。

2. 等价BST：对于两棵拓扑结构不同的二叉树，如果它们的中序遍历序列相同，则称这两棵树是等价的。等价BST有两个重要性质：
   1. 上下可变：联接关系不尽相同，承袭关系可能颠倒。
   2. 左右不乱：中序遍历序列完全一致，全局单调非降。

3. 局部性和等价变换：平衡二叉搜索树的适度平衡性，都是通过对树中每一局部增加某种限制条件来保证的。比如，在红黑树中，从树根到叶节点的通路，总是包含一样多的黑节点；在AVL树中，兄弟节点的高度相差不过1。这些条件设定的比较精妙，除了保证适度平衡性，还具有以下的局部性：

   > 1. 经过单次动态修改操作后，至多只有$O(1)$处局部不再满足限制条件；
   > 2. 总可在$O(\log n)$时间内，使这$O(1)$处局部（以至全树）重新满足限制条件。

   这就意味着：刚刚失去平衡的二叉搜索树，必然可以迅速转换为一棵等价的平衡二叉搜索树。等价二叉搜索树之间的上述转换过程，也称作**等价变换**。

4. 旋转调整：最基本的修复手段，就是通过围绕特定节点的旋转，实现等价前提下的局部拓扑调整。关于二叉平衡树，平衡的调整操作分为包括左旋（Left Rotate 或者 zag）和右旋（Right Rotate 或者 zig） 两种。由于二叉平衡树在调整时需要保证中序遍历序列不变，这两种操作均不改变中序遍历序列。
   1. zig（右旋）：右旋也称为「右单旋转」或「LL 平衡旋转」。对于结点 A 的右旋操作是指：将 A 的左孩子 B 向右上旋转，代替 A 成为根节点，将 A 结点向右下旋转成为 B 的右子树的根结点，B 的原来的右子树变为 A 的左子树。对于右旋操作一般的更新顺序是：暂存 B 结点（新的根节点），让 A 的左孩子指向 B 的右子树 T2，再让 B 的右孩子指针指向 A，最后让 A 的父结点指向暂存的 B。
   
   2. zag（左旋）：左旋也称为「左单旋转」或「RR 平衡旋转」。对于结点 A 的左旋操作是指：将 A 的右孩子 B 向左上旋转，代替 A 成为根节点，将 A 结点向左下旋转成为 B 的左子树的根结点，B 的原来的左子树变为 A 的右子树。对于左旋操作一般的更新顺序是：暂存 B 结点（新的根节点），让 A 的右孩子指向 B 的左子树 T2，再让 B 的左孩子指针指向 A，最后让 A 的父结点指向暂存的 B。
   
   3. zig/zag操作只改变了三组结点关联，相当于对三组边进行循环置换一下，因此需要暂存一个结点再进行轮换更新。
   
      ![bst-rotate](https://oi-wiki.org/ds/images/bst-rotate.svg)
   
   4. 四种平衡性被破坏的情况：

      LL 型：T 的左孩子的左子树过长导致平衡性破坏。

      调整方式：右旋节点 T。

      ![bst-LL](https://oi-wiki.org/ds/images/bst-LL.svg)

      RR 型：与 LL 型类似，T 的右孩子的右子树过长导致平衡性破坏。

      调整方式：左旋节点 T。

      ![bst-RR](https://oi-wiki.org/ds/images/bst-RR.svg)

      LR 型：T 的左孩子的右子树过长导致平衡性破坏。

      调整方式：先左旋节点 L，成为 LL 型，再右旋节点 T。

      ![bst-LR](https://oi-wiki.org/ds/images/bst-LR.svg)

      RL 型：与 LR 型类似，T 的右孩子的左子树过长导致平衡性破坏。

      调整方式：先右旋节点 R，成为 RR 型，再左旋节点 T。

      ![bst-RL](https://oi-wiki.org/ds/images/bst-RL.svg)
## 补充

1. 由 $n$ 个互异条目随机生成的BST，平均查找长度为 $O(\log n)$ ；由 $n$ 个互异条目随机生成的BST，平均查找长度为 $O(\sqrt n)$  的证明详见

   > L. Devroye. A Note on the Height of Binary Search Trees. J. of ACM (1986), 33(3):489-498
   >
   > P. Flajolet & A. Odlyzko. The Average Height of Binary Trees and Other Simple Trees.  Journal of Computer and System Sciences (1982), 25(2):171-213


## 参考资料
1. [二叉搜索树 & 平衡树 - OI Wiki (oi-wiki.org)](https://oi-wiki.org/ds/bst/)

# 平衡二叉树的实例——AVL树

## AVL树简介

1. AVL树（Adelson-Velsky and Landis Tree）是最先发明的自平衡二叉查找树(1962)。在AVL树中任何节点的两个子树的高度最大差别为1，所以它也被称为高度平衡树。查找、插入和删除在平均和最坏情况下的时间复杂度都是$O(\log n)$。

   AVL树的例子：

   ```
       30
      /  \
    15   40
   /  \    \
   10  20   50
   ```

   > **AVL Tree** is a self balancing binary search **tree**, where difference of right subtree and left subtree height to a node is at most 1.

2. 平衡因子：在BST中，任一节点$v$的平衡因子都定义为“其左、右子树的高度差”，记作$balFac(v)=h(ls) - h(rs)$。这里强调，空树的高度为$-1$。

3. AVL树的性质：

   1. 空二叉树是一个 AVL 树；

   2. 如果 T 是一棵 AVL 树，那么其左右子树也是 AVL 树，并且 $|h(ls) - h(rs)| \leq 1$，h 是其左右子树的高度；

   3. （适度平衡）树高为 $O(\log n)$。（反过来，等价于高度为$h$的AVL树，至少包含$Fib(h+3)-1$个节点）

   >对树高为 $O(\log n)$的证明：设 $f_n$ 为高度为 $n$ 的 AVL 树所包含的最少节点数，则有
   >
   >
   >$$
   >f_n=
   >\begin{cases}
   >1&(n=1)\\
   >2&(n=2)\\
   >f_{n-1}+f_{n-2}+1& (n>2)
   >\end{cases}
   >$$
   >
   >
   >
   >根据常系数非齐次线性差分方程的解法，$\{f_n+1\}$ 是一个斐波那契数列。这里 $f_n$ 的通项为：
   >
   >
   >$$
   >f_n=\frac{5+2\sqrt{5}}{5}\left(\frac{1+\sqrt{5}}{2}\right)^n+\frac{5-2\sqrt{5}}{5}\left(\frac{1-\sqrt{5}}{2}\right)^n-1
   >$$
   >
   >
   >
   >斐波那契数列以指数的速度增长，对于树高 $n$ 有：
   >
   >
   >$$
   >n<\log_{\frac{1+\sqrt{5}}{2}} (f_n+1)<\frac{3}{2}\log_2 (f_n+1)
   >$$
   >
   >
   >
   >因此 AVL 树的高度为 $O(\log f_n)$，这里的 $f_n$ 为结点数。
   
   注意：完全二叉树一定是AVL树，但AVL树不一定是完全二叉树。

## AVL树：失衡和重新平衡

1. AVL树的失衡与重新平衡：AVL树和普通BST最大的不同就是是否满足平衡因子的绝对值小于等于1。对于上述的例子，如果我们要插入元素`25`，则会引起`25`的祖父和曾祖父失衡。此时，该树不再满足AVL树的条件。同样的，如果我们删除某一个节点，也会引起某些节点的失衡。

## 插入——旋转算法

1. AVL树的普通插入方法，和普通BST树的方法完全一致。不过，使用单纯的插入算法，可能会使得原AVL树不满足BBST的条件，而退化成BST。为此，我们必须要升级该方法为插入——旋转算法。

2. **失衡节点集（插入）**：一般地，若在插入新节点$x$之后AVL树T失去平衡，则可以将失衡的节点组成集合$U_T(x)$，则有如下的事实：

   > $U_T(x)$中的每个节点都是x的祖先，且高度不低于x的祖父。
   >
   > （推论）$U_T(x)$中各节点的深度互异，且该集合的规模不超过$\text{depth}(x)-1$。若取$g(x)$为$U_T(x)$中最深的节点，则$g(x)$必然存在且唯一。

3. 思路：为了修正失衡的现象，我们从$x$出发逆行向上，依次检查x各层祖先的平衡因子，直到发现$g(x)$。在$x$与$g(x)$之间的通路上，设$p$为$g(x)$的孩子，$v$为$p$的孩子，则由上述推论，得到：$p$必是$x$的真祖先，而$v$是$x$的祖先。此后，根据祖孙三代$g(x),p,v$的位置关系，对$g(x)$和$p$进行旋转，使得树的局部恢复平衡。这一操作也会使得整棵树的平衡性也得到恢复。

4. 单旋(zig, zag)：假设$v$是$p$的左（右）孩子，且$p$也是$g$的左（右）孩子。这种情况下，一定是由于在子树$v$插入新节点$x$而导致$g$失衡。此时，我们只需要对$g$进行一次顺时针/zig旋转（逆时针/zag旋转）即可。（概括下来，就是：失衡节点获得其孩子的高度较低的子树，然后该失衡节点变为其孩子的子树，局部树的根节点变为原失衡节点的孩子。由于经过插入——旋转操作后整棵树的深度和之前一样，所以其他节点不受影响，而那一系列的失衡节点也同时重新平衡。）

   ![bst-RR](https://oi-wiki.org/ds/images/bst-RR.svg)

5. 双旋(zig-zag, zag-zig)：假设$v$是$p$的右（左）孩子，而$p$是$g$的左（右）孩子。这种情况下，一定是由于在子树$v$插入新节点$x$而导致$g$失衡。此时，我们需要对$p$进行一次逆时针/zag（顺时针/zig）旋转，然后再对$g$进行一次顺时针/zig（逆时针/zag）旋转。（概括下来，就是：失衡节点的孩子先获得其孩子的某棵子树（这里，对子树的选取必须符合BST的有关规则），而后，以该失衡节点的孩子为节点所构成的树变成失衡节点的孙子的一棵子树；现在，情况已经变成了一个单旋的操作，祖孙三代按辈分依次是$g,v,p$。最后经过一次单旋的操作，局部树的根节点变为原失衡节点的孙子。也就是说，孙子在两次旋转操作中，逐渐向根节点靠拢。）

   ![bst-LR](https://oi-wiki.org/ds/images/bst-LR.svg)

5. 插入——旋转算法的效果和效率：我们有以下结论：

   > 1. 在AVL树中插入节点x后，若g(x)是失衡的最低节点，则经过上述单旋或双旋调整之后，不仅能使局部重新平衡同时高度也复原，而且整棵树也将重获平衡。
   > 2. 在AVL树中插入一个节点后，至多只需经过两次旋转即可使之恢复平衡。
   > 3. AVL 树的节点插入操作可以在$O(\log n)$时间内完成。

## 删除——旋转算法

1. AVL树的删除算法：删除节点$x$的操作和BST树的删除算法一致。不过，删除操作可能会导致某个节点失衡。为此，我们需要升级删除算法为删除——旋转算法。

2. **失衡节点集（删除）**：一般地，若在删除节点$x$之后AVL树T失去平衡，则可以将失衡的节点组成集合$V_T(x)$，则有如下的事实：（注意此处的表述和上文不完全一致）

   > $V_T(x)$中的每个节点都是x的祖先。
   >
   > （推论）$V_T(x)$中各节点的深度互异。若取$g(x)$为$V_T(x)$中最深的节点，则$g(x)$必然存在且唯一。

3. 删除——旋转算法的思路：为了修正失衡的现象，我们从$x$出发逆行向上，依次检查x各层祖先的平衡因子，直到发现$g(x)$。其左、右孩子的高度应至少相差2，我们将其中的高者记作p。既然p的高度至少为1，故必有孩子。我们按照以下规则在p的孩子中选出节点v：若p的两个孩子不一样高，则取v为其中的高者；否则，取v与p同向（亦即，v与p同为左孩子，或者同为右孩子）。根据祖孙三代节点g(x)、p和v的位置关系，通过对g(x)和p的旋转同样可以使得这一局部恢复平衡。不过，与插入操作不同的是，删除操作后局部平衡的恢复并不意味着整棵树T的平衡也得到恢复。

4. 局部平衡——单旋：假设v是p的左（右）孩子，且p也是g的左（右）孩子。这种情况下，一定是由于在子树v删除节点x而导致g失衡。此时，我们只需要对g进行一次顺时针/zig旋转（逆时针/zag旋转）即可。

5. 局部平衡——双旋：假设v是p的右（左）孩子，而p是g的左（右）孩子。这种情况下，一定是由于在子树v删除节点x而导致g失衡。此时，我们需要对p进行一次逆时针/zag（顺时针/zig）旋转，然后再对g进行一次顺时针/zig（逆时针/zag）旋转。

6. 从局部平衡到整体平衡：在删除操作中，我们只需对失衡节点的祖孙三代进行一次旋转操作，就可以使得局部树恢复平衡。但是，这并不意味着整棵树也恢复平衡。（原因：当原先的失衡节点只有一棵子树时，若该子树进行删除——旋转操作，则会使得整棵树的高度-1。考虑某一祖先节点：可能进行删除——插入操作的子树恰好处于其高度更低的子树上，那么，当原子树进行删除——旋转操作之后，这个节点就会从原先的满足平衡转变为不满足平衡。）为此，我们需要递归地对祖孙三代的祖先进行旋转操作，直到根节点。这样，整棵树的平衡性也得到恢复。

7. 删除——旋转算法的效果和效率：我们有：

   > 1. 在删除AVL节点后，经过上述单旋或双旋调整，最深失衡节点的深度必然减小。
   > 2. 在AVL树中删除一个节点后，至多只需经过O(logn)次旋转操作即可使之恢复平衡。
   > 3. AVL树的节点删除操作可以在$O(\log n)$时间内完成。

## AVL树的`"3+4"`重构

1. 对上述方法进行改进的原因：上述方法虽然必要简明且富有技巧性，但是，方法的分支太多，无法形成有效的桶一的操作。我们需要寻找一个更高效率的统一的算法。所以，我们需要重新审视上述方法中的三个节点和与其相关的四个子树。

2. 具体步骤：

   1. 命名：设$g(x)$为最低的失衡节点，考察祖孙三代：`g,p,v`，并将它们按照中序遍历的次序重命名，得`a<b<c`。而它们总共拥有互不相交的四棵（可能为空的）子树，同样地，将它们按照中序遍历的次序重命名，得`T0<T1<T2<T3`。所以，我们可以得到一个`3+4`的序列：`T0<a<T1<b<T2<c<T3`。这也是BST单调性的具体体现。
   2. 拼接：现在，按照BST单调性的原则，将这些节点和子树直接地拼接起来。此时，`b`为这个局部BST的根节点。

3. 好处：既可以更加概括、深入的了解上述算法的思想，也可以更加简明、高效、鲁棒地设计上述算法。

4. 算法实现：

   ```java
   public static BinTreePosition rotate(BinTreePosition z) {
   	BinTreePosition	y = tallerChild(z);//取y为z更高的孩子
   	BinTreePosition	x = tallerChild(y);//取x为y更高的孩子
   	boolean cType = z.isLChild();//记录：z是否左孩子
   	BinTreePosition	p = z.getParent();//p为z的父亲
   	BinTreePosition	a, b, c;//自左向右，三个节点
   	BinTreePosition	t0, t1, t2, t3;//自左向右，四棵子树
   	/******** 以下分四种情况 ********/
   	if (y.isLChild()) {//若y是左孩子，则
   		c = z;	t3 = z.getRChild();
   		if (x.isLChild()) {//若x是左孩子
   			b = y;	t2 = y.getRChild();
   			a = x;	t1 = x.getRChild();	t0 = (BSTreeNode)x.getLChild();
   		} else {//若x是右孩子
   			a = y;	t0 = y.getLChild();
   			b = x;	t1 = x.getLChild();	t2 = (BSTreeNode)x.getRChild();
   		}
   	} else {//若y是右孩子，则
   		a = z;	t0 = z.getLChild();
   		if (x.isRChild()) {//若x是右孩子
   			b = y;	t1 = y.getLChild();
   			c = x;	t2 = x.getLChild();	t3 = (BSTreeNode)x.getRChild();
   		} else {//若x是左孩子
   			c = y;	t3 = y.getRChild();
   			b = x;	t1 = x.getLChild();	t2 = (BSTreeNode)x.getRChild();
   		}
   	}
   
   	//摘下三个节点
   	z.secede();
   	y.secede();
   	x.secede();
   
   	//摘下四棵子树
   	if (null != t0) t0.secede();
   	if (null != t1) t1.secede();
   	if (null != t2) t2.secede();
   	if (null != t3) t3.secede();
   
   	//重新链接
   	a.attachL(t0);	a.attachR(t1);	b.attachL(a);
   	c.attachL(t2);	c.attachR(t3);	b.attachR(c);
   
   	//子树重新接入原树
   	if (null != p)
   		if (cType)	p.attachL(b);
   		else p.attachR(b);
   
   	return b;//返回新的子树根
   }//rotate
   ```

   ## 对重构算法的综合评估
   
   1. 优点：无论查找、插入或删除，最坏情况下的复杂度均为$O(\log n)$；存储空间的占用为$O(n)$。
   2. 缺点：
      1. 借助高度或平衡因子，为此需改造元素结构，或额外**封装**（伸展树）；
      2. 实测复杂度与理论值尚有差距：
         1. 插入/删除后的旋转，成本不菲；
         2. 删除操作后，最多需旋转$\Omega(\log n)$次（Knuth：平均仅0.21次）；
         3. 若需频繁进行插入/删除操作，未免得不偿失。
      3. 单次动态调整后，全树拓扑结构的变化量可能高达$\Omega(\log n)$。