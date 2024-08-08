# 映射(Map)、词典(Dictionary)和散列(Hash)

## 键值对

1. 键值对是计算系统和应用程序中的一种基本数据表示方式。设计人员通常希望采用一种开放式数据结构，以便在不修改现有代码或数据的情况下进行未来扩展。在这种情况下，数据模型的全部或部分内容可以用 `<属性名、值>`形式的 2 元组集合来表示，每个元素都是一个属性值对。根据特定应用和程序员选择的实现方式，属性名可能是唯一的，也可能不是唯一的。

   > A name–value pair, also called an attribute–value pair, key–value pair, or field–value pair, is a fundamental data representation in computing systems and applications. Designers often desire an open-ended data structure that allows for future extension without modifying existing code or data. In such situations, all or part of the data model may be expressed as a collection of 2-tuples in the form <attribute name, value> with each element being an attribute–value pair. Depending on the particular application and the implementation chosen by programmers, attribute names may or may not be unique.
   >
   > ——From [Name–value pair - Wikipedia](https://en.wikipedia.org/wiki/Name–value_pair)

## 映射和词典的定义

1. **词条(Entry)**：一种组合关系，由**关键码(Key)**和**数据项(Value)**合成。
2. **映射(Map)**的定义：
   1. 基本定义：假设两个非空集合$X, Y$，存在一个法则$f$，使得对$X$中的每个元素$x$，按法则$f$，在$Y$中有唯一确定的元素$y$与之对应，那么称$f$为从$X$到$Y$的映射。
   2. 在计算机科学中，映射是一种存放一组条目的容器，每个条目(Entry)都包含一个关键码(key)和一个值(value)。每个键都是唯一的，用于标识与之相关联的值。映射提供了一种快速查找值的方法，只需知道其关联的键。需要特别指出，在映射中，各条目的关键码**不允许重复冗余**。
   3. 映射中的元素由其关键码唯一标识，而且映射的作用就是通过关键码直接找到对应的元素，故不妨就把关键码理解为指向对应元素的”地址引用“。

3. **词典(Dictionary)**：词典是一种数据结构，它也存储键值对。在词典中，键也是唯一的，每个键都对应一个值。词典提供了一种方法，可以通过键快速访问、插入或删除对应的值。在词典中，各条目的关键码**可以重复冗余**。
4. 对于**映射和词典**这类结构，在作为基本数据单位的词条内部，关键码与数值的地位等同，二者不必加以区分。此类结构所支持的这种新的数据访问方式，即所谓的循数值访问。
5. 特别地，在许多编程语言中，映射和词典实际上是同一种数据结构。例如，在Python中，被称为字典（Dictionary），而在Java中，则被称为映射（Map）。

## Map的ADT及具体实现

1. Map类的接口：	

   ```java
   public interface Map { 
       //查询映射结构当前的规模 
       public int getSize(); 
       //判断映射结构是否为空 
       public boolean isEmpty(); 
       //若映射中存在以key为关键码的条目，则返回该条目的数据对象；否则，返回null 
       public Object get(Object key); 
       //若映射中不存在以key为关键码的条目，则插入条目(key, value)并返回null 
       //否则，将已有条目的数据对象替换为value，并返回原先的数据对象 
       public Object put(Object key, Object value); 
       //若映射中存在以key为关键码的条目，则删除之并返回其数据对象；否则，返回null 
       public Object remove(Object key); 
       //返回映射中所有条目的一个迭代器 
       public Iterator entries(); 
   } 
   ```

   > 这里，没有采取意外错的形式来处理退化情况。原因是在实际中，出现退化情况的概率很高，若采用报意外错的形式，效率十分低下。不过，直接返回null的做法也存在不足，其中最大的问题在于歧义性。比如，按照这一约定，我们将无法以key = null为条件进行查找。

2. 判等器(Equality Tester)：

   1. 映射结构必须能够比较任意一对关键码是否相等，每个映射结构在被创建的时候，都需要指定某一具体标准，以便进行关键码的比较。因此，为了实现映射结构，首先必须实现这样的一个判等器。

      ```java
      public interface EqualityTester { 
      	public boolean isEqualTo(Object a, Object b);//若a与b相等，则返回true；否则，返回false 
      }
      public class EqualityTesterDefault implements EqualityTester { 
          public EqualityTesterDefault() {} 
          public boolean isEqualTo(Object a, Object b) 
          { return (a.equals(b)); }//使用Java提供的判等器 
      }
      ```

   2. Java提供的`equals()`是一个较为通用的判等器，我们也可以新建一个类，利用Java提供的`euqals()`方法和自己实现的某些额外方法，实现一个更加通用的判等器。利用这种模式，程序员完全可以编写出独立的通用判等器，而无需触及对象内部的结构。


## java.util包中的映射类

1. Java在`java.util`包中已定义了一个名为`java.util.Map`的映射接口，而且也约定禁止关键码的重复。与第6.1.1节定义的映射接口相比，`java.util.Map`接口并未直接提供迭代器方法，而是通过两个名为`keys()`和`values()`的方法，间接地提供关于关键码或数据对象的迭代器。
2. [Map (Java Platform SE 8 ) (oracle.com)](https://docs.oracle.com/javase/8/docs/api/java/util/Map.html)

## C++ STL Unordered-map

1. std::unordered_map 是一种关联容器，含有带唯一键的键-值对。搜索、插入和元素移除拥有平均常数时间复杂度。

   元素在内部不以任何特定顺序排序，而是组织进桶中。元素放进哪个桶完全依赖于对应键的散列。具有相同散列码的键出现于同一个桶。这允许对单独元素的快速访问，因为一旦计算其散列，它即代表元素所放进的确切的桶。

   如果映射的键相等性谓词在传递两个键时返回 true，则它们被认为 等价。如果两个键等价，则散列函数必须对它们返回相同的值。

   std::unordered_map 满足容器 (Container) 、知分配器容器 (AllocatorAwareContainer) 和无序关联容器 (UnorderedAssociativeContainer) 的要求。

2. [std::unordered_map - cppreference.com](https://zh.cppreference.com/w/cpp/container/unordered_map)

## 哈希表

如果将条目的关键码视作其在映射结构中的存放位置，则可以散列表（Hash table）的形式来实现映射结构。哈希表又称散列表，一种以「key-value」形式存储数据的数据结构。所谓以「key-value」形式存储数据，是指任意的键值 key 都唯一对应到内存中的某个位置。只需要输入查找的键值，就可以快速地找到其对应的 value。可以把哈希表理解为一种高级的数组，这种数组的下标可以是很大的整数，浮点数，字符串甚至结构体。

## 从IBM公司电话说起

1. 在[联系 IBM 支持 - IBM 文档](https://www.ibm.com/docs/zh/i/7.3?topic=overview-contacting-support)中，IBM公司的电话号码是`1-800-IBM-CALL`和`1-800-IBM-4YOU`。为什么电话号码中可以出现字母？观察9键键盘，在某些数字键下面，会有英文字母出现。所以，IBM公司的电话号码可以由`1-800-426-2255`改写成`1-800-IBM-CALL`。IBM公司的这种方式很是巧妙，借助个性化的电话号码，让用户能够加深对其的印象。这中间蕴含的思维方式，即是**散列**。
2. IBM的巧妙体现在两个方面：
   1. 让你能够记住这一家公司的电话号码；
   2. 拨号时，仍然使用的是数字系统，只需要对键盘进行字母+数字改造即可实现上述内容。

## 各种访问方式的比较

1. 循秩访问(Call by Rank)：Vector；
2. 循位置访问(Call by Position)：List；
3. 循关键码访问(Call by Key)：BST（二叉搜索树）；
4. 循数值访问(Call by Value)：Hashing（哈希、杂凑、**散列**）。

## 联合数组

联合数组(Associative Array)：可以根据数据元素的取值，直接访问。其下标不再是整数，甚至没有了大小次序，这样，更为直观、便捷。支持的语言有：`Java, Python, Perl, Ruby, PHP`等。

部分语言的例子：

```java
//1.java: HashMap+HashTable
import java.util.*;
public class Hash {
    public static void main(String[] args) {
        HashMap HM = new HashMap(); //Map
        HM.put("东岳", "泰山"); HM.put("西岳", "华山"); HM.put("南岳", "衡山");
        HM.put("北岳", "恒山"); HM.put("中岳", "嵩山"); System.out.println(HM);
        Hashtable HT = new Hashtable(); //Dictionary
        HT.put("东岳", "泰山"); HT.put("西岳", "华山"); HT.put("南岳", "衡山");
        HT.put("北岳", "恒山"); HT.put("中岳", "嵩山"); System.out.println(HT);
	}
}
```

```python
# 2.python: Dictionary Class
beauty = dict( { "沉鱼":"西施", "落雁":"昭君", "闭月":"貂蝉", "羞花":"玉环" } )
print( beauty )
beauty["红颜"] = "圆圆"
print( beauty )
for (alias, name) in beauty.items() :
	print( alias, ":", name )
for alias, name in sorted( beauty.items() ) :
	print( alias, ":", name )
for alias in sorted( beauty.keys(), reverse = True ) :
	print( alias, ":", beauty[alias] )
```

## 实例：电话簿

需求： 为一所学校制作电话簿。号码$\leftrightarrows$教员、学生、员工、办公室

1. 使用蛮力法（使用数组实现，按电话号码索引）（R=Real，实际拥有的个数；P=Possible，所有可能的个数。）
   1. 时间复杂度：$O(1)$。
   2. 空间复杂度：$O(P+R)$。
   3. 效率：$\eta=\dfrac{R}{P}$，过低（一般地，不到千分之一）。

2. 使用散列（简单地改造数组为桶数组(Bucket Array)，即散列表/哈希表(Hash Table)）
   1. 桶(Bucket)：直接或间接地指向一个词条。
   2. 哈希表：容量为$M$，满足$N<M<<R$，空间复杂度$O(N+M)=O(N)$。（压缩数组的长度，尽量保证$M=\Theta(N)$）
   3. 定址（散列(Hashing)）：根据词条的key（未必可比较）而直接确定散列表的入口。（无论表有多长）
   4. 确定散列函数`hash()`：`hash(key) -> &entry`，将词条的关键码转化为某一个桶单元，并根据这一个桶单元而找到目标词条。
   5. 时间复杂度：若合理地进行设计，则为$O(1)$。
3. 散列的具体实现：
   1. 确定哈希表：这里哈希表长为$M=\Theta(N)$。（C为常数）
   2. 确定散列函数：可以有很多种构造方法，这里使用取余法`hash(key) = key % M`。
   3. 空间效率：一般可以使用装填因子$\lambda=\dfrac{N}{M}$来衡量。

## 散列表的冲突

1. 散列冲突(Hash Collision)：关键码不同的词条被映射到同一散列地址的情况，即`hash(key1)=hash(key2)`，而`key1!=key2`。原因是，某些散列函数只是将一个相对较大的集合映射到一个远远相对较小的集合，而按照抽屉原理/鸽巢原理，冲突无法彻底避免。

2. 总而言之，随机越强、规律性越弱的散列函数越好。当然，完全符合上述条件的散列函数并不存在，我们只能通过先验地消除可能导致关键码分布不均匀的因素，最大限度地模拟理想的随机函数，尽最大可能降低发生冲突的概率。

3. 举例：将人所对应的词条按照生日（月/日）做散列存储。散列表长$M = 365$，装填因子$\lambda=\dfrac{N}{M}$（N为在场人数） 。

   > 至少有两位同学生日相同的可能性$P_{365}(n) = ?$
   >
   > 
   > $$
   > P_{365}(1) = 0, P_{365}(2) = 1/365,\cdots, P_{365}(22) = 47.6\%, \\P_{365}(23) = 50.7\%,\cdots P_{365}(100)=99.999969\%
   > $$


4. 在装填因子确定之后，散列策略的选取将至关重要，散列函数的设计也很有讲究。

5. 两项基本任务：

   1. 精心设计散列表及散列函数，尽可能降低冲突的概率。
   2. 制定可行的预案，以便在发生冲突时，能够尽快予以排解。

# 散列函数

## 什么是哈希函数

1. 哈希函数/散列函数(Hash Function)：要让键值对应到内存中的位置，就要为键值计算索引，也就是计算这个数据应该放到哪里。这个根据键值计算索引的函数就叫做哈希函数，也称散列函数。

## 散列函数的评价标准和设计原则

1. 确定(Determinism)：同一关键码总是被映射至同一地址；
2. 快速(Efficiency)：期望的时间复杂度为$O(1)$；
3. 满射(Surjection)：尽可能充分地利用整个散列空间；
4. 均匀(Uniformly)：关键码映射到散列表各位置的概率尽量接近，以有效避免聚集(Clustering)现象。

## 完美哈希函数(PHF)

1. 完美哈希函数(Perfect Hash Function, PHF)：函数Hash()将$N$个Key映射到$M$个整数上($M\ge N$)，且对于任意的$Key_1$和$Key_2$，$\text{Hash}(Key_1)\neq \text{Hash}(Key_2)$，即是没有冲突的哈希函数，则称之为完美哈希函数。
2. 最小完美哈希函数(Minimal Perfect Hash Function, MPHF)：对于所有的完美哈希函数，若存在一个函数$\text{MinHash}()$，使得$M=N$，则该函数称之为最小完美哈希函数。
3. 在现实情况中，很难构造完美的散列算法。因为通常几乎不可能提前知道要散列的完整数据集。例如，在我们马上将探讨的一个程序中，散列表用于统计文本中的重复单词。由于我们事先不知道哪些单词出现在文本中，就不能编写一个完美的算法。数据仓库的查询索引，还有一些不需要更新且对性能有要求的场景，这个算法是适用的。
4. 通常情况下，PHF或MPHF是针对静态集合的。也就是，在使用PHF或MPHF时，所有的 KEY 值是事先已知并且固定的。不过，也有针对动态集合的一个算法。
5. 使用PHF和MPHF的一般流程
   1. （准备阶段）将已知的所有的 KEY 值传给PHF或MPHF生成算法，生成PHF或MPHF以及相应的数据；（这也是完美hash函数的一大缺点：必须事前必须知道原数据集，并且需要花一定的CPU来生成这个函数。）
   2. （使用阶段）调用已有的PHF或MPHF以及相应的数据快速计算哈希值并进行相应的操作。（其实在实际使用中我们只关心步骤2，但步骤1的生成算法却是PHF或MPHF的关键。）
6. 更多详见：[完美哈希函数(Perfect Hash Function) |博客园 ](https://www.cnblogs.com/eve-walle/archive/2012/09/17/2688914.html)

## 散列函数的某些构造方法
### 整数关键码的散列函数构造
   1. 直接构造法：适用于键值为整数且范围比较小的时候，直接让键值作为数组的下标。

   2. 除余法：$\text{Hash}(key)=key\%M$。适用于键值为整数且范围比较大的时候。一般把键值模一个较大的**质数**作为索引，也就是取$f(x)=x\mod M$作为哈希函数。
      1. 关于M的选择：尽量不要选$M=2^k$类似的数。因为，其等效于只是截取了key的最后k位(Bit)，而前面的$n-k$位没有影响。此时，$key\mod M=key\  \&\ (M-1)$。则若某些数的最后k位相同，就会发生冲突（以同余类构成的集合中的元素全部会冲突）。据说，**M为素数时，数据对散列表的覆盖最充分，分布最均匀**；但实际上，对于理想随机的序列，表长是否素数，无关紧要！而理想随机的序列，出现的概率十分低，在这样的情况下，我们还是应该要选择一个较大的素数M。
      2. 以蝉为师：思考：为什么蝉的生命周期为素数？考虑以特定的步长$S$来遍历数列。令$G=\gcd(S,M)$，并假设有一段连续的地址空间属于哈希表，其长度为$M$。每一次以$S$为步长向前遍历哈希表。注意，最后的位置再次步进可以至于最先的位置。如果能够做到访问所有的位置，我们说其具有均匀性。当且仅当$G=1$时，即$S,M$互素时，其有均匀性。回到之前的问题，其天敌的生命周期为$S$，当蝉的生命周期$M$为素数时，在自然界中，$G=gcd(S,M)=1$的出现频率大大增加，蝉更容易生存。

   3. MAD法：即Multiply-Add-Divide，除余法的改进

      1. 除余法的缺陷：
         1. 不动点：无论表长M取值如何，总有$\text{Hash}(0)\equiv0$；
         2. 相关性（零阶均匀）：$[0,R)$的关键码尽管系平均分配至$M$个桶，但相邻关键码的散列地址也必相邻。

      2. 至少需要达到一阶均匀，以获得更好的随机性，即临近的关键码，其哈希地址不再相邻。高低阶的均匀性需要根据使用场合来确定。
      3. 使用MAD法，构造哈希函数：$\text{hash}(key)=(a\times key+b)\% M$，其中$M$为素数，$a>1,b>0,M\nmid a$。

   4. 数字分析法：抽取key中的某几位，构成地址。

      1. 最简单的例子：取十进制表示的奇数位。

      2. 平方取中(Mid-square)：取key平方的中间若干位，构成地址。

         > 为什么要**取中**？
         >
         > 观察乘法的竖式计算，在答案中，居中的数位由多个数字相加而成，而在两旁的数位由较少的数字加和而成，所以，取中会使得更多的关键码中的数码参与到对地址的贡献中去，即会大大地提高随机性。

   5. 折叠法(folding)：将key分割成等宽的若干段（有自左向右和往复折返等方式），取其总和作为地址。

   6. 位异或法XOR：将key分割成等宽的二进制段有自左向右和往复折返等方式），经异或运算得到地址。

   7. 还有其他的方式，总之，越是**随机**，越是没有规律，越好。

### （伪）随机数发生器

   1. C语言的`rand()`函数可以实现伪随机数。`rand()`函数的代码和下面的类似：

      ```c
      static unsigned long next = 1;
      
      /* RAND_MAX assumed to be 32767 */
      int myrand(void) {
          next = next * 1103515245 + 12345;
          return((unsigned)(next/65536) % 32768);
      }
      
      void mysrand(unsigned seed) {
          next = seed;
      }
      ```

      每一个随机数都是由前一个随机数递推而得，因而更应该称之为“伪随机数”。

   2. （伪）随机数算法和散列函数的关系：散列函数的设计原则和评价标准，也是伪随机数算法的设计原则和评价标准。
      1. 伪随机数算法：$rand( x + 1 )  =  [ a \times rand( x ) ] \% M$，其中$M$为素数，$a>1$。
      2. $hash(key) = rand(key) = [rand(0) \times a^{key}] \% M$
      3. （伪）随机数发生器的实现，因具体平台、不同历史版本而异，创建的散列表可移植性差——故需慎用此法。

### 字符串关键码的散列函数构造
1. 多项式法：由于不支持以字符串作为数组下标，并且将字符串转化成数字存储也可以避免多次进行字符串比较。所以，一般不直接把字符串作为键值，而是先算出字符串的哈希值，再把其哈希值作为键值插入到哈希表里。关于字符串的哈希值，我们一般采用进制的思想，将字符串想象成一个 $a$ 进制的数。那么，对于每一个长度为 $n$ 的字符串 $s$，就有：

$$
x = s_0 \cdot a^0 + s_1 \cdot a^1 + s_2 \cdot a^2 + \ldots + s_n \cdot a^n
$$


   我们可以将得到的 $x$ 对 $2^{64}$（即 `unsigned long long` 的最大值）取模。这样 `unsigned long long` 的自然溢出就等价于取模操作了。可以使操作更加方便。这种算法的时间复杂度为 $O(n)$，其中 $n$ 为字符串的长度。

2. 多项式法的改进：这种方法对英文字符串非常有效。

   ```cpp
   static size_t hashCode( char s[] ) {
       size_t n = strlen(s); size_t h = 0;
       for ( size_t i = 0; i < n; i++ ) {
           h = (h << 5) | (h >> 27);//将h所代表的32位二进制整数的前5位和后27位交换顺序
           h += s[i];//累计每一位字符的贡献
       } //乘以32，加上扰动，累计贡献
       return h;
   }
   ```

3. 最简单的方法：直接相加法。将每一个字母和一个数字对应起来，用所有数码之和代表哈希。哈希函数为

$$
Hash(S)=\sum_{c\in S}\text{code(upper(c))}
$$

不过，这种方式最能造成冲突。如：`hash(I am Lord Voldemort)=hash(Tom Marvolo Riddle)=hash(He's Harry Potter)=196`。所以，必须要解决散列冲突。

# 散列冲突

## 为什么要对哈希冲突制定对策

1. 散列表的基本构思，可以概括为：开辟物理地址连续的桶数组`ht[]`，借助散列函数`hash()`，将词条关键码`key`映射为桶地址`hash(key)`，从而快速地确定待操作词条的物理位置。  

2. 然而遗憾的是，无论散列函数设计得如何巧妙，也不可能保证不同的关键码之间互不冲突。 在实际应用中，不发生任何冲突的概率远远低于我们的想象。  

   1. 考查如下问题：某课堂的所有学生中，是否有某两位生日（birthday，而非date of birth） 相同？这种情况也称作生日巧合。那么，发生生日巧合事件的概率是多少？  若将全年各天视作365个桶，并将学生视作词条，则可按生日将他们组织为散列表。如此， 上述问题便可转而表述为：若长度为365的散列表中存有n个词条，则至少发生一次冲突的概率有多大？

     > 至少有两位同学生日相同的可能性$P_{365}(n) = ?$
     >
     > 
     > $$
     > \begin{aligned}
     > P_{365}(1) = 0, P_{365}(2) = 1/365,&\cdots, P_{365}(22) = 47.6\%, \\P_{365}(23) = 50.7\%,&\cdots P_{365}(100)=99.999969\%
     > \end{aligned}
     > $$
   
   
   
   2.   不难理解，对于更长的散列表，只需更低的装填因子，即有50%的概率会发生一次冲突。鉴 于实际问题中散列表的长度M往往远大于365，故“不会发生冲突”只是一厢情愿的幻想。因此， 我们必须事先制定一整套有效的对策，以处理和排解时常发生的冲突。

## 解决方案

对于哈希冲突，我们可以使用开散列法、闭散列法。其中，开散列法包含多槽位法、独立链法、公共溢出区法；闭散列法包括线性试探法、查找链法，懒惰删除法、重散列法、平方试探法、双向平方试探法、双散列法等。

## 开散列法（封闭定址）

在这种方法中，每个数组位置都存储一个链表。当发生哈希冲突时，新的键值对会被添加到对应位置的链表中。实例：Java的HashMap就是采用封闭定址的策略。

封闭定址法可以处理任意数量的冲突，但是如果链表过长，查找效率会降低。

### 多槽位法

多槽位法(Multiple Slots)：将每个桶单元细分为若干个槽位，解决桶内冲突。只要槽位的数目不太多，仍然可以保证$O(1)$的时间效率。

缺点：slots过多，空间浪费；slots过少，仍然发生冲突。

### 独立链法（拉链法）

拉链法(Separate Chaining)：是在每个存放数据的地方开一个链表（每个桶存放一个指针），如果有多个键值索引到同一个地方，只用把他们都放到那个位置的链表里就行了。查询的时候需要把对应位置的链表整个扫一遍，对其中的每个数据比较其键值与查询的键值是否一致。如果索引的范围是 $1\ldots M$，哈希表的大小为 $N$，那么一次插入/查询需要进行期望 $O(\frac{N}{M})$ 次比较。

优点：无需为每个桶预备多个槽位；任意多次的冲突都可解决；删除操作实现简单、统一 。

缺点：指针本身占用空间；节点的动态分配和回收需耗时间；**空间未必连续分布，系统缓存很难生效**。

实现：

```cpp
//cpp
const int SIZE = 1000000;
const int M = 999997;

struct HashTable {
  struct Node {
    int next, value, key;
  } data[SIZE];

  int head[M], size;

  int f(int key) { return (key % M + M) % M; }//哈希函数

  int get(int key) {//首先通过哈希函数找到对应的链表头，然后进行遍历查找
    for (int p = head[f(key)]; p; p = data[p].next)
      if (data[p].key == key) return data[p].value;
    return -1;
  }

  int modify(int key, int value) {//修改给定键的值：找到链表头后遍历链表查找键
    for (int p = head[f(key)]; p; p = data[p].next)
      if (data[p].key == key) return data[p].value = value;
  }

  int add(int key, int value) {//添加新的键值对
    if (get(key) != -1) return -1;
    data[++size] = (Node){head[f(key)], value, key};
    head[f(key)] = size;
    return value;
  }
};
```

```python
# python
M = 999997
SIZE = 1000000

class Node:
    def __init__(self, next=None, value=None, key=None):
        self.next = next
        self.value = value
        self.key = key


data = [Node() for _ in range(SIZE)]
head = [0] * M
size = 0

def f(key):
    return key % M

def get(key):
    p = head[f(key)]
    while p:
        if data[p].key == key:
            return data[p].value
        p = data[p].next
    return -1

def modify(key, value):
    p = head[f(key)]
    while p:
        if data[p].key == key:
            data[p].value = value
            return data[p].value
        p = data[p].next

def add(key, value):
    if get(key) != -1:
        return -1
    size = size + 1
    data[size] = Node(head[f(key)], value, key)
    head[f(key)] = size
    return value
```

### 补充内容

这里再提供一个封装过的模板，可以像 map 一样用，并且较短。

```cpp
struct hash_map {  // 哈希表模板

  struct data {
    long long u;
    int v, nex;
  };  // 前向星结构

  data e[SZ << 1];  // SZ 是 const int 表示大小
  int h[SZ], cnt;

  int hash(long long u) { return (u % SZ + SZ) % SZ; }

  // 这里使用 (u % SZ + SZ) % SZ 而非 u % SZ 的原因是
  // C++ 中的 % 运算无法将负数转为正数

  int& operator[](long long u) {
    int hu = hash(u);  // 获取头指针
    for (int i = h[hu]; i; i = e[i].nex)
      if (e[i].u == u) return e[i].v;
    return e[++cnt] = (data){u, -1, h[hu]}, h[hu] = cnt, e[cnt].v;
  }

  hash_map() {
    cnt = 0;
    memset(h, 0, sizeof(h));
  }
};
```

在这里，hash 函数是针对键值的类型设计的，并且返回一个链表头指针用于查询。在这个模板中我们写了一个键值对类型为 `(long long, int)` 的 hash 表，并且在查询不存在的键值时返回 -1。函数 `hash_map()` 用于在定义时初始化。

## 闭散列法（开放定址）

在这种方法中，所有的键值对都直接存储在哈希表数组中（连续的空间）。当发生哈希冲突时，会寻找其他位置来存储新的键值对。只要有必要，任何散列桶都可以接纳任何词条。

开放定址法可以避免链表的额外内存开销，但是如果数组填充度过高，会导致冲突频繁，降低效率。

### 查找链/试探链

在闭散列法下面，我们为每一个桶都事先约定若干个备用桶（优先级逐次下降），它们构成了一个查找链。

查找算法：沿试探链，逐个转向下一桶单元，直到**命中成功**，或者**抵达一个空桶**（存在则必能找到？）而**失败**。

如何约定试探链？

### 线性试探法

1. 线性试探(Linear Probing)：一旦冲突，则试探后一紧邻的桶；直到命中（成功），或抵达空桶（失败）。

2. 优点：在散列表内部解决冲突；无需附加的指针、链表或溢出区等；整体结构保持简洁 。而且，只要还有空桶，迟早会找到。试探链连续，数据局部性良好。

3. 缺点：新增非同义词之间的冲突；数据堆积（clustering）现象严重。比如，一个孤立的数和一组连续的数插入到桶中。如果第一个数是孤立的，那么后面的一群连续的数都会发生冲突。不过，可以通过装填因子而有限控制冲突与堆积。

4. 具体实现：

   ```cpp
   const int N = 360007;  // N 是最大可以存储的元素数量
   
   class Hash {
    private:
     int keys[N];
     int values[N];
   
    public:
     Hash() { memset(values, 0, sizeof(values)); }
   
     int& operator[](int n) {
       // 返回一个指向对应 Hash[Key] 的引用
       // 修改成不为 0 的值 0 时候视为空
       int idx = (n % N + N) % N, cnt = 1;
       while (keys[idx] != n && values[idx] != 0) {
         idx = (idx + cnt * cnt) % N;
         cnt += 1;
       }
       keys[idx] = n;
       return values[idx];
     }
   };
   ```


### 懒惰删除法

1. 按照开放定址的策略：先后插入、相互冲突的一组词条，都将存放于同一查找链中。

2. 插入与删除：对于插入，新词条若尚不存在，则存入试探终止处的空桶。但试探链可能因而彼此串接、重叠。对于删除，也不能简单地清除命中的桶，否则经过它的试探链都将因此断裂，导致后续词条丢失——明明存在，却访问不到。

3. 懒惰删除(Lazy Removal)：仅做标记，不对试探链做更多调整。此后，带标记的桶，角色因具体的操作而异：

   1. 查找词条时，被视作“**必不**匹配的**非空**桶”，试探链在此得以延续。
   2. 插入词条时，被视作“**必然**匹配的**空闲**桶”，可以用来存放新词条。

4. 实现：

   ```typescript
   class LazyHashTable<K, V> {
       private table: Map<K, V>;
       private deleted: Set<K>;
   
       constructor() {
           this.table = new Map<K, V>();
           this.deleted = new Set<K>();
       }
   
       // 添加元素
       add(key: K, value: V): void {
           //如果键在已删除集合中，需要从已删除集合中移除
           if (this.deleted.has(key)) {
               this.deleted.delete(key);
           }
           this.table.set(key, value);
       }
   
       // 获取元素
       get(key: K): V | undefined {//联合类型，表示返回类型可以是V或undefined
           //如果键在已删除集合中，返回 undefined
           if (this.deleted.has(key)) {
               return undefined;
           }
           return this.table.get(key);
       }
   
       // 修改元素
       modify(key: K, value: V): void {
           //如果键在已删除集合中，不做任何操作
           if (this.deleted.has(key)) {
               return;
           }
           if (this.table.has(key)) {
               this.table.set(key, value);
           }
       }
   
       // 删除元素（懒惰删除）
       delete(key: K): void {
           //如果键在哈希表中，添加到已删除集合
           if (this.table.has(key)) {
               this.deleted.add(key);
           }
       }
   
       //重哈希（移除所有已删除的元素）
       rehash(): void {
           for (let key of this.deleted) {
               this.table.delete(key);
           }
           // 清空已删除集合
           this.deleted.clear();
       }
   }
   ```


### 平方试探法

1. 相较于开散列法，闭散列法拥有很好的性能。然而，对于闭散列的线性试探。试探位置的间距太近。所以，要适当地拉开试探的间距。

2. 所谓平方试探法，就是以平方数为距离，确定下一试探桶单元，即`[hash(key)+(i++)^2] % M`。

3. 优点：数据聚集现象有所缓解；试探链上，各桶间距线性递增；一旦冲突，可“聪明”地跳离是非之地。

4. 弊端：(1) 若涉及外存：I/O次数会增加。

   > 估算：
   >
   > 在通常情况下，缓存的规模在若干个KB左右。若这里为1KB，每一次只记录相应的引用（4字节），故每一个缓存页面都足以容纳至少256个桶单元。所以，每一次做I/O兑换，都要连续的发生16$(256=16^2)$次冲突。

   (2) 且试探问题的加大，可能会使很多的桶单元仍为空，而无法被试探出来。如，$M=12$，有$i^2$ mod 12={0,1,4,9}。其中$i\in\mathbb{N}$。

   > 由二次剩余的相关知识，我们有：
   >
   > 若$M$为合数，则$n^2\mod M$可能的取值必然少于$\left\lceil\dfrac{M}{2}\right\rceil$种。若$M$为素数，则$n^2\mod M$可能的取值恰好等于$\left\lceil\dfrac{M}{2}\right\rceil$种。


### 双向平方试探法

1. 引入：考查单向平方试探法，设散列表长度为素数M > 2。试证明：

   1. 任意关键码所对应的查找链中，前$\left\lceil\dfrac{M}{2}\right\rceil$个桶必然互异。

      > 证明：反证。假设存在$0\le a<b<\left\lceil\dfrac{M}{2}\right\rceil$，使得查找链上的第a个位置与第b个位置冲突，于是$a^2$和$b^2$必然同属于关于M的某一同余类，亦即： 
      >
      > 
      > $$
      > a^2\equiv b^2 (\text{mod }M)
      > $$
      > 
      >
      > 于是便有：
      >
      > 
      > $$
      > a^2-b^2=(a+b)(a-b)\equiv 0 (\text{mod }M)
      > $$
      > 
      >
      > 然而，无论是$(a + b)$还是$(a - b)$，绝对值都严格小于$M$，故均不可能被$M$整除——这与M 是素数的条件矛盾。

   2. 在装填因子尚未增至 50%之前，插入操作必然成功（而不致因无法抵达空桶而失败）。

      > 由上可知，查找链的前$\left\lceil \frac{M}{2} \right\rceil$项关于M，必然属于不同的同余类，也因此互不冲突。在装填因子尚不足50%时，这$\left\lceil \frac{M}{2} \right\rceil$项中至少有一个是空余的，因此不可能发生无法抵达空桶的情况。

   3. 在装填因子超过 50%之后，若适当调整各个桶的位置，则下一插入操作必然因无法抵达空桶而失败。

      > （只需证明：{0^2,1^2,2^2,...}关于M的同余项恰好只有$\left\lceil \frac{M}{2} \right\rceil$个）
      >
      > 证明：
      >
      > 任取$\left\lceil\dfrac{M}{2}\right\rceil\le c<M-1$，并考察查找链上的第c项。可以证明 ，总是存在$0\le d< \left\lceil \dfrac{M}{2} \right\rceil$ ，且查找链上的第$d$项和第$c$项冲突。实际上，我们只要令$d=M-c\ne c$，则有
      >
      > 
      > $$
      > c^2-d^2=(c+d)(c-d)=M(c-d)\equiv 0(\text{mod }M)
      > $$
      > 
      >
      > 于是$c^2$和$d^2$关于M同属一个同余类，作为散列地址相互冲突。

   4. 当M为合数时，即使在装填因子未达到50%之前，平方试探也有可能因无法抵达空桶而失败。

      > 在此时，对于$0\le a<b<\left\lceil\dfrac{M}{2}\right\rceil$，即便$a\pm b\equiv0(\text{mod }M)$​均不成立，也依然可能有
      >
      > 
      > $$
      > a^2-b^2=(a+b)(a-b)\equiv 0 (\text{mod }M).
      > $$

   

2. 由单向变双向：双向平方试探。其策略是，自冲突位置起，交替地沿着两个方向试探，均按平方来确定距离。

3. 考察子试探链：正向和反向的子试探链，各自包含$\left\lceil \frac{M}{2} \right\rceil$个互异的桶。对于形如$M=4k+3$的素数，可以保证两个子试探链彼此独立，即试探链的前$M$项均互异。而形如$m=4k+1$的素数，两个子试探链彼此数据相同，只是排列的顺序不同。所以，**表长必须取4k+3型的素数**。（由费马平方和定理可证得）

   > 如：
   >
   > n=7: 5, 3, 6,0,1,4,2; （从-25到+25）
   >
   > n=11: 8, 6, 2, 7, 10, 0, 1, 4, 9, 5, 3（从-36到+36）

### 公共溢出区法

1. 单独开辟一块连续空间，发生冲突的词条，顺序存入此区域。
2. 优点：结构简单，算法易于实现。
3. 缺点：但是，不冲突则已，一旦发生冲突，则最坏情况下，处理冲突词条所需的时间将正比于溢出区的规模。

### 重散列法

1. 随着装填因子增大，冲突概率、排解难度都将激增——此时，不如“集体搬迁”至一个更大的散列表。
2. 即：若装填因子高于50%，则重散列。

### 双散列法

1. 预先约定第二散列函数，当冲突时，则由其确定偏移量并确定下一试探位置。（更一般地，偏移增量同时还与key相关。）
