---
title: ADT中Map容器介绍
categories:
  - LLVM
tags:
  - [LLVM, MLIR, ADT]
date: 2024-03-16 17:03:21
---

LLVM本身提供了一些Map相关的容器，同时在LLVM还可以直接std中标准map容器，该如何正确使用map？

# LLVM中Map容器

## DenseMap
LLVM提供DenseMap的目的为了加速使用Map的效率。它具有以下特点：
1. 针对key、value组合成一个Pair，然后根据Map的初始长度分配一段连续的内存，其中每个pair作为内存的元素。因此Densemap是Key-Value组合在一起的连续内存。其存储结构如下：
![](ADT-Map/17105802768390.jpg)
 
3. DenseMap需要解决存储时Hash的问题，对于常见类型作为Key时LLVM项目中已经提供了Hash算法，而对于自定义Key时需要实现自己的Hash算法；
4. 当对Key进行Hash后，如果产生冲突，则进行二次散列（实现是基于当前的位置+变化的偏移）。
5. 由于DenseMap底层本质上采用了数组进行存储，所以需要区别不同Key的状态用于标记元素的有效性，例如设置为1《Bit表示空；2《Bit表示元素曾经有用过，但现在已经被删除，设置为Tombe，此时这个槽位并不能被再次占用。
6. DenseMap还考虑了自动内存的扩展，当使用元素超过3/4时会自动扩容；当Tomeb元素超过一定比例（1/8）会对DenseMap重写构造，并将Tombe对应的槽位进行真正回收重用。因此增删元素会导致DenseMap扩容/缩减，进而引起迭代器失效。另外扩容时会删除旧缓存中的元素(析构Key与Value)，因此需要注意Key与Value成员的所有权问题。
7. 需要自己实现迭代器，否则无法遍历容器数据。

本意DenseMap希望提供高性能的Map，但是2014年LLVM大会提供的材料表明DenseMap效果和std中map、unordered_map相比并不完全占优，相反在Insert、lookup性能均裂于unordered_map。

## SmallDenseMap

DenseMap在对分配时最低要求64个元素，所以在数据量较少时存在内存浪费问题，所以提供了一个小数组模拟底层存储（初始默认值为4个元素），当元素超过64时就退化为DensMap。
SmallDenseMap底层使用一个Union组合一个数组和DenseMap。

## ImmutableMap

ADT中提供了不可变Map，它的底层存储时基于平衡二叉树，但是它最大的特点是不袁允许对于Map进行修改，例如插入或者删除元素后都会新生成一个新的ImmutableMap。因此它更适合用于一次构造，多次查询的场景。

## IndexedMap

这是一个特殊的Map，接受两个参数。代码如下：

```template <typename T, typename ToIndexT = identity<unsigned>>
class IndexedMap {
```

第一个参数为Map的类型，第二个参数为索引位置计算的函数。在底层它使用了SmallVector作为存储结构，同时使用函数计算元素在SmallVector的位置。

## IntervalMap

这是LLVM特意为Live Interval设计的数据结构，key是一个区间，value是值，key可以支持前闭后闭、前闭后开两种区间。
IntervalMap采用树形进行存储，分为两类结点：中间结点和叶子结点，叶子结点存放的区间和值，一个叶子结点可能包括多个区间，例如[(start1,end1),value1]，[(start2,end2),value2，[(start3,end3),value3]。主要原因是IntervalMap设计时期望访问能按照CacheLine对其，假设CacheLine为64字节，按照3倍CacheLine设计存储空间（即192字节），而192个字节可能包括多个区间。而中间结点目的是将区间形成一棵树，方便插入、删除和查找。
由于IntervalMap以区间为Key，当插入新的元素后会尝试进行区间合并（当然合并时Value必须相等，并且区间是相邻）。

## StringMap

ADT为了处理String，特别引入StringMap类型，主要是因为String的长度不固定，所以一般需要分配在堆中，所以不太适合直接使用DenseMap进行处理。
StringMap的结构如下所示：
![](ADT-Map/17105803752754.jpg)

可以看到StringMap使用连续内存来处理Key-Value，但是又把Key/Value真实的值放在堆中。
除了底层存储不同以外StringMap在很多地方类似与DenseMap，比如Hash冲突、Key不允许重复，同样地StringMap也会存在扩缩容的问题。

默认长度为16.
所以StringMap更适合的场景是对key、value连续访问，元素个数不多或者hash冲突不严重的场景，且尽可能少的发生扩缩容。

# LLVM其中高级Map
## ValueMap

ValueMap本质上利用DenseMap实现LLVM特殊的需求，例如Value可以被其它的Instruction使用，但是Value发生了变化（例如被删除、RACU），那么Value的使用者需要更新变化情况，使用ValueMap就非常有用。ValueMap存储Value和使用的映射关系，当Value变化时，例如从V1变到V2，则老得映射V1-》Target可以被删除，并添加一个新的映射V2-》Target。
注意：需要通过Value中delete或者replaceAllUseWith对应的API调用才能完成更新。

## MapVector

这个类型是利用DenseMap和std::vector组合的一个新的类型，它和DensMap最大的区别是Value会按照顺序进行存储。这样在一些map的遍历中可以保证顺序。
实际上MapVector首先将Key存储在DenseMap中，如同时将《key、value》的Pair再插入vector中，所以key会被插入两次，因此有空间浪费。但是MapVector相对于vector来说查询效率可能更好，因此查询之前总是先通过DenseMap查询key是否存在，如果key不存在则无需到vector中查询，如果key存在则仍然需要遍历vector。

## SmallMapVector

SmallMapVector和MapVectorhe非常类似，唯一的不同点是SmallMapVector中的Vector使用的是SmallVector而不是std::vector。

## IntEqClasses

这是对整数进行等价类划分，所以它也类似于Map功能。具体方式比较简单，将容器中所有整数进行划分，相同的就是一个等价类。划分完成中每个整数都有一个唯一的等价类标号。
IntEqClasses底层使用smallvector进行实现，如果确定两个元素是等价类，使用join进行连接，连接后返回等价类中第一个元素的索引，当使用compress后等价类标号从0开始依次增加（不再是等价类第一个元素的索引）。

# Std中Map容器

## Map/multiMap
Map采用红黑树进行实现，它时一种平衡二叉树，能保证树的高度大约是对数，在插入、删除、查找时复杂度为哦（nlog（n），由于红黑树保证树的平衡，所以在插入、删除时需要对树进行再平衡，所以性能略差，但是对于查找友好（由于红黑树的特性，可以认为构建的map是以key进行排序进行的）。


Multimap和map最大的区别是key是可以重复的。因此mulltimap提供的一些API和Map略有不同，multimap可能返回iterator，用于表示可能存在多个Key-Value，而Map通常返回pair即可。

## unordered_map/undered_multimap

undered_map采用的是链表方式实现，和DenseMap类似。但是又有所不同，undered-map中的key是连续的，但是value采用了链表的方式进行存储，所以在解决冲突时较为简单。但是通常插入时需要使用malloc进行分配。


<!-- more -->
