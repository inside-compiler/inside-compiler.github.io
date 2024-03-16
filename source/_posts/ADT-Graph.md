---
title: ADT中图和迭代器介绍
categories:
  - LLVM
tags:
  - [LLVM, MLIR, ADT]
date: 2024-03-16 18:21:20
---

# 图 
ADT中定义了一个有向图结构，LLVM中一些图会基于它进行实现。
## DirectedGraph

ADT中提供的一个有向图基类，这个类描述图的三个信息：节点、边和图。以及提供了构图的基本API（如connect）以及findNode、findEdgesTo、findIncomingEdgesToNode等图中常用的操作。
但是这个类不包含节点数据信息，所以在使用时需要对Node进行扩展。

# 迭代器
ADT提供了一系列的迭代器，用于遍历图或者寻找子图。
## 图特质

最常见的图遍历方式为深度遍历和宽度遍历，深度遍历一般需要借助于栈实现，宽度遍历一般需要借助队列实现。ADT中提供df_iterator、bf_itertor等辅助迭代器用于遍历图，使用df_iterator、bf_itertor要求被遍历的对象继承GraphTraits，并实现一些基础的API，包括识别根节点，子节点迭代器、访问子节点。

``` typedef NodeRef           - 图中节点类型
 typedef ChildIteratorType - 迭代器定义如何迭代图的子节点
 static NodeRef getEntryNode(const GraphType &) - 图的根节点
 static ChildIteratorType child_begin(NodeRef) - 节点的第一个子节点
 static ChildIteratorType child_end  (NodeRef) - 节点的最后一个子节点
```
 
GraphTraits除了上述的接口外，还有一些使用其他场景的接口，例如适用于获取所有节点、边识别的情况。但是对于df_iterator、bf_itertor、po_iterator、scc_iterator迭代器的只需要实现上面的定义即可。

## df_iterator

ADT中df_iterator包含了两个字段，分别是Visited和VisitStack，它们分别是SmallPtrSet和std::vector类型，其中Visited表示已经访问的节点，VisitStack表示待访问的节点。df_iterator还提供了一个特别的实现，即Visited可以使用外部存储空间，不需要df_iterator分配。所以提供了df_ext_iterator用于标识Visited使用外部空间的情况。除了df_iterator、df_ext_iterator外还提供了idf_iterator、idf_ext_iterator用于逆序访问图的节点。

## bf_iterator

类似于df_iterator，bf_iterator也包括三个字段，分别是Visited、VisitQueue和Level，它们分别是SmallPtrSet、std::queue和unsigned类型，其中Visited表示已经访问的节点，VisitStack表示待访问的节点、Level表示图的访问层次。

## po_iterator

df_iterator是默认的preorder顺序访问图，为了顺利对图进行后续遍历，提供了po_iterator。它的实现和df_itertator非常类似，区别在于构建po_iterator时先构造VisitStack，在遍历时依次从VisitStack的尾部获取元素，从而实现后序遍历。
例外基于po_ext_iterator可以提供更方便的功能，即当重启后序遍历时，外部空间中Visited中存放的节点表示不需要再次访问了。
另外开发者可以继承po_iterator_storage，实现自己的insertEdge、finishPostorder，这两个API可以位PreOrder、PostOrder记录一些信息。
基于po_iterator可以非常方便的实现ReversePostOrderTraversal，它是po_iterator中VisitStack的逆序访问。

## scc_iterator

获取强连通分量。从图中依次获取强连通分量，该算法是基于Tarjan的DFS实现。在图遍历是也只需要子节点迭代器。

<!-- more -->
