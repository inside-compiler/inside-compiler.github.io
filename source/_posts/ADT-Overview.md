---
title: ADT-概述
categories:
  - LLVM 
tags:
  - LLVM, MLIR, ADT
date: 2024-02-02 21:54:44
---

ADT（Abstract Data Type）是LLVM定义的一套高级数据类型，是LLVM项目的基础组件。ADT定义了“基础类型”、“容器”、“算法”、“迭代器”，具体来说：
- 基础类型：定义了APInt（任意精度的整数）、APSInt（任意精度符号整数）、APFloat（任意精度的浮点数）、APFixedPoint（任意精度的定长数）类型；
- 容器：定义了list、map、set、string、array、bitvector等容器；
- 算法：定义了enumerate、zip、seq、enum_seq等高级算法；
- 迭代器：定义了depth_first、post_order、scc_iterator、graphtraits等图处理的迭代器。

我们知道STD中也定义类似的数据结构，为什么LLVM又重复提供类似的功能呢？笔者理解主要因素：
- 性能原因：编译器在编译大型项目时非常耗时，所以LLVM希望设计性能更好的收据结构，从而加速编译效率；
- 编程效率：编译器中很多算法基于图实现，在ADT中定义了depth_iterator、graph_iterator可以直接遍历图，从而提高编程效率；
- 编程规范：定义统一的类型以及相关API，可以提供更丰富的实现，方便实现多种编译器后端。

需要提一点，LLVM中实现的ADT也有很多约束，和STD相比，主要区别有：
- 没有异常；
- 没有allocators作为模板参数；
- 没有API或者ABI稳定性保证；
- 更少的防御型代码；
- 只建议在LLVM内部使用。

另外LLVM中这一套新的数据结构对开发者提出了更高的挑战，需要学习新的数据结构以及学习API的使用。我们将在后续文章中剖析LLVM定义的ADT，主要介绍ADT的功能、使用。

<!-- more -->
