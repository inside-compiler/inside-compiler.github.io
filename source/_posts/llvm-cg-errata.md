---
title: 《深入理解LLVM：代码生成》勘误表
categories:
  - LLVM
tags:
  - LLVM
date: 2049-10-06 15:33:38
sticky: 100
---
# 勘误表

2025年1月第二版次印刷修正了如下错误。

1. 部分图文不一致说明
由于印刷原因，书中部分图文不完全一致。书中文字介绍图中蓝色线由于印刷统一印刷为红色线，例如在第97页第二段提到：DAG中chain关系用蓝色虚线，实际印刷为红色虚线，glue关系用蓝色实线，实际印刷为红色实线。故图7-7中红色虚线表示chain关系，在本书其它地方类似。

2. 在”5.2.1 循环识别“这一小节的第一段最后一句“逆序遍历待控制流对应的支配树”修正为“逆序遍历待分析的控制流对应的支配树”。

3. 45 页表3-3 liveuse集合 错误
   p75 代码6-6上 calss ->class
4. p53：到达基本块 6 的所有路径都经过基本块 2

    p159：LiveRefGens[rx] --> LiveRegGens[rx]

    p237：Indexe -->Indexes

     p251：REG_SEQUCENCE --> REG_SEQUENCE

     p336：PROT顺序 --> RPOT顺序

    p396：垃圾回收期 --> 垃圾回收器

5. 书中“寄存器分配前的调度算法...而寄存器分配后的调度算法...还需要关注寄存器分配的压力”。
这里的“寄存器分配前”和“寄存器分配后”写反了。

感谢网友 @ywgrit，@punkyc，@IanIsMyUsername，@mengdouer，sunshaoce 的反馈。
<!-- more -->
