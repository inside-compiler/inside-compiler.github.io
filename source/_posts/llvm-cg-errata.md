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

1. 部分图文不一致说明
由于印刷原因，书中部分图文不完全一致。书中文字介绍图中蓝色线由于印刷统一印刷为红色线，例如在第97页第二段提到：DAG中chain关系用蓝色虚线，实际印刷为红色虚线，glue关系用蓝色实线，实际印刷为红色实线。故图7-7中红色虚线表示chain关系，在本书其它地方类似。

2. 在”5.2.1 循环识别“这一小节的第一段最后一句“逆序遍历待控制流对应的支配树”修正为“逆序遍历待分析的控制流对应的支配树”。
<!-- more -->
