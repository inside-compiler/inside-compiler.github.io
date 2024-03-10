---
title: Survey_on_Clang
categories:
  - Clang
tags:
  - Clang
date: 2023-12-31 17:03:58
---

## 背景

clang 是一个类 C 语言的编译器前端和基础设施工具。开发 clang 的目的是为了更好的编译报错提示信息、更优的 IDE 亲和性和更友好的 License。

<!-- more -->

## 整体架构

clang 是基于模块化的架构设计，具有以下几个模块。

- libsupport，基础支持库。
- libsystem，系统抽象库。
- libbasic，诊断信息，源代码相关信息，文件缓存。
- libast，表示C 语法树、C 类型系统和内置函数等各种数据结构。还有 AST 分析和操作的辅助函数（访问器、打印器等）。
- liblex，词法分析和预处理。
- libparse，语法分析。
- libsema，语义分析，构建标准语法树。
- libcodegen，将AST变换成LLVMIR。
- librewrite，改写文本缓冲区。
- libanalysis，静态分析。
- clang，驱动器。

## 开发手册

这个手册是面向clang开发者的，主要是描述了重要API的信息和内部设计决策信息。

### LLVM支持库

提供了底层库和数据结构。

- 命令行选项处理
- 系统抽象层

### Clang基础库

提供了源码缓冲区管理、诊断信息机制、单词信息、目标抽象信息和语言信息。

### 诊断子系统

诊断信息是编译器和开发人员之间沟通的重要组成部分。一个诊断信息由以下几部分组成：

- ID
- 英文描述的报错信息
- 源码位置信息
- 严重程度（如，WARNING/ERROR）

在Diagnostic*Kinds.td配置好每个诊断的信息，然后通过tblgen生成clang所需的诊断信息格式。

## 代码过程



## 其它

模块化设计的原则：

1. 以独立代码库的方式组织每个功能；
2. 每个库有清晰的 APIs，相互解耦，易于修改和扩展；
3. 有简洁的分层和合适的设计。



## 参考

1. [clang 常用的命令]([Clang - Getting Started (llvm.org)](https://clang.llvm.org/get_started.html))
2. [开发手册](["Clang" CFE Internals Manual — Clang 18.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/InternalsManual.html))
3. 