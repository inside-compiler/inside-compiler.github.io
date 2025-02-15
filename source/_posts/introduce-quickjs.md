---
title: introduce quickjs
categories:
  - virtual machine
  - js engine
tags:
  - virtual machine
date: 2025-02-08 15:23:23
---

## 背景

QuickJS是Fabrice Bellard和Charlie Gordon开源的一个轻量级JS引擎，在2019年开源了第一个版本代码。目前最新版本是20240113，已经支持了ES2023大部分特性。下面主要介绍下它的组成模块和衍生项目，其它的项目信息和性能数据可以参考[官网文档](https://bellard.org/quickjs/)。

<!-- more -->

## 主要模块

QuickJS是一个基于解释器的引擎，没有JIT和AOT，它由以下模块构成：

- 字节码生成器，它读入JS源文件并解析成解析树，然后生成字节码。可以离线执行这个字节码生成器生成字节码文件。
- 解释器，基于字节码的解释器，用于解析字节码命令的语义。
- 垃圾回收器，使用引用计数的垃圾回收算法进行内存回收。
- 运行时，管理JS程序执行期间的上下文信息，如栈、常量和堆数据等；以及提供一些内置的功能函数。

其中，QuickJS的字节码是按栈式的形式设计的，这是因为栈式字节码更紧凑，可以减少字节码文件大小。

目前QuickJS[代码库](https://github.com/bellard/quickjs)里的总代码量有86K行，其中C代码有77K行。它编译后的静态库libquickjs.a的大小是2.3M。

## 衍生项目

因为QuickJS在低内存占用的情况下，有比较不错的性能表现，在海量低性能设备上具有比较大的可用性，所以出现了不少QuickJS的开源衍生项目。下面是一些关注度比较高的项目。

- [quickjs-ng](https://github.com/quickjs-ng/quickjs)，社区维护的QuickJS，做了一些性能优化。
- [OpenQuickJS](https://github.com/OpenQuickJS/quickjs)，webf定制的QuickJS，做了一些性能优化。
- [vincentdchan/quickjs](https://github.com/vincentdchan/quickjs)，给QuickJS添加了CMake构建系统。
- [PetterS/quickjs](https://github.com/PetterS/quickjs)，一个基于QuickJS的Python库，方便Python中使用JS代码。
- [GodotJS](https://github.com/godotjs/GodotJS)，一个兼容了V8/JSC/QuickJS的JS引擎。

## 总结

Quick JS 通过极简架构与高效实现，在嵌入式开发、轻量级脚本工具等场景中展现了独特的竞争力。对于追求高性能、低资源占用的开发者，Quick JS 是值得深入研究的解决方案。此外，由于它的代码量相对较小，也是入门学习语言虚拟机原理和实现的优秀项目。

## 参考资料

1. [QuickJS官方网站](https://bellard.org/quickjs/)
2. [QuickJS code](https://github.com/bellard/quickjs)

