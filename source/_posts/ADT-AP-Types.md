---
title: ADT-任意精度类型介绍
categories:
  - LLVM
tags:
  - [LLVM, MLIR, ADT]
date: 2024-02-02 22:00:26
---

# 基础数据类型
ADT的基础类型有APInt、APSInt、APFloat和APFixedPoint。

<!-- more -->

## 整数类型

ADT提供整数类型，包括APInt（Arbitrary Precision Integer）和APSInt（Arbitrary Precision Sign Integer）两种类型，分别是任意精度的整数、任意精度的符号整数。

### APInt

APInt主要描述任意精度整数，类似于传统的unsinged、unsigned long、uint64_t。但是APInt还能表述任意位数的整数，例如3
、7、15，甚至超过64位精度的整数。

APInt的设计思路为：使用一个字段bitwidth记录整数的位宽，使用另外一个字段U记录数据。bitwidth可以是任意宽度，U字段是一个联合结构体，当bitwdith小于等于64时，直接使用unit64_t类型存储原始数据；当bitwidth大于64时使用unit64_t类型的指针表示。APInt的数据结构如下所示：

从APInt的设计中可以看出，任意整数都被转换成无符号数进行存储，例如-1在APInt中被表示为0XFFFF,FFFF,FFFF,FFFF。在APInt中判断整数是否是负数的依据是最高位是否为1，如果最高位为1，说明表示的值为负数。需要注意的是，APInt本身并不知道数字是否有无符号，依赖于使用者进行确定。对于加法来说，直接根据unit64_t类型直接相加，这里可能带来一个问题，对于unit64_t直接相加的结果可能溢出，所以APInt提供了一系列关于加法相关的API，加法API例如：
- operator+=：两个APInt直接相加，由使用者确定结果该如何使用；
- sadd_ov：两个APInt直接相加，但是将结果视为符号数，如果结果的最高位和两个操作数只要的最高位不同，则说明发生了溢出；
- uadd_ov：两个APInt直接相加，但是将结果视为无符号数，如果结果大于其中一个操作数则说明发生了溢出；
- sadd_sat：调用sadd_ov，如果不发生溢出，直接返回；如果发生溢出，则返回最大值（结果为正数）或者最小值（结果为负数时）；
- uadd_sat：调用uadd_ov，如果不发生溢出，直接返回；如果发生溢出，则返回最大值。
所以开发者需要明确确定该属于哪一个API。

APInt提供一些基础API，包括加减乘除、取模、左移、右移、取整、逻辑运算等基础操作；还提供了一些高级API，例如计算0的个数、1的个数、log等；还有和unit64_t、float、double、string转换的方法。

#### APInt示例

### APSInt

由于APInt中统一使用unit64_t进行各种运算，对于符号数、无符号数需要开发者根据场景选择不同的API。这对使用者不太方便，为此ADT中封装了APSInt，其中明确指定了符号标记信息，从而在使用统一的API而不用刻意区分符号数、无符号数的API。

#### APSInt示例

## 浮点数类型

APFloat主要描述任意浮点数，这里的任意并不是指和APInt一样任意位数的浮点数，任意指的是将所有可以支持的浮点数进行封装，例如float、double、bfloat、quard、x87、8e5m2等IEEE标准以及x87特殊浮点数等各种后端支持的浮点数类型。所有的浮点数可以通过三部分表示，这三部分是：符号数、指数和尾数三个部分。
| type | 指数最大值 | 指数最小值 | 符号位 |  尾数位| 总位数 |
|--|--|--|--|--|--|
| half | 15 | -14 | 1 | 11 | 16 |
| bfloat | 127 | -126 | 1 | 8 | 16 |
| float | 127 | -126 | 1 | 24 | 32 |
| double | 1023 | -1022 | 1 | 53 | 64 |
| 8E5M2 | 15 | -14 | 1 | 3 | 8 |
| 8E5M2FNUZ | 15 | -15 | 1 | 3 | 8 |
| 8E4M3FN | 8 | -6 | 1 | 4 | 8 |
| 8E4M3FNUZ | 7 | -7 | 1 | 4 | 8 |
| 8E4M3B11FNUZ | 4 | -10 | 1 | 4 | 8 |
| FloatTF32 | 127 | -126 | 1 | 11 | 19 |
| X87DoubleExtended | 16383 | -16382 | 1 | 64 | 80 |
| PPCDoubleDouble | -1 | 0 | 1 | 0 | 128 |
| PPCDoubleDoubleLegacy | 1023 | -1022+53 | 1 | 53+53 | 128 |

APFloat也提供了一些对应的API，包括加减乘除、比较、abs、比较操作等，以及提供了从float、double、string和APFloat的相互转化。

#### APFloat示例
APFloat F1(-0.0);



## 定长类型

APFixedPoint主要描述任意的定义浮点数，这里定长指的是浮点数的总位数固定，任意指的是提供不同类型的指数、小数偏移位（Scale缩放因此）可以存在不同的划分方式。该数据类型的标准在ISO/IEC/ANSI进行定义，主要用在嵌入式系统中。示例如下所示：


例如 （16,7）

17.99609375f 对应的二进制为0x11.FF。

```Val = APFloat(17.99609375f);
sem = FixedPointSemantics(/*width=*/16, /*scale=*/7, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);

APFixedPoint(val, sem);
```

0x8FF，实际上这个数字等于17.9921875 （并不等于Val17.99609375f ）。
