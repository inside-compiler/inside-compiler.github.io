---
title: MLIR学习系列1-MLIR概述
categories:
  - MLIR
tags:
  - MLIR
date: 2024-01-01 10:21:22
---

# MLIR学习系列1-MLIR概述

MLIR自2019年发布后就成为编译界的新贵，不仅仅AI编译器、领域编译器、新型硬件公司和新型语言都在尝试使用MLIR，其中以AI编译器取得的成果最为显著。本文简单介绍MLIR的基础知识、MLIR构建和MLIR相关工具的使用。

## MLIR是什么？

MLIR是Multi-Level Intermediate Representation的缩写。MLIR提供一套新的编译基础架构，方便编译器开发人员定义多层IR，并基于IR进行优化。
MLIR定义的IR通过方言（Dialect）进行管理，Dialect又包含了Operation，而Operation是IR对于的操作，因此可以认为Dialect是Operation的集合。
通过Dialect的定义，可以将代码转换至对应的方言，再通过对方言的降级和优化，最后下降至LLVM IR，重用LLVM的编译框架完成优化、代码生成。

## MLIR构建

MLIR是LLVM的子项目，默认情况下构建LLVM时并不会构建MLIR子项目。所以为了使用MLIR在构建LLVM时需要添加额外的参数。编译命令如下：

```
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build . --target check-mlir
```
在构建时主要添加构建参数`-DLLVM_ENABLE_PROJECTS=mlir`即可。编译完成后就可以看到MLIR配套相关的工具。

## MLIR相关工具使用

MLIR项目构建完成后会生成一些工具。其中最主要的包括：
* mlir-tblgen：将TD文件转换C++代码，和MLIR'框架配合使用
* mlir-opt：对MLIR方言进行下降或者执行优化
* mlir-translate：将LLVM方言转换为LLVM IR；

另外，其它还有一些工具，例如mlir-pdll、mlir-cpu-runner等在后续的blog中涉及到相关内容再进行介绍。

## MLIR使用示例

下面通过一个简单的示例演示MLIR的工作流程。

### 使用MLIR定义一个Hello MLIR的代码

```
module {
        memref.global "private" constant @string : memref<13xi8> = dense<[0x48,0x65,0x6c,0x6c,0x6f,0x2c,0x20,0x4d,0x4c,0x49,0x52,0x21,0]>
        llvm.func external @puts(!llvm.ptr<i8>) -> ()
        func.func @main() -> i64 {
                %c0_i64 = arith.constant 0 : i64
                %0 = memref.get_global @string : memref<13xi8>
                %1 = memref.extract_aligned_pointer_as_index %0 : memref<13xi8> -> index
                %2 = arith.index_cast %1 : index to i64
                %3 = llvm.inttoptr %2 : i64 to !llvm.ptr<i8>
                llvm.call @puts(%3) : (!llvm.ptr<i8>) -> ()
                return %c0_i64 : i64
        }
}
```
在这个例子中，使用4个方言，分别是memref、func、arith和llvm（例如代码中memerf.global表示memref方言中的global operation）。通过memref定义一个全局变量的字符串“Hello，MLIR！”
使用llvm方言引用一个外部函数put；
使用func方言定义main函数，并在main函数获取字符串，并将其作为put函数的参数，调用put函数。

### 将MLIR方言下降至LLVM方言
在MLIR世界中，需要将其它的方言下降至LLVM方言（包含LLVM和后端硬件方言，烈日ArmSVE方言），在本例中需要将func、arith、memref下降到LLVM方言，使用命令如下：
`mlir-opt  -convert-func-to-llvm  -finalize-memref-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts hello.mlir `

这个代码如下：

```
module attributes {llvm.data_layout = ""} {
  llvm.mlir.global private constant @string(dense<[72, 101, 108, 108, 111, 44, 32, 77, 76, 73, 82, 33, 0]> : tensor<13xi8>) {addr_space = 0 : i32} : !llvm.array<13 x i8>
  llvm.func @puts(!llvm.ptr<i8>)
  llvm.func @main() -> i64 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(13 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.null : !llvm.ptr
    %4 = llvm.getelementptr %3[13] : (!llvm.ptr) -> !llvm.ptr, i8
    %5 = llvm.ptrtoint %4 : !llvm.ptr to i64
    %6 = llvm.mlir.addressof @string : !llvm.ptr
    %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<13 x i8>
    %8 = llvm.mlir.constant(3735928559 : index) : i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %7, %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.insertvalue %13, %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %1, %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.insertvalue %2, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr<i8>
    llvm.call @puts(%18) : (!llvm.ptr<i8>) -> ()
    llvm.return %0 : i64
  }
}
```
该代码只包含了LLVM方言
### 将LLVM方言转换为LLVM IR
将LLVM方言转换为LLVM IR，使用命令如下`mlir-translate --mlir-to-llvmir ***.mlir
`
生成的LLVM IR如下

```
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@string = private constant [13 x i8] c"Hello, MLIR!\00"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @puts(ptr)

define i64 @main() {
  call void @puts(ptr @string)
  ret i64 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

### 运行测试LLVM IR
生成的LLVM IR就可以接入到LLVM体系中，可以使用LLVM的优化、代码生成、汇编、反汇编工具。例如可以使用lli直接运行上述的LLVM IR，可以得到如下结果：
`Hello， MLIR！`

## 小结

本文作为MLIR概述的第一篇，简单介绍MLIR的基本概念和工具，通过一个示例演示通过MLIR工具生成LLVM IR的过程。后续我们将逐步展开介绍MLIR方言、模式匹配、方言降级和优化等工作。