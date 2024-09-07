---
title: 第4章（1）：约束
categories:
  - MLIR
tags:
  - MLIR
date: 2024-06-15 17:03:29
---

在第2.3节提到的MLIR框架通过谓词、特质为操作、类型、属性提供约束，从而尽可能保证操作、属性、类型的正确性。接口借助于特质完成操作、属性、类型动态处理能力。本章主要介绍谓词、特质、接口的实现以及如何正确使用它们。

## 4.1 谓词
谓词（Pred）是对操作或方言使用的类型、属性等进行限制。由于谓词是在操作或者方言构建时使用，本书将其称为静态约束。在MLIR框架中还有约束（Constraint），它继承于谓词，并且约束对谓词增加了描述字段（Summary），可以为谓词提供更好的错误提示（由于约束在中文中含义过于笼统，本书统一使用谓词）。同时谓词之间可以通过And（交集-多个谓词同时满足）、Or（并-多个谓词至少满足一个）、Contact（链接-将多个谓词链接成一个谓词）、Negative（非-满足谓词取反）、SubstLeaves（替代-将谓词中部分进行替代为新的谓词）等进行组合谓词。
MLIR框架定义最基础（或原子）的谓词定义CPred，其定义如代码4-1所示。

```class CPred<code pred> : Pred {
  //谓词的真正实现代码，是cpp代码
  code predExpr = "(" # pred # ")";
}
```

CPred传递的类型为TD的code类型。在参数中可以传递一些占位符（Placeholder）和执行上下文进行关联，例如针对操作提供的占位符有$_builder、$_op、$_self等，这些占位符会被mlir-tblgen工具替换为合适的对象，例如$_builder会被替换为mlir::Builder（表示对象构造），$_op表示当前的操作，而$_self则会根据上下文进行替换为当前的对象。例如一个简单的谓词定义为CPred<"::llvm::isa<::mlir::F32>">，其中::llvm::isa<::mlir::F32>对应的代码为llvm::isa<Float32Type>，其中F32表示Float32Type。
MLIR框架提供的谓词可以针对类型、属性、区域和后继。其中区域和后继约束作用于操作的IR结构，类型约束主要作用于操作的操作数和结果的定义，属性约束作用于操作中的属性。
在MLIR中，对每一种类型进行验证都会在TD中定义相应的类。例如要验证类型是否为Float16Type，TD定义F16，如代码4-2所示。

```class Constraint<Pred pred, string desc = ""> {
  // 谓词，描述约束
  Pred predicate = pred;
  // 描述信息
  string summary = desc;
}
//类型约束类
class TypeConstraint<Pred predicate, string summary = "",
                     string cppClassNameParam = "::mlir::Type"> :
    Constraint<predicate, summary> {
  // 限定生成的代码位于::mlir::Type命名空间内
  string cppClassName = cppClassNameParam;
}
//类型的约束，在TD中用Type表示
class Type<Pred condition, string descr = "",
           string cppClassName = "::mlir::Type"> :
    TypeConstraint<condition, descr, cppClassName> {
  string description = "";
  string builderCall = "";//构造cpp中类型对象的函数
}
class BuildableType<code builder> {
  code builderCall = builder;
}
//在TD中可用的浮点约束数记录类，以F开头，后跟数字
class F<int width>
    : Type<CPred<"$_self.isF" # width # "()">,
           width # "-bit float", "::mlir::FloatType">,
      BuildableType<"$_builder.getF" # width # "Type()"> {
  int bitwidth = width;
}
//F16示例
def F16 : F<16>;
```

使用mlir-tblgen工具将代码4-2翻译成记录，结果如代码4-3所示。


```//F16的记录
def F16 {// Constraint TypeConstraint Type BuildableType F，是F16的继承记录信息
  Pred predicate = anonymous_35; //谓词对应的记录类
  string summary = "16-bit float";//简单描述
  string cppClassName = "::mlir::FloatType";//cpp类名
  string description = "";//描述信息
  string builderCall = "$_builder.getF16Type()";//对应的类型对象构造函数
  int bitwidth = 16;//浮点数位宽
}
//F16的谓词
def anonymous_35 {
  //这个谓词判断当前类型是否为Float16类型
  string predExpr = "($_self.isF16())";
}

```

例如在一个操作为MyOperation中可以直接类型约束F16、F32、F64等。在MyOperation有arguments和results，它们分别表示MyOperation的输入和输出。示例代码如如代码4-4所示。

```//在MyDialect中定义一个操作MyOperation
def MyOpeation : MyDialect_Op<"MyOpeartion", [Pure]> {
    //操作数有2个，分别是lhs和rhs，它们的类型约束分别为F16和F32
    let arguments = (ins F16:$lhs, F32:$rhs);
    //操作的返回值，类型约束为F64
    let results = (outs F64:$result);
}
```

操作MyOperation的arguments表示操作有两个参数分别是lhs和rhs，它们的类型约束分别是F16和F32（F16和F32是MLIR社区定义的浮点数类型，它们是类型为::mlir::FloatType，并且宽度分别是16和32位）；results表示MyOperation的输出变量为result，类型约束为F64。
使用mlir-tblgen工具对代码4-4生成C++代码，结果类似于3.3节AddI操作。类似于AddI操作MyOperation对应的C++代码有两个函数：verifyInvariants和verifyInvariantsImpl，在函数verifyInvariantsImpl中会验证输入、输出类型的验证（验证不仅仅包含输入、输出类型，还包括区域个数、后继基本块信息、属性、谓词特质信息），验证规则是TD的定义得到。F16、F32、F64等类型验证规则是根据字段predExpr生成相应的代码，当然在生成代码时先将predExpr中$_self替换为MyOpertion中对应的参数。例如对于F16和F32生成的验证规则对应C++代码如代码4-5所示。

```static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MyOperationOps0(
    ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef valueKind, unsigned valueIndex) {
  //判断类型是否是F16和F32，如果不是则报错
  if (!(((type.isF16())) || ((type.isF32())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 16-bit float or 32-bit float, but got " << type;
  }
  return ::mlir::success();
}
```
验证规则对应的代码__mlir_ods_local_type_constraint_MyOperationOps0在MyOperation中的verifyInvariantsImpl函数中被调用。
在构建操作对象后会验证对象，在创建操作对象后调用verifyInvariants，最终调用到具体的类型验证，即上述的__mlir_ods_local_type_constraint_MyOperationOps0代码，从而保证MyOperation对象输入和输出都是合法类型。除了mlir-tblgen工具自动生成的类型、属性、区域、后继等信息验证外。MLIR框架也支持开发者自己实现的验证函数（在操作定义时通过设置let hasVerifier = 1，mlir-tblgen工具会生成verify的函数声明，实现需要开发者完成），参考第3.3节介绍。
注意：谓词和编译器中类型系统异同点有哪些？

4.1.1自定义谓词和使用

MLIR社区定义了很多谓词，能满足大多数场景的使用。当开发者遇到一些需要额外约束的场景，开发者可以自定义谓词。例如我们要约束操作的操作数个数是否满足要求，可以定义谓词class CheckNumOperands<int i> : CPred<"$_op.getNumOperands() == " # i>。定义完成后就可以在TD中直接使用该谓词。而mlir-tblgen工具会将谓词转换为相应的代码。不过需要注意的是这个谓词是功能谓词，不是类型或者属性，可以通过PredTrait使用（4.2节介绍）。

4.1.2MLIR中常见谓词概览 （TODO）

<!-- more -->
