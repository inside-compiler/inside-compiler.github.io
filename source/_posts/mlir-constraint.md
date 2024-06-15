---
title: 第4章（1）：约束
categories:
  - MLIR
tags:
  - MLIR
date: 2024-06-15 17:03:29
---

正如2.3节提到的MLIR框架通过特质为操作、类型、属性提供约束，从而尽可能保证操作、属性、类型的正确性。接口借助于特质完成操作、属性、类型公共处理能力。本章主要介绍谓词、特质、接口的实现以及如何正确使用它们。

## 4.1 谓词
谓词（Pred）是对操作和方言进行类型、属性等进行限制。由于谓词是在操作或者方言构建时使用，本书将其称为静态约束。在MLIR约束（Constraint）继承于谓词。约束对谓词增加了描述字段（Summary），可以为谓词提供更好的错误提示。同时谓词之间可以通过And、Or、Contact、Negative、SubstLeaves等进行组合谓词。

MLIR定义最基础（或原子）的谓词定义CPred，其定义如下

```class CPred<code pred> : Pred {
  code predExpr = "(" # pred # ")";
}
```

CPred传递的类型为TD的code类型。在参数中可以传递一些占位符（Placeholder）和执行上下文进行关联，例如针对操作提供的占位符有$_builder、$_op、$_self等，这些占位符会被mlir-tblgen替换为合适的对象，$_builder会被替换为mlir::Builder，$_op表示当前的操作，而$_self则会根据上下文进行替换。

通过继承于约束，实现类型、属性、区域和后继约束。其中区域和后继约束作用于操作的IR结构，类型约束主要作用于操作的操作数和结果的定义，属性约束作用于操作中的属性，在MLIR为了针对每一种类型进行验证都会在TD中定义相应的类。例如F16类型，在TD中继承体系如下代码所示：

``class Constraint<Pred pred, string desc = ""> {
  // The predicates that this constraint requires.
  Pred predicate = pred;
  // User-readable one line summary used in error reporting messages. If empty,
  // a generic message will be used.
  string summary = desc;
}``

```class TypeConstraint<Pred predicate, string summary = "",
                     string cppClassNameParam = "::mlir::Type"> :
    Constraint<predicate, summary> {
  // The name of the C++ Type class if known, or Type if not.
  string cppClassName = cppClassNameParam;
}

class Type<Pred condition, string descr = "",
           string cppClassName = "::mlir::Type"> :
    TypeConstraint<condition, descr, cppClassName> {
  string description = "";
  string builderCall = "";
}

class BuildableType<code builder> {
  // The builder call to invoke (if specified) to construct the BuildableType.
  code builderCall = builder;
}

class F<int width>
    : Type<CPred<"$_self.isF" # width # "()">,
           width # "-bit float", "::mlir::FloatType">,
      BuildableType<"$_builder.getF" # width # "Type()"> {
  int bitwidth = width;
}

def F16 : F<16>;
```

对应的记录

```def F16 {// Constraint TypeConstraint Type BuildableType F
  Pred predicate = anonymous_35;
  string summary = "16-bit float";
  string cppClassName = "::mlir::FloatType";
  string description = "";
  string builderCall = "$_builder.getF16Type()";
  int bitwidth = 16;
}

def anonymous_35 {// Pred CPred
  string predExpr = "($_self.isF16())";
}

```

例如我们在一个操作MyOperation中可以直接类型F16、F32、F64等。在MyOperation中通常可以定义arguments和results，它们分别表示MyOperation的输入和输出。示例代码如下：

```def MyOpeation : MyDialect_Op<"MyOpeartion", [Pure]> {

    let arguments = (ins F16:$lhs, F32:$rhs);
let results = (outs F64:$result);

}
```

其中arguments表示在MyOperation会有两个参数分别是lhs和rhs，它们的类型分别是F16和F32（F16和F32是MLIR社区定义的浮点数类型，它们是类型为::mlir::FloatType，并且宽度分别是16和32位）。results表示MyOperation的输出变量为result，类型为F64。

在mlir-tblen生成C++表示的MyOperation，其定义是根据TD文件得到（参考*.*.*）。同时会在C++的MyOperation中增加一个函数：::mlir::LogicalResult verifyInvariants()和verifyInvariantsImpl()，在verifyInvariantsImpl()中会验证输入、输出类型的验证（验证不仅仅包含输入、输出类型，还包括region个数、successor信息、属性、谓词特质信息），验证规则是TD的定义得到。验证规则就是根据字段“predExpr”生成相应的代码，当然在生成代码时先将$_self替换为MyOpertion中对应的参数。例如对于F16和F32生成的验证规则对应C++代码如下：

```static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MyOperationOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isF16())) || ((type.isF32())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 16-bit float or 32-bit float, but got " << type;
  }
  return ::mlir::success();
}
```

__mlir_ods_local_type_constraint_MyOperationOps0在MyOperation中的verifyInvariantsImpl函数被调用。

在Op构建时进行验证，当创建操作对象后会调用verifyInvariants，最终调用到具体的类型验证。除了mlir-tblgen自动生成的类型、属性、区域、后继等信息验证外。MLIR也支持开发者自己实现的验证函数（let hasVerifier = 1时mlir-tblgen会生成verify的函数申明，而实现需要开发者进行实现），参考第3章介绍。

4.1.1自定义谓词和使用

开发者可以自定义谓词，例如我们要操作的操作数个数是否正确，可以定义谓词如下：

class CheckNumOperands<int i> : CPred<"$_op.getNumOperands() == " # i>;

这样就可以在TD中直接使用。mlir-tblgen会将谓词转换为相应的代码。不过需要注意的是这个谓词是功能谓词，不是类型或者属性，所以使用的使用一般可以通过PredTrait使用。

4.1.2MLIR中常见谓词概览 （TODO）

<!-- more -->
