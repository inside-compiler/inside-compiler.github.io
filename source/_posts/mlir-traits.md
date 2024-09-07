---
title: 第4章（2）：特质
categories:
  - MLIR
tags:
  - MLIR
date: 2024-06-15 17:42:13
---


特质（Trait）是一种抽象机制的实现，针对属性、操作和类型进行修饰，主要有两个作用。
* 提供不同属性之间或不同操作之间或不同类型之间的通用机制的实现.
* 对属性、操作和类型提供约束。
根据特质的用途，目前MLIR主要提供了以下四种特质。
* NativeTrait：提供一种方式，将C++的声明和定义封装为一个特质，然后用于TD中修饰对象。
* ParamNativeTrait：继承于NativeTrait，但是这个特质目的是定义一个嵌套特质。
* GenInternalTrait：并没有与之对应的C++代码，而是用于mlir-tblgen自动生成代码。
* PredTrait：包含谓词的特质，通过该特质可以访问对应的谓词。例如mlir-tblgen工具会为PredTrait自动生成验证的代码。它和谓词最大的区别是，谓词一般用于单实体约束，而PredTrait通常用于多实体约束。

### 4.2.1特质的使用

特质作为一种基础能力，广泛用于MLIR中。可以通过操作的hasTrait成员函数判断操作是否存在相应的特质，例如操作为Op，特质为MyTrait，判断操作是否具有特质可以通过op->hasTrait<MyTrait>()完成。目前特质在MLIR中有三种典型的用途。
* mlir-tblgen工具自动生成代码过程，将操作定义的特质用作验证代码。例如特质SameVariadicOperandSize、AttrSizedOperandSegments等为操作中的操作数提供额外约束，mlir-tblgen工具会为操作生成额外的代码，用于约束操作数。这类特质一般继承于GenInternalTrait。
* 定义的特质作为标记符，在MLIR框架框架中通常被作为一组操作具有的公共能力。例如在MLIR有一个特质IsolatedFromAbove，使用该特质修饰的操作，表示该操作和它之前的区域隔离，即该操作不能再使用它之前区域定义的变量。这类特质主要用于MLIR框架中公共代码，例如对于定义IsolatedFromAbove的操作，MLIR框架可以认为操作不能进行跨区域优化（例如代码提升），从而约束操作相关优化。
* 定义特质用于运行时验证，此时要求特质除了定义外，还需要提供具体的验证实现。一般来说，特质需要实现verifyTrait接口。
注意：一般来说定义特质，仅用于表示操作具有这类属性，而不需要对特质中定义的成员函数提供具体的实现。原因是操作直接继承于特质模版，而特质中的成员函数不是虚函数，如果特质中成员函数提供具体的实现，所有操作都继承了特质中成员函数的实现都，所以一般特质较少提到具体的实现，更多是用于标记符。当然，通过模版继承的方式也可以让操作实现自己特有的功能，例如verifyTrait就是这样的例子，要求在模版中实现verifyTarit，操作中实现verifyTraitImpl，模版中verifyTrait调用操作的verifyTraitImpl，从而完成静态多态。具体的实现方法可以参考《深入理解LLVM：代码生成》中附录C关于奇异递归模版模式（CRTP）相关内容。

### 4.2.2自定义特质
下面以自定义特质为例，介绍如何定义和使用特质。
#### 1.定义特质
由于特质的用途不同，特质的定义和实现也有所不同。上面提到的三种用途，对于前两种，只需要在TD文件中定义特质，而最后一种用途除了在TD定义外，还需要在C++文件中实现相关接口。

##### TD中定义特质
由于特质可以分为四类，其中NativeTrait、ParamNativeTrait和GenInternalTrait都可以作为特质定义的基类。假设定义两个特质如代码4-6所示。

```//定义特质，名字为MyTrait
def MyTrait : NativeOpTrait<"MyTrait">;

// 定义待参数特质，第一个参数为cpp类名，第二个参数为字符串列表
class MyParametricTrait<int prop>  : NativeOpTrait<"MyParametricTrait", !cast<string>(!head(parameters))>;
```

这样2个特质继承于NativeOpTrait，表示它们是针对操作的特质。

##### C++中实现特质
在TD定义特质后，使用mlir-tblgen生成记录然后再生成C++代码。由于前面中已经详细介绍了从TD到记录再到C++代码的生成过程，本节不再赘述这一过程。默认情况下开发者需要实现verifyTrait和foldTrait，当然可以在特质定义时在TD中决定是否生成相关函数签名，针对代码4-6需要实现的C++代码如代码4-7所示。
```//实现MyTrait
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  //为具体操作重载verifyTrait函数，用于验证具体操作的正确性
  static LogicalResult verifyTrait(Operation *op) {
    // ...
  }
  //为具体操作重载foldTrait，用于执行操作折叠
  static OpFoldResult foldTrait(Operation *op, ArrayRef<Attribute> operands) {
    // ...
  }
  //不同foldTrait的签名
  static LogicalResult foldTrait(Operation *op, ArrayRef<Attribute> operands, SmallVectorImpl<OpFoldResult> &results) {
    // ...
  }
};
//特质MyParametricTrait的实现
template <int Parameter>
class MyParametricTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, Impl> {
    //通过参数Impl可以自由访问模版参数
    ...
  };
};
```
 
#### 2.关联特质
特质定义好以后就可以使用。既可以在TD中直接使用特质，也可以在C++代码直接使用，下面给出TD和C++代码使用特质的示例。
##### TD中使用特质
TD中使用特质比较简单，操作、类型、属性等都有参数用于指定特质。例如定义一个操作OpWithInferTypeInterfaceOp，它有参数用于接受特质列表，假设它使用特质MyTrait和MyParametricTrait，示例代码如4-8所示。
```def OpWithInferTypeInterfaceOp : Op<...[MyTrait, MyParametricTrait<10>]> { ... }
```

##### C++代码中直接使用
在C++使用特质也非常，如果在C++定义属性、操作和类型时，它们也都可以继承模版类，而模版类中都包含一个。假定定义属性MyAttr、MyOp和MyType，并且使用特质MyTrait和MyParametricTrait，代码如4-9所示。
```//定义属性MyAttr，其C++基类AttrBase包含一个变长的模版参数，用于接受多个特质
class MyAttr : public Attribute::AttrBase<MyAttr, ..., MyTrait, MyParametricTrait<10>::Impl> {};
//定义操作MyOp，其C++基类Op包含一个变长的模版参数，用于接受多个特质
class MyOp : public Op<MyOp, MyTrait, MyParametricTrait<10>::Impl> {};
//定义操作MyType，其C++基类TypeBase包含一个变长的模版参数，用于接受多个特质
class MyType : public Type::TypeBase<MyType, ..., MyTrait, MyParametricTrait<10>::Impl> {};

```

### 4.2.3MLIR中常见特质概览(TODO)


<!-- more -->
