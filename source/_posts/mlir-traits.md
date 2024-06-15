---
title: 第4章（2）：特质
categories:
  - MLIR
tags:
  - MLIR
date: 2024-06-15 17:42:13
---


特质（Trait）是一种抽象机制的实现，针对属性、操作和类型进行修饰，主要有两个作用：
1. 1）提供不同属性之间或者不同操作之间，或者不同类型之间的通用机制的实现；
2. 2）对属性、操作和类型提供约束。

根据特质的用途，目前MLIR主要提供了以下4种特质：
1. NativeTrait：提供一种方式，将C++的声明和定义封装为一个特质，然后用于TD中修饰对象。
2. ParamNativeTrait：继承于NativeTrait，但是这个特质目的是定义一个嵌套特质。
3. GenInternalTrait：并没有与之对应的C++代码，而是用于mlir-tblgen自动生成代码。
4. PredTrait：包含谓词的特质，通过该特质可以访问对应的谓词。例如在mlir-tblgen中会为PredTrait生成自动验证的代码。它和谓词最大的区别是，谓词一般用于单实体约束，而PredTrait通常用于多实体约束。

### 4.2.1特质的使用

在MLIR中特质作为一种基础能力，广泛用于MLIR中。通常来说可以通过操作的接口hasTrait判断是否存在相应的特质，例如操作为Op，特质为MyTrait，判断操作是否具有特质可以通过op->hasTrait<MyTrait>()进行。目前特质在MLIR中有3种典型的用途：
1.1、定义的特质用于mlir-tblgen自动代码生成验证代码使用。例如特质SameVariadicOperandSize、AttrSizedOperandSegments等为操作中的操作数提供额外约束，会为操作生成额外的代码。这类特质一般继承于GenInternalTrait。
2.2、定义的特质用于占位符，作为MLIR框架识别一类操作应该具有的公共能力。例如在MLIR定义IsolatedFromAbove，它表示该操作和它之前的区域隔离，即该操作不能再使用它之前区域定义的变量。这类特质主要用于MLIR框架中用于公共代码，例如对于定义IsolatedFromAbove的操作，MLIR框架可以认为操作不能进行跨区域优化（例如代码提升），从而约束操作相关优化。一般来说定义特质，仅用于表示操作具有这类属性，而不需要对特质提供具体的实现。
3.3、定义特质用于运行时验证，此时要求特质除了定义外，还需要提供具体的验证实现。一般来说，特质需要实现verifyTrait接口。

### 4.2.2自定义特质

#### 1.定义特质
由于特质的用途不同，特质的定义和实现也有所不同。上面提到的3种用途，对于前两种，只需要在TD文件中定义特质，而最后一种用途还需要在C++文件中实现用途相关接口。

##### TD中定义特质
除了上面的提到的继承GenInternalTrait特质外，还可以继承NativeTrait、ParamNativeTrait定义特质，假设定义两个特质分别如下：

```def MyTrait : NativeOpTrait<"MyTrait">;

// The first argument is the parent c++ class name. The second argument is a
// string containing the parameter list.
class MyParametricTrait<int prop>
  : NativeOpTrait<"MyParametricTrait", !cast<string>(!head(parameters))>;
```

这样2个特质继承于NativeOpTrait，表示它们是针对操作的特质。

##### C++中实现特质

```template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  /// Override the 'verifyTrait' hook to add additional verification on the
  /// concrete operation.
  static LogicalResult verifyTrait(Operation *op) {
    // ...
  }
};

  static OpFoldResult foldTrait(Operation *op, ArrayRef<Attribute> operands) {
    // ...
  }
 
  static LogicalResult foldTrait(Operation *op, ArrayRef<Attribute> operands,
                                 SmallVectorImpl<OpFoldResult> &results) {
    // ...
  }
 

template <int Parameter>
class MyParametricTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, Impl> {
    // Inside of 'Impl' we have full access to the template parameters
    // specified above.
  };
};
```
 
#### 2.关联特质

##### C++代码中直接使用

```class MyAttr : public Attribute::AttrBase<MyAttr, ..., MyTrait, MyParametricTrait<10>::Impl> {};
/// Here we define 'MyOp' along with the 'MyTrait' and `MyParametric trait
/// classes we defined previously.
class MyOp : public Op<MyOp, MyTrait, MyParametricTrait<10>::Impl> {};
/// Here we define 'MyType' along with the 'MyTrait' and `MyParametric trait
/// classes we defined previously.
class MyType : public Type::TypeBase<MyType, ..., MyTrait, MyParametricTrait<10>::Impl> {};

```
#####  TD中使用

```def OpWithInferTypeInterfaceOp : Op<...[MyTrait, MyParametricTrait<10>]> { ... }
```

### 4.2.3MLIR中常见特质概览

是抽象出实现细节以及在许多不同属性/操作/类型等之间通用的属性（properties）的机制。特性可用于说明这个对象特殊的属性以及约束，包括一个操作是否有副作用，或者它的输出是否与输入有相同的类型。操作特性的一些例子有Commutative、SingleResult、Terminator等。

GenInternalOpTrait：在mlir-tblgen生成代码时会根据该特质约定的信息进行合法性验证，如果不合法不会生成代码；
NativeOpTrait：生成真实的C++代码；

AttrSizedOperandSegments：描述的是操作中变长参数实际长度。一些操作中使用了Variadic的参数，所以需要通过Trait进行约束，说明该变长参数真实的长度。对于该特质来说，mlir-tblgen会为操作对应的C++类生成字段operandSegmentSizes，该字段是Array类型，存储了Variadic的真实长度。

SameVariadicOperandSize：

SameOperandsAndResultType

DataLayoutOpInterface

HasDefaultDLTIDataLayout

AttrSizedResultSegments:：描述的是操作中结果为变长参数的实际长度，非常类似于AttrSizedOperandSegments。对于该特质来说，mlir-tblgen会为操作对应的C++类生成字段resultSegmentSizes，该字段是Array类型，存储了Variadic的真实长度。

IsolatedFromAbove：表示该操作和它之前Region的操作进行了隔离，即该操作不能再使用上面Region中操作定义的Value。使用该操作最主要的目的是重新定义Value之间的支配关系。例如FuncOp操作就会定义该Trait，这样每一个Func中定义的Value不会被其他Func支配，每个Func中的外部引用必须通过参数或者属性传递到Func中。再例如一些操作提升操作中，如果操作定义了该Trait，则对应的操作提升不能跨Region进行提升。在MLIR框架中有不少地方依赖该Trait，例如OpPassManager中的操作必须标记该Trait。

例如有一个方言foo，其中操作region_op定义了IsolatedFromAbove，假设有一个代码片段如下：
%result=arith.constant10:i32
foo.region_op{
foo.yield%result:i32
}
这个代码片段是非法，原因是region_op使用了%result，而result属于另外一个Region。对于这样的代码，需要将%result作为参数传递给region_op。

RegionKindInterface：表示操作中的区域属性，分为SSACFG、Graph两种，其中SSACFG表示操作中Def-Use使用符合SSA、CFG特性；例如Fucn操作就是典型的SSACFG。而Graph属性表示Region中操作没有任何顺序要求。注意当前Region为Graph属性时，仅支持一个基本块；例如Module操作则是典型的Graph属性，Module操作中包含的操作并无执行顺序要求。

ConstantLike：表示操作定义了常量。在框架中会使用该Trait进行优化，例如进行常量折叠时可以要求操作定义了该Trait。

ConditionallySpeculatable：是一个OpInterface，定义了操作是否产生UB行为或者无限循环行为。例如Aith中Div操作需要使用该Trait，表示Div操作可能存在溢出、除零等UB行为；再例如一些对于数组、向量、张量等访存行为，如果允许越界，一般会定义越界后是UB行为。一般定义该Trait的操作，需要实现getSpeculatability函数，定义对于什么样的情况结果情况，一般结果可以为Speculation::NotSpeculatable（产生UB行为），Speculatable（没有产生UB行为），RecursivelySpeculatable（需要进一步根据操作包含的子操作判断是否产生UB行为）三种。

AlwaysSpeculatableImplTrait：该Trait定义了默认投机行为，即定义了getSpeculatability函数，默认返回Speculatable。

RecursivelySpeculatableImplTrait：该Trait定义了默认投机行为，即定义了getSpeculatability函数，默认返回RecursivelySpeculatable。

AlwaysSpeculatable：是ConditionallySpeculatable和AlwaysSpeculatableImplTrait两个Trait的合体，直接使用该Trait较少（更多情况是使用下面的Pure）。

RecursivelySpeculatable：是ConditionallySpeculatable和RecursivelySpeculatableImplTrait两个Trait的合体。例如Affine的if操作、Parallel操作直接使用该Trait。

MemoryEffectsOpInterface：是一个OpInterface（注意OpInterface继承于OpInterfaceTrait，即OpInterface也会被视为Trait），跟踪操作是否对内存修改。

MemoryEffect：继承于MemoryEffectsOpInterface，为了方便使用，传递不同的参数表达不同的内存影响，例如Alloc、Free、Read、Write等不同的内存行为产生影响不同，所以进一步可以封装MemAlloc、MemAllocAt、MemFree、MemFreeAt、MemRead、MemReadAt、MemWrite、MemWriteAt。注意MemoryEffect是继承于OpInterface。

MemoryEffects：接受MemoryEffect类型的list作为参数，并且继承于SideEffectsTraitBase。可以理解为将MemoryEffect转换为MemoryEffects的Trait类型。

NoMemoryEffect：继承于MemoryEffects，但是不传递任何参数，表示操作不存在内存影响。

RecursiveMemoryEffects：表示操作对内存的影响，依赖于操作包含的子操作决定。所以会递归处理操作包含的子操作。

Pure：是AlwaysSpeculatable和NoMemoryEffect两个Trait的合体。表示操作即没有内存影响也没有UB或者无限循环行为，所以操作可以被提升或者下沉。

AffineScope：该Trait主要用于限制多面体优化的范围。例如Affine.FuncOp这类操作定义了AffineScope，表示FuncOp操作定义了多面体优化范围，在FuncOp顶部的操作可以被后面操作（如Affine.for、Affine.load、Affine.store等使用）；支配FuncOp操作也可以被FuncOp内的操作使用；或者FuncOp的参数也可以被FUncOp内的操作使用。

AutomaticAllocationScope：定义该Trait的操作会对分配内存进行自动释放。例如FuncOp操作中定义了该Trait，那么对于该操作中需要使用临时变量时则可以进行分配。再例如对于Bufferization优化中，会寻找包含该Trait的操作作为Buffer分配的位置。

IsTerminator：表示操作中区域的最后一个的基本块中最后一个操作是Terminator属性的操作。

NoTerminator：表示操作中区域的最后一个的基本块中没有Terminator属性的操作。

NoRegionArguments：表示操作中的区域没有参数，主要用于验证。

ReturnLike：表示操作蕴含Return的含义，它主要用于控制流中，表示当前的控制流执行结束（和函数的Return有所不同，ReturnLike弱于函数的Return功能）。所以Return操作、Yield操作都可以定义ReturnLike特质。在编译优化中可以直接判断操作是否包含该Trait，在部分优化中还会将操作转换为ReturnOp进行优化。

```Def ReturnLike:TraitList<[
DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
NativeOpTrait<
/*name=*/"ReturnLike",
/*traits=*/[],
/*extraOpDeclaration=*/"",
/*extraOpDefinition=*/[{
::mlir::MutableOperandRange $cppClass::getMutableSuccessorOperands(
::mlir::RegionBranchPointpoint){
Return ::mlir::MutableOperandRange(*this);
}
}]
>
]>;
```


SymbolTable：

Symbol：

MemRefsNormalizable

SingleBlock：表示操作中区域仅仅包含一个基本块。

SingleBlockImplicitTerminator：表示操作中区域仅仅包含一个基本块，并且包含一个有Terminator属性的操作（有Trait的模板参数确定）。使用该Trait后，对应的操作的parse需要调用ensureTerminator函数，用于显式指定操作包含Terminator属性。例如AffineForOp包含了SingleBlockImplicitTerminator<AffineYieldOp>，表示ForOp最后一个操作是AffneYieldOp。

TypesMatchWith


```
//A type constraint that denotes`transform(lhs.getType())==rhs.getType()`.
//An optional comparator function maybe provided that changes the above form
//into:`comparator(transform(lhs.getType()),rhs.getType())`.

Class TypesMatchWith<string summary, string lhsArg, string rhsArg,
String transform,string comparator="std::equal_to<>()">
:PredOpTrait<summary,CPred<
comparator#"("#
!subst("$_self","$"#lhsArg#".getType()",transform)#
",$"#rhsArg#".getType())">>{
String lhs=lhsArg;
String rhs=rhsArg;
String transformer=transform;
}

Class PredOpTrait<string descr,Pred pred,list<Trait> traits=[]>
:PredTrait<descr,pred>{
//Specify the list of traits that need to be verified before the verification
//of this PredOpTrait.
list<Trait> dependentTraits=traits;
}

Class PredTrait<string descr,Pred pred>:Trait{
String summary=descr;
Pred predicate=pred;
}
```

HasParent

```OpTrait::HasParent<op>::Impl

//Op's parent operation is the provided one.
Class HasParent<string op>
:ParamNativeOpTrait<"HasParent",op>, StructuralOpTrait;

Class ParamNativeOpTrait<string prop,string params,
list<Trait> traits=[]>
:ParamNativeTrait<prop,params,"Op">{
//Specify the list of traits that need to be verified before the verification
//of this ParamNativeOpTrait.
list<Trait> dependentTraits = traits;
}

Class ParamNativeTrait<string prop,string params,string entityType>
:NativeTrait<prop#"<"#params#">::Impl", entityType>;
```


<!-- more -->
