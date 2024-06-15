---
title: 第4章（3）：接口
categories:
  - MLIR
tags:
  - MLIR
date: 2024-06-15 18:02:41
---


在MLIR中使用方言封装操作、类型和属性，通过自定义方言，开发者实现不同抽象层次的语义。这对MLIR框架来说带来一个问题，分析和转换可能需要针对每个方言或者每个操作。而接口机制正是为了解决这一问题而提出的，它可以为一类方言或者操作提供统一的转换和分析接口，而不需要涉及到一个具体的方言和操作。


### 4.3.1方言接口

方言接口通常适用于转换或分析，这些转换或分析希望对一组属性、操作或者类型进行通用操作（而非一个属性、操作或者类型），并且处理的属性、操作或者类型可能位于不同的方言定义中，由于方言接口的广泛性，所以一般它仅用于少数的分析或转换。例如，在一些场景中，直接在每个操作上注册接口过于复杂和繁琐。方言接口最典型的例子就是操作内联，它通常查询方言中操作的高级信息，如成本建模和合法性，这些信息通常不特定于某一个操作，因此多个操作可以共享内联这一功能。
需要注意的是：方言接口的核心不是操作，只是针对操作进行特定转换。
#### 1.接口定义
MLIR定义了方言接口基类DialectInterfaceBase::Base<>，开发者定义统一的方言接口，需要继承于该基类。然后针对不同的方言，继承自定义方言接口并实现方言特殊的处理。例如内联接口定义为：

```class DialectInlinerInterface :
    public DialectInterface::Base<DialectInlinerInterface> {
public:
  /// Returns true if the given region 'src' can be inlined into the region
  /// 'dest' that is attached to an operation registered to the current dialect.
  /// 'valueMapping' contains any remapped values from within the 'src' region.
  /// This can be used to examine what values will replace entry arguments into
  /// the 'src' region, for example.
  virtual bool isLegalToInline(Region *dest, Region *src,
                               IRMapping &valueMapping) const {
    return false;
  }
};
```

而不同的方言可以进行自己的特例化实现，例如Affine实现的内联如下：

```struct AffineInlinerInterface : public DialectInlinerInterface {
  /// Affine structures have specific inlining constraints.
  bool isLegalToInline(Region *dest, Region *src,
                       IRMapping &valueMapping) const final {
    ...
  }
};

/// Register the interface with the dialect.
AffineDialect::AffineDialect(MLIRContext *context) ... {
  addInterfaces<AffineInlinerInterface>();
}
```

#### 2.MLIR框架如何使用接口

```Dialect *dialect = ...;
if (DialectInlinerInterface *interface = dyn_cast<DialectInlinerInterface>(dialect)) {
  // The dialect has provided an implementation of this interface.
  ...
}

```
对于内联MLIR框架还辅助实现一个集合类（DialectInterfaceCollection）帮助访问所有内联，该集合类方便找到各种对象对应的方言，从而方便开发者通过集合对象统一访问方言接口。
#### 3.常见方言接口概览

### 4.3.2操作、属性和类型接口
MLIR允许分别为操作、属性和类型提供接口，它们的实现和使用方法几乎一致，本节以操作接口为例进行介绍。
#### 1.接口定义和实现
为了方便实现自定义接口，框架提供了OpInterface、AttributeInterface和TypeInterface基类用于支持自定义操作、属性和类型接口，它们的类结构如下所示：


下面我们以操作接口为例介绍接口的实现。操作接口需要继承于OpInterface，OpInterface的定义如下。

```template <typename ConcreteType, typename Traits>
class OpInterface
    : public detail::Interface<ConcreteType, Operation *, Traits,
                               Op<ConcreteType>, OpTrait::TraitBase> {
public:
  using Base = OpInterface<ConcreteType, Traits>;
  using InterfaceBase = detail::Interface<ConcreteType, Operation *, Traits,
                                          Op<ConcreteType>, OpTrait::TraitBase>;

  /// Inherit the base class constructor.
  using InterfaceBase::InterfaceBase;

protected:
  /// Returns the impl interface instance for the given operation.
  static typename InterfaceBase::Concept *getInterfaceFor(Operation *op) {
    OperationName name = op->getName();

    // Access the raw interface from the operation info.
    if (std::optional<RegisteredOperationName> rInfo =
            name.getRegisteredInfo()) {
      if (auto *opIface = rInfo->getInterface<ConcreteType>())
        return opIface;
      // Fallback to the dialect to provide it with a chance to implement this
      // interface for this operation.
      return rInfo->getDialect().getRegisteredInterfaceForOp<ConcreteType>(
          op->getName());
    }
    // Fallback to the dialect to provide it with a chance to implement this
    // interface for this operation.
    if (Dialect *dialect = name.getDialect())
      return dialect->getRegisteredInterfaceForOp<ConcreteType>(name);
    return nullptr;
  }

  /// Allow access to `getInterfaceFor`.
  friend InterfaceBase;
};
```

这里可以有一个关键的函数getInterfaceFor，它

其中OpInterface继承于Interface。它提供相关的API可以确定Operation实现的Interface。

```template <typename ConcreteType, typename ValueT, typename Traits,
          typename BaseType,
          template <typename, template <typename> class> class BaseTrait>
class Interface : public BaseType {
public:
  using Concept = typename Traits::Concept;
  ...

private:
  /// A pointer to the impl concept object.
  Concept *conceptImpl;
};

```
而Interface又继承于Op类。

```template <typename ConcreteType, template <typename T> class... Traits>
class Op : public OpState, public Traits<ConcreteType>... {
public:
 
  /// Return if this operation contains the provided trait.
  template <template <typename T> class Trait>
  static constexpr bool hasTrait() {
    return llvm::is_one_of<Trait<ConcreteType>, Traits<ConcreteType>...>::value;
  }
 
...

}
```

所以当定义操作，假设为class ExampleOpInterface : public OpInterface<ExampleOpInterface,...> ，当我们知道操作后可以根据集成关系通过强制类型转换获得接口类型。

```Operation *op = ...;
if (ExampleOpInterface example = dyn_cast<ExampleOpInterface>(op))
  llvm::errs() << "hook returned = " << example.exampleInterfaceHook() << "\n";
```
当然不是任意操作都可以通过强制类型转换得到对应的接口类型，如要保证类型换转换成功需要操作继承于对应的特质。

#### 2.接口框架机制
这里隐含了一个问题，为什么MLIR框架这样设计接口？
主要原因是方便开发者扩展内置接口的实现。

虽然接口提供了通用性处理，但是对于一些操作来说，部分接口的实现可能并不满足定制化需求。

```struct ExampleOpInterfaceTraits {
  //在特质中定义Concept类，这个类包含了所有接口类声明的API
  struct Concept {
    virtual unsigned exampleInterfaceHook(Operation *op) const = 0;
    virtual unsigned exampleStaticInterfaceHook() const = 0;
  };

  //在mLIR框架中通过会为内置接口提供Model类的实现，这样就保证了内置接口提供了默认的行为
  template <typename ConcreteOp>
  struct Model : public Concept { /*...*/ };

  //在特质中还有FallbackModel和ExternalModel两个，用于对接开发者自定义实现
  //MLIR框架会为FallbackModel类实现提供所有API的实现，但是这些实现仅仅是把真正的调用转发到getImpl()对应的真实Model
  template <typename ConcreteOp>
  struct FallbackModel : public Concept {
    unsigned exampleInterfaceHook(Operation *op) const override {
      getImpl()->exampleInterfaceHook(op);
    }
    unsigned exampleStaticInterfaceHook() const override {
      ConcreteOp::exampleStaticInterfaceHook();
    }
  };

  //MLIR框架会为ExternbackModel类实现提供所有API的声明，通常实现为空。
  template <typename ConcreteModel, typename ConcreteOp>
  struct ExternalModel : public FallbackModel<ConcreteModel> {
    unsigned exampleInterfaceHook(Operation *op) const override {
      // Default implementation can be provided here.
      return type.cast<ConcreteOp>().callSomeTypeSpecificMethod();
    }
  };
};

```

所以当开发者将自己实现的model注册到接口时，就会执行到开发者提供的代码

```MyFallBackExampleOpInterface : public ExampleOpInterfaceTraits::FallbackModel<
          MyFallBackExampleOpInterface> {
   ...
}
```

注意为了代码简写

 ``` using FallbackModel = typename Traits::template FallbackModel<T>;
  using InterfaceBase =
      Interface<ConcreteType, ValueT, Traits, BaseType, BaseTrait>;
  template <typename T, typename U>
  using ExternalModel = typename Traits::template ExternalModel<T, U>;
所以上面的  ExampleOpInterfaceTraits::FallbackModel也可以写为ExampleOpInterface::FallbackModel
```

然后将Fallbackmodel注册到方言中

MyDialect::attachInterface<MyFallBackExampleOpInterface>(context);

对于ExternalModel稍微有点不同，它需要传递一个ConcreteOp，这个ConcreteOp是接口运行的操作。例如针对内置操作func::CallOp添加一个新的接口。

```struct MyExternalExampleOpInterface
    : ExampleOpInterface::ExternalModel<ExampleOpInterface,
                                       func::CallOp> {
  unsigned exampleInterfaceHook(Operation *op) const override {
    ...
    return true;
  }
};
```

将新实现的接口注册到func::CallOp，注册代码示例如下：

```void mlir::builtin::registerFuncCallOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
func::CallOp::attachInterface<MyExternalExampleOpInterface>(*ctx);
  });
}
```

#### 3.接口案例分析

##### TD文件
```def FunctionOpInterface : OpInterface<"FunctionOpInterface", [
    Symbol, CallableOpInterface
  ]> {
 
  ...

}
```

第一步：mlir-tblgen在处理OpInterface时处理TD中的继承。

```def FunctionOpInterface {// Interface Trait NativeTrait InterfaceTrait OpInterfaceTrait OpInterface
  code description = [{...  }];
  string cppInterfaceName = "FunctionOpInterface";
  string cppNamespace = "::mlir";
  list<InterfaceMethod> methods = [...]; // InterfaceMethod
  string extraClassDeclaration = "";
  code extraSharedClassDeclaration = [{...  }];
  string extraClassOf = "";
  list<Interface> baseInterfaces = [Symbol, CallableOpInterface];
  string trait = "FunctionOpInterface::Trait";
  code extraConcreteClassDeclaration = [{}];
  code extraConcreteClassDefinition = [{}];
  code extraTraitClassDeclaration = [{...}];
  string verify = "return function_interface_impl::verifyTrait(cast<ConcreteOp>($_op));";
  bit verifyWithRegions = 0;
  list<Trait> dependentTraits = [];
}
```

在mlir-tblgen的第二步中会将对应的字段进行解析，生成对应的C++代码。这里仅关注继承这个功能，本质上mlir-tblgen中并不能直接支持C++类的继承关系的生成，所以后端实现。在mlir-tblgen中引入了字段baseInterfaces字段，用于保存接口继承要继承的父接口。
而生成的类时，会直接接口的继承层次打平，即直接进行继承。

生成的C++对象如下所示

```struct FunctionOpInterfaceInterfaceTraits {
    struct Concept {...};
class Model : public Concept {...};
class FallbackModel : public Concept {...};
class ExternalModel : public FallbackModel<ConcreteModel> {...};
}

class FunctionOpInterface : public ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits> {
public:
    using ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits>::OpInterface;
    template <typename ConcreteOp>
    struct Trait : public detail::FunctionOpInterfaceTrait<ConcreteOp> {};
 
    ...
// 函数主要来来自于FunctionOpInterface中InterfaceMethod、extraSharedClassDeclaration定义

};

namespace detail {
template <typename ConcreteOp>
struct FunctionOpInterfaceTrait : public ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
   ...
   // 函数主要来来自于FunctionOpInterface中extraTraitClassDeclaration、extraSharedClassDeclaration定义

}
}
```

#### 4.常见操作、属性和类型接口概览

FunctionOpInterface：是一个OpInterface，在Interface介绍时提到该接口的生成过程，这里需要注意它不仅是Interface，还是一个Trait，通过OpInterface处理后，操作还可以继承于OpInterface（所以定义了OpInterface后操作可以转换为FunctionOpInterface指针）。接口定义了很多Function相关的API，例如getFunctionType、getFunctionBody、getBlocks、getArguments、getArgAttrs等操作以及针对操作进行Insert、Erase相关操作。
FuncOp直接继承于Traits？所以强制转换为FuncOpInterface类型。为什么不直接对Trait也使用强转，而是使用hasTrait？代码更为优雅。

RegionKindInterface：是一个OpInterface，通过该接口定义了getRegionKind、hasSSADominance函数。主要用于区分不同类型Region。

OpAsmOpInterface：是一个OpInterface，通过该接口和AsmPrinter、AsmParser进行交互。定义了getAsmResultNames、getAsmBlockArgumentNames、getAsmBlockNames、getDefaultDialect共计4个API。前面3个更多的使用输出定制化IR，最后一个API返回默认的方言名字。例如Arith中ConstOp会实现getAsmResultNames，对于i1这个类型将结果1和0分别转换为true和flase。

CallOpInterface：是一个OpInterface，类似于函数调用，控制流从一个子函数到另一个子函数。例如直接函数调用、间接函数调用。提供getCallableForCallee、setCalleeFromCallable、getArgOperands、getArgOperandsMutable、resolveCallable共计4个API。主要用于在一些跨函数分析中，例如inliner、数据流分析。调用者需要使用该特质属性。

CallableOpInterface：是一个OpInterface，接口主要约束函数被调用。提供getCallableRegion、getArgumentTypes、getResultsTypes、getArgAttrsAttr、getResAttrsAttr、setArgAttrsAttr、setResAttrsAttr、removeArgAttrsAttr、removeResAttrsAttr相关API。例如FunctionOpInterface包含了还特质，说明FuncOp可以被调用。该特质也主要用于优化，例如Inliner、数据流分析、调用图等。

CastOpInterface：是一个OpInterface，类似于cast操作，将输入类型转换输出类型。提供areCastCompatible、foldCastInterfaceOp共2个接口。通过该接口可以判断操作是否可以接受相关类型的转换。还可以通过该接口尝试折叠操作（目前的约束是操作输入类型和输出类型完全相同）。

BranchOpInterface：是一个OpInterface，对于分支、跳转类操作可以定义该特质，表示操作有分支行为，例如br、cond_br、switch等操作可以使用该特质。通过该特质可以明确确定操作是分支操作，可以用于优化，例如在一些CFG处理中寻找分支指令并进一步分析。

RegionBranchOpInterface：是一个OpInterface，描述了操作包含的区域中有分支行为。例如for或者if操作，其中的区域存在分支行为。定义了getEntrySuccessorOperands、getEntrySuccessorRegions、getSuccessorRegions、getRegionInvocationBounds、areTypesCompatible、isRepetitiveRegion等API。该接口主要用于优化操作，例如bufferization基于该接口判断是否可以对数据、内存进行缓存，通过BranchOpInterface、RegionBranchOpInterface、RegionBranchTerminatorOpInterface找出操作所有依赖的Value，从而确定Buffer的位置。在例如在代码下沉优化时，可以直接通过RegionBranchOpInterface的后续Region作为下沉的位置。

RegionBranchTerminatorOpInterface：是一个OpInterface，和RegionBranchOpInterface对应，描述操作是终止区域。

CopyOpInterface：类似于copy操作，例如memref.copy、buffer.clone包含该接口。该特质getSource、getTarget共2个API。在MemRef的copy降级过程可以借助该特质验证源和目的类型。

DataLayoutOpInterface（TBD）
DataLayoutSpecInterface
DataLayoutEntryInterface
DataLayoutTypeInterface

DerivedAttributeOpInterface：通过该接口获取操作继承的属性。目前社区中暂无实际使用。

DestinationStyleOpInterface（TBD）：描述目的传递风格的接口（Destination-PassingStyle）。主要用于优化中，表示输入和输出可以共用内存。

InferIntRangeInterface：该接口表示操作可以支持推断结果的边界，例如Arith中Add操作，当输入范围确定，则可以推断结果的范围（当然结果可能发生溢出，则范围则为Null）。需要操作实现inferResultRanges函数。它主要用于数据流分析的整数范围推断。

InferTypeOpInterface：该接口表示操作可以支持推断结果的类型。

InferShapedTypeOpInterface：该接口表示操作可以支持推断结果的Shape类型。

ReifyRankedShapedTypeOpInterface：该接口表示操作可以支持得到结果的Shape类型。

LoopLikeOpInterface：该接口表示操作是一个循环，例如for操作继承于该接口。该接口定义了一系列API，可以用于进行优化（比如LICM）。和RegionBranchOpInterface有一定的关系？？

PromotableAllocationOpInterface：表示分配操作可以提升，当然提升需要借助支配关系。例如llvm方言中alloc操作继承于该接口，在Mem2Reg对alloca分配位置进行优化。
PromotableMemOpInterface：表示访存操作可以提升，例如llvm方言中load、store操作可以继承于该接口。
PromotableOpInterface：表示一些特殊的内存操作可以提升，例如llvm方言中bitcast、addrspacecast、gep操作可以继承于该接口。
DestructurableAllocationOpInterface：表示分配操作可以SORA优化，例如llvm中的alloca、memref方言中的alloca操作。
SafeMemorySlotAccessOpInterface：表示访存操作可以安全访问，这样可以方便调度IR，例如llvm方言中load、store、gep操作可以继承于该接口。
DestructurableAccessorOpInterface：表示访存操作可以安全析构。例如llvm方言中gep操作可以继承于该接口、memref方言中的load、store操作。
DestructurableTypeInterface：表示操作中的类型可以分解为子类型处理，例如llvm方言中的array操作可以分解为元素类型处理。

ParallelCombiningOpInterface：定义一个区域可以并行组合的接口。在Buffer优化中需要将并行合并放在Buffer之外。

RuntimeVerifiableOpInterface：操作支持运行时进行眼验证。主要是针对动态方言的功能，继承该接口的操作需要实现generateRuntimeVerification。

SubsetOpInterface：操作可以使用Tensor的子集。
SubsetExtractionOpInterface：操作支持从Tensor抽取一个子集。
SubsetInsertionOpInterface：操作支持向Tensor插入一个子集。

TilingInterface：继承该接口的操作允许Tilling。

PartialReductionOpInterface：对scf.forall进行tilling。目前有一个操作TileReductionUsingForallOp继承该接口。

ValueBoundsOpInterface：这是一个基类，表示操作如果有Index类型或者Shaped类型存在一直值边界（Value-Bound）。

VectorUnrollOpInterface：接口表示操作可以被展开。例如vector方言中的很多操作contract、reduciton、transfer_read、transfer_write、gather、transpoe等都可以展开。另外Aith、math所有操作都可以支持展开。

VectorTransferOpInterface：表示transfer操作。例如vector方言中的transfer_read、transfer_write都继承该操作。

ViewLikeOpInterface：表示操作支持View操作。例如memref、vector中的一些操作可以继承该接口。View更多的是在Buffer中以及一些优化中使用。

OffsetSizeAndStrideOpInterface：

ShapedDimOpInterface：

GPU_AsyncOpInterface

LinalgStructuredInterface


DeclareOpInterfaceMethods：OpInterface的辅助功能，它组合OpInterface还继承于OpInterface，主要目的时提供接口封装，可以屏蔽原始接口的实现，仅仅将traits增加到操作中。什么时候需要使用？

Class DeclareOpInterfaceMethods<OpInterfaceinterface,
list<string>overridenMethods=[]>
:DeclareInterfaceMethods<overridenMethods>,
OpInterface<interface.cppInterfaceName,interface.baseInterfaces>{
Let description=interface.description;
Let cppInterfaceName=interface.cppInterfaceName;
Let cppNamespace=interface.cppNamespace;
Let methods=interface.methods;
Let baseInterfaces=interface.baseInterfaces;
}

这个辅助类主要是约束操作是否实现接口中尚未定义的函数。如果使用DeclareOpInterfaceMethods对接口进行封装后，继承于该辅助类的操作需要显式实现接口中尚未实现的API。

下面以func方言中的CallOp为例，看看使用辅助类的主要作用。
defCallOp:Func_Op<"call",
[CallOpInterface, MemRefsNormalizable,
DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{

...

Let extraClassDeclaration=[{
FunctionType getCalleeType();

///Get the argument operands to the called function.
operand_range getArgOperands(){
return{arg_operand_begin(),arg_operand_end()};
}

MutableOperandRange getArgOperandsMutable(){
Return getOperandsMutable();
}

operand_iterator arg_operand_begin(){return operand_begin();}
operand_iterator arg_operand_end(){return operand_end();}

///Return the callee ofthis operation.
CallInterfaceCallable getCallableForCallee(){
return(*this)->getAttrOfType<SymbolRefAttr>("callee");
}

///Set the callee for this operation.
Void setCalleeFromCallable(CallInterfaceCallable callee){
(*this)->setAttr("callee",callee.get<SymbolRefAttr>());
}
}];

}

存在三个Traits。以CallOpInterface和SymbolUserOpInterface这两个接口为例来介绍相关用法。

对于CallOp直接继承于DeclareOpInterfaceMethods<SymbolUserOpInterface>，说明SymbolUserOpInterface中尚未实现的API都会在CallOp中声明，并要求CallOp提供对应的实现，例如SymbolUserOpInterface中verifySymbolUses没有实现，所以在CallOp中会重新声明，并要求提供实现。如果不实现，在编译期就会出错。这一方法要求接口中InterfaceMethod没有实现，否则不会在操作中生成接口的声明。

另外对于DeclareOpInterfaceMethods的使用除了提供接口名外，还可以携带另外一个参数alwaysOverriddenMethods，这个参数指定了需要在操作实现的函数，而不是将接口中所有未定义的函数都放在操作中实现。也就是说操作仅仅使用参数alwaysOverriddenMethods指定的函数，如果使用其他未重新实现的函数也会出错。这一方法即使接口中InterfaceMethod有实现，也会在操作中生成接口的声明，因此参数称为重载函数名。

对于CallOp直接继承于CallOpInterface，而在CallOpInterface定义了4个API，分别是getCallableForCallee、setCalleeFromCallable、getArgOperands、getArgOperandsMutable。并且这4个API在CallOpInterface中没有实现。而在CallOp直接继承于CallOpInterface，默认情况CallOp并也不会生成这个4个API的定义和实现。也就说是，如果CallOp不进行额外的处理，当通过CallOp（例如强制转换为CallOpInterface）调用这4个API函数会出错。所以在CallOp中提供了extraClassDeclaration字段，并在TD中对CallOpInterface涉及到的4个API进行了实现。


Class CallOp : public::mlir::Op<CallOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults,::mlir::OpTrait::ZeroSuccessors,::mlir::OpTrait::VariadicOperands,::mlir::OpTrait::OpInvariants,::mlir::BytecodeOpInterface::Trait,::mlir::CallOpInterface::Trait,::mlir::OpTrait::MemRefsNormalizable,::mlir::SymbolUserOpInterface::Trait>{
public:
...
::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection& symbolTable);
...

public:
FunctionType getCalleeType();

///Get the argument operands to the calle dfunction.
operand_rangeget ArgOperands(){
Return { arg_operand_begin(), arg_operand_end() };
}

MutableOperandRange getArgOperandsMutable(){
Return getOperandsMutable();
}

operand_iterator arg_operand_begin(){returnoperand_begin();}
operand_iterator arg_operand_end(){returnoperand_end();}

///Return the callee of this operation.
CallInterfaceCallable getCallableForCallee(){
return(*this)->getAttrOfType<SymbolRefAttr>("callee");
}

///Set the callee for this operation.
Void setCalleeFromCallable(CallInterfaceCallable callee){
(*this)->setAttr("callee",callee.get<SymbolRefAttr>());
}
};


上述讨论的函数InterfaceMethod
Description：方法的描述。
ReturnType：方法的返回值。
MethodName：方法的名字。
Arguments：方法参数，可选提供。
MethodBody：方法体，可选提供。如果提供了方法体，则方法体则会放在接口中。
DefaultImplementation：方法体，可选提供。如果提供了方法体，则方法体则会放在Traits中。

<!-- more -->
