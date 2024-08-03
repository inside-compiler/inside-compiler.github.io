---
title: 第5章：Pass和Passmanager
categories:
  - MLIR
tags:
  - MLIR
date: 2024-08-03 19:13:49
---
# 第5章：Pass和Passmanager

[TOC]

MLIR中也提供了变换（Transformation）和分析（Analysis）的概念，变换一般指的是将IR进行优化，生成新的IR（和方言降级略有不同，降级一般不涉及优化，仅仅简单的讲某一算子变成一个或多个新类型的IR）；而分析是提供一些信息供优化使用。但是在MLIR中分析并不是以Pass的形式存在，而是以类的形式存在，即分析后的数据通过类结构进行缓存，在多处进行共享。本节主要讨论变换Pass，最后简单介绍分析管理。
注意：MLIR的分析和LLVM中的分析存在很大不同，LLVM分析是以Pass的方式存在，并且可以穿插在变换之间，同时提供了一套相对完善的用法，例如缓存分析结果等。但是MLIR非常困难提供类似于LLVM中的分析Pass，原因是MLIR中IR不统一，基于一种IR的分析很难在其他类型的IR中使用，导致很难提供一套既能在多种IR共享数据，且同时可以作用对多种IR进行分析的框架。所以MLIR中分析是以类的形式存在，仅仅提供多种IR共享数据的能力。

## 5.1Pass、Pipeline及管理

提供Pass、Pass Pipeline、OpPassManager、PassManager等，方便自定义Pass、Pass Pipeline。在MLIR中定义了Pass和Passpipeline处理变换，而Pass和Passpipeline则通过PassManager进行管理。
### 5.1.1Pass介绍
MLIR中变换的基础是Operation。所以框架中定义了基类OperationPass，所有的Pass都继承于OpeartionPass，它有几个特点：
* 针对特定的Operation进行处理；对于一般的Pass来说，如果没有指定Operation则是针对Void进行处理，但是对于Pass Pipeline来说必须指定一些信息，其中包含Operation，否则不能执行；
* 提供了接口canScheduleOn，用于指定Pass运行于特定的Operation；
* 提供了接口getAnalysis，用于获取分析结果。

为了方便，MLIR还提供了InterfacePass，表示Pass仅仅作用于某一些Interface限定的Operation。如果Operation没有限定Interface，不会执行相关Pass。

当Pass定义过滤条件时，如显式指定Pass适用的Operation或者指定适用的Interface，Pass运行，如果发现过滤条件不满足，会终止对该Operation的处理（即不会运行所有的Pass，并不是跳过不适用的Pass）。
这在框架中也有所体现，如果定义了Pass、Pass Pipeline，当多个Pipeline连续在一起时，会进行合并，同时将有过滤条件的pipeline放置在前面，在其中的Pass、Pipeline都不会执行。

### 5.1.2Pass定义

Pass可以通过TD文件进行定义。再TD文件中Pass有以下属性：
* summary：定义Pass概要的信息。
* description：定义Pass详细的描述信息，在自动生成的文档。
* dependentDialects：声明Pass依赖的的方言，只有在此处声明后的方言，才可以在该Pass中使用。该字段对应的函数韦getDependentDialects，并且在该函数中针对registry调用Insert函数，将Dialect加载到Pass中，这样Pass就可以使用。如果没有加载Dialect，Pass使用相关Dialect就会出错。
* constructor：定义实例化Pass代码，如果没有在TD文件中定义，则自动生成一个默认的函数，实例化Pass。
* options：定义Pass使用的参数。
* statistics：定义Pass的统计属性，在统计Pass信息时使用。

注意：Pass中dependentDialects使用会有一些怪异，Pass中可以使用的方言必须是MLIRContext已经加载过的方言，如果方言未加载则不能使用，如果使用则会报错（找不到对应方言）。另外方言的加载可能发生在多线程执行环境中，所以加载一般需要放在Pass真正运行之前，否则会报“并发运行错误和不安全运行错误”。但是MLIRContext已经加载的方言比较难以确定，因为方言降级路径不唯一，所以会导致MLIRContext中加载的方言不确定。所以一般会在此处将该Pass所依赖的方言都进行加载，如果方言已经加载并不会重复加载。当然如果可以确定所有进入到该Pass的路径都已经加载了相关方言，Pass可以进行忽略加载方言。例如有一个Pass要处理linalg方言中的操作，如果无论何种路径进入到该Pass时都已经加载过linalg方言，那么该Pass可以不添加依赖再去加载linalg方言。
### 5.1.3Pass管理
Pass和Pass pipeline的管理和执行由PassManager负责。PassManager首先将所有Pass、Pass pipeline中定义的依赖方言全部加载到PassManager中，只有加载过的方言才能被使用。
对Pass进行排序（Pass pipeline作为一个Pass参与），排序时按照过滤Pass优先、一般Pass在后的原则进行。只有当所有的Pass都可以针对Op执行时才会真正执行Pass，否则直接返回。（不会存在执行部分Pass的场景）

### 5.1.4Pass注册
Pass和Pass PipeLine需要注册至PassManager后才能使用。
Pass和Pipeline分别管理，它们分别使用一个全局Map管理注册信息。在每个Pass和Pipeline中会调用框架中的函数mlir::registerPass和PassPipelineRegistration分别将Pass和Pipeline注册到全局变量中。在构造PassManager时将相关的Pass、Pipeline进行初始化。

### 5.1.5Pass定义以及Pass和框架配合

在TD文件中定义新的Pass，主要涉及到几个变量，例如名字、基类、参数等。在MLIR框架中在TD中通过PassBase约定了自定义Pass包括了哪些参数，PassBase对应的代码片段
```Class PassBase<string passArg,string base>{
//Pass的名字，在opt命令行中使用
String argument = passArg;
//定义了Pass的基类，这个基类指的是C++类，通过是Operation
String baseClass = base;
//Pass的简单描述，这个描述体现在opt--help输出
String summary = "";
//Pass的完整描述，这个描述体现在自动生成的文档中
String description = "";
//Pass的构造函数，如果没有在TD文件中定义，则自动生成一个默认的函数签名
Code constructor=[{}];
//Pass依赖的方言，当Pass要使用方言中的Operation，说明Pass依赖了方言，需要将方言注册到Pass中
//该字段对应的函数为getDependentDialects，并且在该函数中针对registery调用Insert函数，将Dialect加载到Pass中，这样Pass就可以使用。如果没有加载Dialect，Pass使用相关Dialect就会出错。
list<string> dependentDialects=[];
//Pass的参数
list<Option> options=[];
//Pass的统计信息，在统计Pass信息时使用。在TD文件中定义变量名、描述信息，就可以统计Pass的执行次数，并可以通过PassManager中的enableStatistics函数打开Pass执行次数统计，并在Pass执行结束后输出信息，在mlir-opt工具中可以通过参数mlir-pass-statistics直接使用。
list<Statistic> statistics=[];
}
```
注意：Pass中dependentDialects使用会有一些怪异，Pass中可以使用的Dialect必须是当前Context已经加载过的Dialect，如果方言未加载则不能使用。另外方言的加载可能发生在多线程执行环境中，所以加载一般需要放在Pass真正运行之前，否则会报“并发运行不安全运行错误”。但是Context已经加载的方言本质上又不确定，因为方言降级路径不唯一，所以会导致Context中加载的方言不确定。所以一般会在此处将该Pass所依赖的方言都进行加载，如果方言已经加载并不会重复加载。如果可以确定所有进入到该Pass的路径都已经加载了相关方言，Pass可以进行忽略。例如Pass处理linalg算子，那么无论何种路径进入到Pass都已经加载过linalg方言，否则在进入Pass之前就会出错，那么该Pass不需要再次加载linalg方言。

例如MLIR中Affine中有一个Pass提前循环中不变量，将其提升到循环的外部，从而加速执行效率。

```Def AffineLoopInvariantCodeMotion
:Pass<"affine-loop-invariant-code-motion","func::FuncOp">{
Lets ummary="Hoist loop invariant instructions out side of affine loops";
Let constructor="mlir::createAffineLoopInvariantCodeMotionPass()";
}
```
通过命令行使能该Pass的名字为affine-loop-invariant-code-motion，该Pass处理操作类型为func::FuncOp（这是func方言中的FuncOp操作类），不会处理其他操作。Summary字段定义了Pass的简单描述，在opt的帮助中出现。constructor表示为Pass提供了构造函数，函数名为createAffineLoopInvariantCodeMotionPass，这个函数需要用户进行实现。
该记录展开后对应的记录为：

```Def AffineLoopInvariantCodeMotion{//Pass BasePass
String argument="affine-loop-invariant-code-motion";
String baseClass="::mlir::OperationPass<func::FuncOp>";
String summary="Hoist loop invariant instructions out side of affine loops";
String description="";
String constructor="mlir::affine::createAffineLoopInvariantCodeMotionPass()";
list<string> dependentDialects=[];
list<Option> options=[];
list<Statistic> statistics=[];
}
```
该Pass通过mlir-tblgen后得到C++风格的头文件如下：

```template<typename DerivedT>
Class AffineLoopInvariantCodeMotionBase : public ::mlir::OperationPass<func::FuncOp>{
public:
Using Base = AffineLoopInvariantCodeMotionBase;

AffineLoopInvariantCodeMotionBase() : ::mlir::OperationPass<func::FuncOp>(::mlir::TypeID::get<DerivedT>()){}
AffineLoopInvariantCodeMotionBase(const AffineLoopInvariantCodeMotionBase &other) : ::mlir::OperationPass<func::FuncOp>(other){}

///Returns the command-line argument attached to this pass.
Static const expr::llvm::StringLiteral getArgumentName(){
Return ::llvm::StringLiteral("affine-loop-invariant-code-motion");
}
::llvm::StringRef getArgument() const override { return"affine-loop-invariant-code-motion"; }

::llvm::StringRef getDescription() const override { return "Hoistloopinvariantinstructionsoutsideofaffineloops";}

///Returns the derived passn ame.
Static constexpr::llvm::StringLiteral getPassName(){
Return ::llvm::StringLiteral("AffineLoopInvariantCodeMotion");
}
::llvm::StringRef getName() const override { return "AffineLoopInvariantCodeMotion"; }

///Support is a/dyn_cast functionality for the derived pass class.
Static bool classof(const::mlir::Pass *pass){
Return pass->getTypeID()==::mlir::TypeID::get<DerivedT>();
}

///A clone method to create a copy of this pass.
std::unique_ptr<::mlir::Pass>clonePass() const override{
Return std::make_unique<DerivedT>(*static_cast<const DerivedT*>(this));
}

///Return the dialect that must be loaded in the context before this pass.
Void getDependentDialects(::mlir::DialectRegistry&registry) const override{

}

///Explicitly declare the TypeID for this class. We declare an explicit private
///instantiation because Pass classes should only be visible by the current
///library.
MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineLoopInvariantCodeMotionBase<DerivedT>)


};
```
注意：该Pass继承于C++模板类mlir::OperationPass，这说明了MLIR中变换的基础是Operation，所以框架中定义了模板类OperationPass。模板类是MLIR中Pass执行的基础框架部分。

除了生成Pass相关代码，还会生成辅助代码，将Pass注册MLIR框架中。例如：

```Inline void registerAffineLoopInvariantCodeMotion(){
::mlir::registerPass([]()->std::unique_ptr<::mlir::Pass>{
Return mlir::affine::createAffineLoopInvariantCodeMotionPass();
});
}
```
函数mlir::registerPass本质上通过一个全局变量管理所有注册的Pass，该变量是Map结构，其中key为TD文件中的passArg，表示pass的名字；而value为一个结构体，包含了（passArg，Description，Functor）其中passArg和key相同，decription表示pass的描述，而Functor则是registerPass中参数，这个参数是一个函数指针，会调用pass的构造器。
另外mlir-tblgen还是生成辅助代码，将TD文件中所有定义的pass都添加辅助函数，形如register+groupName+Passes()的函数中，例如Affine中所有的pass会有一个对应的辅助函数registerAffinePasses，在registerAffinePasses会调用每一个Pass的注册函数，例如会调用registerAffineLoopInvariantCodeMotion。代码：
```Inline void registerAffinePasses(){
...
registerAffineLoopInvariantCodeMotion();
...
}
```
除了自动生成的代码外，用户需要实现Pass的构造函数。实现过程中通常会定义一个类继承于上述自动生成的类，例如LoopInvariantCodeMotion继承于AffineLoopInvariantCodeMotionBase，并且在Pass的构造函数中实例化对象，对应的代码：

```Struct LoopInvariantCodeMotion
: public affine::impl::AffineLoopInvariantCodeMotionBase<
LoopInvariantCodeMotion>{
Void runOnOperation()override;
...
};

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopInvariantCodeMotionPass(){
Return std::make_unique<LoopInvariantCodeMotion>();
}
```
在自定义的类中最关键的函数为runOnOperation，这个函数在模板类模板类mlir::OperationPass为虚函数（准确说模板类mlir::OperationPass继承于基类Pass，runOnOperation是Pass中的虚函数）

通过mlir-tblgen就可以将Pass定义、注册和MLIR框架结合起来，主要用户实现runOnOperation就可以被MLIR中Pass框架调用执行。

在TD文件中定义通用Pass定义，即适用于任意的Operation。例如CSE表示消除公共子表达式。其Pass定义为：

```Def CSE:Pass<"cse">{
...
}
```
只需要传递为Pass传递第一个参数，Pass的名字，不需要第二个参数。此时生成的代码
```template<typename DerivedT>
Class CSEBase : public ::mlir::OperationPass<>{
public:
...
}
```
而mlir::OperationPass<>是一个特殊的类，等价于参数为void，MLIR社区用这个类匹配任意的Operation，也称为any操作。

为了方便，MLIR还提供了InterfacePass，表示Pass仅仅作用于某一些Interface限定的Operation。如果Operation没有限定Interface，不会执行相关Pass。

当Pass定义过滤条件时，如显式指定Pass适用的Operation或者指定适用的Interface，Pass运行，如果发现过滤条件不满足，会终止对该Operation的处理（即不会运行所有的Pass，并不是跳过不适用的Pass）。

### 5.1.6PassPipeline

为了更好的管理Pass，MLIR社区还提供了PassPipeline、PassManager以及OpPassmanager概念。Pass表示一个Pass，而PassPipeline表示一组确定好顺序的Pass。OpPassManager用于定于嵌套PassManager。一个PassPipeline将通过一个OpPassManager进行管理。

PassManager充当顶级入口点，并包含用于整个传递管道的各种配置。该类OpPassManager用于安排传递在特定嵌套级别运行。顶层PassManager也充当OpPassManager.

```Void pipelineBuilder(OpPassManager& pm){
pm.addPass(std::make_unique<MyPass>());
pm.addPass(std::make_unique<MyOtherPass>());
}
```

类似于Pass，MLIR社区也提供了PassPieline的注册机制，它通过PassPipelineRegistration完成Pipeline的注册，注册后也是通过全局变量进行管理Pipeline对象。
```
Void registerMyPasses(){
//Register an existing pipelinebuilder function.
PassPipelineRegistration<>(
"argument", "description", pipelineBuilder);

//Register an inline pipeline builder.
PassPipelineRegistration<>(
"argument","description",[](OpPassManager &pm){
pm.addPass(std::make_unique<MyPass>());
pm.addPass(std::make_unique<MyOtherPass>());
});
}
```
### 5.1.7Pass执行框架

所有的Pass都继承于OpeartionPass，它有几个特点：
* 针对特定的Operation进行处理；对于一般的Pass来说，如果没有指定Operation则是针对Void进行处理，但是对于PassPipeline来说必须指定一些信息，其中包含Operation，否则不能执行；
* 提供了接口canScheduleOn，用于指定Pass运行于特定的Operation；
* 提供了接口getAnalysis，用于获取分析结果。
用户重写runOnOperation函数，则runOnOperation可以通过框架得到执行。

canScheduleOn这个接口是过滤Pass执行的条件，在Pass定义时，可以看到Pass继承于模板类OperationPass，而OperationPass有参数限定了Pass处理的操作类型。其处理逻辑就在这个函数中，只有待处理的Operation类型和Pass定义中限定的模板参数一直Pass才可能会被执行，否则直接放弃处理Operation。

下面以mlir-opt工具执行为例介绍Pass如何得到执行。
想要使用mlir-opt工具执行Pass，首先需要将Pass注册到mlir-opt工具，然后就可以通过mlir-opt工具执行Pass。例如MLIR框架中提供的mlir-opt工具中会调用registerAffinePasses，表示所有Affine相关的Pass都可以通过mlir-opt工具使用，当用户使用mlir-opt-affine-loop-invariant-code-motion命令触发相关的Pass，执行runOnOperation。下面讨论框架触发Pass的具体过程。


### 5.1.8Pass执行顺序

我们看到在Pass定义的时候通常约定要处理的Operation，这个Operation也称为Pass的锚点，Pass的构造函数会返回模板类OperationPass和对应Operation的实例，例如形如OperationPass<func::FuncOp>的对象，这样的Pass称为特定Pass（op-specific）。也有一些Pass可以处理任意的Operation，例如CSE，它继承于mlir::OperationPass<>，这样的Pass称为通用Pass（op-agnostic），它们的构造函数返回一个基类Pass的实例。

对于锚点Pass来说，Pass只需要处理约定的Opearion，其他无关的Operation无需处理。而Operation在MLIR中具有层级结构，所以MLIR的Pass执行顺序也应该和Operation定义的层次一致，为了更好管理Pass的执行顺序，MLIR提供了PassManager。

官网中给出这样的一个例子，示例代码为：
```Module {
spirv.module "Logical" "GLSL450"{
Func @foo(){
...
}
}
}
```
这个代码片段蕴含IR层级为：
```
`builtin.module`
`spirv.module`
`spirv.func`
```
当使用Pass处理时，所以构造Pass也应该按照这样的层级结构。官网提供了一个示例：
```
Auto pm = PassManager::on<ModuleOp>(ctx);
pm.addPass(std::make_unique<MyModulePass>());
OpPassManager &nestedModulePM = pm.nest<spirv::ModuleOp>();
nestedModulePM.addPass(std::make_unique<MySPIRVModulePass>());
OpPassManager &nestedFunctionPM = nestedModulePM.nest<func::FuncOp>();
nestedFunctionPM.addPass(std::make_unique<MyFunctionPass>());
OpPassManager &nestedAnyPM=nestedModulePM.nestAny();
nestedAnyPM.addPass(createCanonicalizePass());
nestedAnyPM.addPass(createCSEPass());
```
这个代码片段对应的Pass结构为：
```
OpPassManager<ModuleOp>
MyModulePass
OpPassManager<spirv::ModuleOp>
MySPIRVModulePass
OpPassManager<func::FuncOp>
MyFunctionPass
OpPassManager<>
Canonicalizer
CSE
```
这里看到两类OpPassManager，分别是OpPassManager<func::FuncOp>和OpPassManager<>，它们分别使用用于处理操作func::FuncOp和任意操作的PassManager。处理特定操作的PassManager是由锚点Pass确定，任意操作PassManager由任意Pass确定。

注意：默认情况下PassManager要求文件中顶层Operation是builtin.module，如果测试文件中不符合该要求，直接运行会报错can'trun'func.func'passmanageron'builtin.module'op。可以通过在命令行中添加完整的Operation层级，例如--pass-pipeline="builtin.module(func.func(gen-dispatch-regions))"来解决。

PassManager对于Pass的执行顺序的约定如下：
1.对于同一层级中多个相邻的嵌套PassManager在满足一定条件可以进行合并，合并的前提是PassManager中的Pass互不冲突（冲突指的是多个Pass同时针对同一类型Operation进行处理，例如一个特定Pass和一个通用Pass同时针对一个Operation进行处理，则PassManager不能合并它们，比如特定Pass直接限定于Opearation，而通用Pass通过Interface获取Operation，关于Interface在**章介绍）。
对于可以合并的嵌套PassManager，特定Pass在前，通用Pass在后，并且按照Pass处理的Opeartion的名字进行Pass排序。确定好顺序后，

*针对针对这一个Operation执行所有的Pass；
*如果Pass是嵌套PassManger，针对Operation中的Region、Blokc进行遍历，然后针对遍历的每一个Operation寻找对应的PassManager，并执行；
*Pass可以多线程执行，由于执行时可能会依赖分析Pass，所以需要为Operation准备对应的分析Pass即可，当多个并行执行的Pass有一个失败，整体认为并行执行失败。
### 5.1.9Pass插桩机制

MLIR提供了一个可定制的框架，通过类来检测传递执行和分析计算PassInstrumentation。此类提供了PassManager的挂钩来观察各种事件：

runBeforePipeline
该回调在执行传递管道（即传递管理器）之前运行。
runAfterPipeline
无论成功与否，此回调都会在传递管道执行后立即运行。
runBeforePass
此回调在执行传递之前运行。
runAfterPass
此回调在成功执行传递后立即运行。如果这个钩子被执行，则runAfterPassFailed不会。
runAfterPassFailed
此回调在传递执行失败后立即运行。如果这个钩子被执行，则runAfterPass不会。
runBeforeAnalysis
此回调在计算分析之前运行。
如果分析请求另一个分析作为依赖项，则可以从当前/对内部调用依赖项的runBeforeAnalysis/对。runAfterAnalysisrunBeforeAnalysisrunAfterAnalysis
runAfterAnalysis
此回调在计算分析后立即运行。
PassInstrumentation实例可以通过该方法直接向PassManager实例注册addInstrumentation。添加到PassManager的检测以类似堆栈的方式运行，即最后一个执行runBefore*挂钩的检测将是第一个执行相应runAfter*挂钩的检测。类的钩子PassInstrumentation保证以线程安全的方式执行，因此不需要额外的同步。下面是一个示例仪器，用于计算分析的计算次数DominanceInfo：
```
Struct DominanceCounterInstrumentation : public PassInstrumentation{
///The cumulative count of how many times dominance has been calculated.
Unsigned &count;

DominanceCounterInstrumentation(unsigned &count):count(count){}
Void runAfterAnalysis(llvm::StringRef,TypeID id,Operation*)override{
if( id == TypeID::get<DominanceInfo>())
++count;
}
};

MLIRContext *ctx = ...;
PassManager pm(ctx);

//Add the instrumentation to the passmanager.
Unsigned domInfoCount;
pm.addInstrumentation(
std::make_unique<DominanceCounterInstrumentation>(domInfoCount));

//Run the passmanager on a module operation.
ModuleOp m=...;
if(failed(pm.run(m)))
...

llvm::errs() << "DominanceInfo was computed" << domInfoCount << "times!\n";

```
### 5.1.10Pass实现的约束

MLIR的Pass执行时针对Operation进行处理，同时MLIR的Pass框架在设计之初就确定了可以多线程执行，所以Pass实现需要遵守一定的规则，主要限制包括：

不得检查当前操作的同级操作的状态，不得访问嵌套在这些同级下的操作。因为其他线程可能正在并行修改这些操作。但是可以允许检查祖先/父操作的状态。
不得修改除当前操作下嵌套的操作以外的操作的状态。这包括从祖先/父块添加、修改或删除其他操作。同样是因为其他线程可能同时对这些操作进行操作。作为例外，当前操作的属性可以自由修改。这是修改当前操作的唯一方法。（即不允许修改操作数等。）
不得在runOnOperation的调用之间维护可变的Pass状态。因为Pass可以在许多不同的操作上运行，但执行时没有严格的执行顺序保证。当多线程处理时，特定的pass实例甚至可能不会在IR内的所有操作上执行。因此，一次传递不应依赖于在所有操作上运行。
不得维护任何全局可变状态，例如源文件中的静态变量。所有可变状态都应该由Pass的实例来维护。
必须是可复制构造的，PassManager可以创建Pass的多个实例以并行处理操作。

此操作类型必须符合以下要求：必须被注册且被标记为IsolatedFromAbove

### 5.1.11标准插桩

在PassManager中支持对Pass进行插桩，插桩通过类PassInstrumentation进行定义，它定义了一系列API，主要包括：
runBeforePipeline：该回调在执行传递管道（即传递管理器）之前运行。
runAfterPipeline：无论成功与否，此回调都会在传递管道执行后立即运行。
runBeforePass：此回调在执行传递之前运行。
runAfterPass：此回调在成功执行传递后立即运行。如果这个钩子被执行，则runAfterPassFailed不会。
runAfterPassFailed：此回调在传递执行失败后立即运行。如果这个钩子被执行，则runAfterPass不会。
runBeforeAnalysis：此回调在计算分析之前运行。如果分析请求另一个分析作为依赖项，则可以从当前runBeforeAnalysis/runAfterAnalysis对依赖runBeforeAnalysis/runAfterAnalysis对进行调用。
runAfterAnalysis：此回调在计算分析后立即运行。

这些API会被整合到Pass的执行过程。例如PassManager包含一个Pass的场景，首先PassManager的顶层是PassPipeline，当包含了Pass插桩时其执行过程如下所示：


而插桩中Analysis/runAfterAnalysis在Pass执行过程中获取分析结果时会被执行。

当用户实现自己的Pass插桩后，通过PassManager的addInstrumentation接口就可以把插桩注册到PassManager中，并在相应的调用点执行插桩中的回调函数。

MLIR社区提供了3个基于Pass插桩的有用功能，时间统计、IR打印、Pass失败捕获。

#### 1.时间统计

用户在自定义PassManager时可以通过PassManager中的enableTiming打开Pass执行时间统计信息，而默认的mlir-opt工具可以通过参数-mlir-timing使用该功能。时间统计实现原理是基于Pass插桩的能力，在runBefore*中记录起始时间，在runAfter*中获取结束时间，从而在PassManager运行结束后可以打印Pass的统计信息。
Pass执行信息在串行执行和并行执行输出有所不同，读者可以参考官网了解时间统计的具体格式和含义。

#### 2.IR打印

IR也是利用Pass插桩功能，实现了runBeforePass、runAfterPass、runAfterPassFailed截获执行的Operations，从而实现IR打印。基于这个Pass插桩，MLIR实现了和LLVM一样的IR输出。为了便于读者只关注关心的IR，MLIR社区还提供了一系列的参数控制IR打印的范围。例如：
mlir-print-ir-before：设置关注的Pass，在Pass运行之前打印IR。
mlir-print-ir-before-all：在每个Pass运行之前都打印IR。
mlir-print-ir-after：设置关注的Pass，在Pass运行之后打印IR。
mlir-print-ir-after-all：在每个Pass运行之后都打印IR。
mlir-print-ir-after-change：如果Pass改变了IR则在Pass执行后打印IR。该选项需要和mlir-print-ir-after或者mlir-print-ir-after-all配合使用。
mlir-print-ir-after-failure：在Pass执行失败后打印IR。
mlir-print-ir-module-scope：打印当前操作的顶层操作全部打印出来，该参数需要禁止Pass并发执行（需设置mlir-disable-threading）。

### 5.1.12Pass执行失败捕获与重放机制

Pass在执行过程中可能发生错误，在复杂的系统中通常输入包含了许多Operation以及多种Pass的组合，在编译执行过程中可能遇到Pass失败的场景，而这时要定位到哪个Pass在对哪个Operation处理时发生错误就非常困难。所以MLIR提供了Pass失败捕获机制以及重放机制。

失败捕获机制的实现原理比较简单，是基于Pass的插桩机制实现的。在Pass插桩中有一个接口runAfterPassFailed，当Pass执行失败可以调用一个回调函数。而失败捕获机制就是利用这个API实现Pass执行失败的信息，并将其记录下来。为了准确记录Pass执行失败的信息，还需要记录Pass执行的上下文信息，所以失败捕获机制还会实现Pass插桩中的runBeforePass、runAfterPass函数，在runBeforePass中会记录相关上下文信息，主要包括要执行的Pass以及对应的Operation；当Pass成功运行时runAfterPass会删除上下文信息。
当Pass执行失败时，会将Operation以及执行的PassPipeline执行记录下来。在Pass执行过程可以传递不同的参数mlir-pass-pipeline-crash-reproducer和mlir-pass-pipeline-local-reproducer用于记录PassPipeline信息，分别记录Pass执行失败时Operation对应的完整PassPipeline、仅失败Pass。参数mlir-pass-pipeline-local-reproducer仅仅记录Pass执行失败时最新的PassPipeline信息，所以该选项要求Pass执行不能并行执行（可以通过参数mlir-disable-threading设置），因为并行执行时最新记录的Pass上下文信息可能和失败Pass信息并不相同。而参数mlir-pass-pipeline-crash-reproducer可以支持Pass并发执行。

例如Pass执行失败格式示例如下所示，其中Pass失败捕获参数为mlir-pass-pipeline-crash-reproducer。
```
func.func @foo(){
%0=arith.constant 0:i32
return
}

func.func @bar(){
return
}

{-#
external_resources:{
mlir_reproducer:{
verify_each:true,
pipeline:"builtin.module(func.func(cse, canonicalize{max-iterations=1 max-num-rewrites=-1 region-simplify=false top-down=false}))",
disable_threading:true
}
}
#-}
```
同时MLIR社区还提供了重放机制，例如在mlir-opt工具中通过参数-run-reprodcuer可以重新运行指定的操作和PassPipeline。这个功能的实现也比较简单，从mlir-reprodcuer中获取PassPipeline等信息，然后针对相应的Operation执行Pass即可。
## 5.2分析Pass
与转换过程一样，一个重要的概念是分析。这些在概念上类似于转换过程，只不过它们计算特定操作的信息而不修改它。在MLIR中，分析不是过程，而是独立的类，它们按需延迟计算并缓存以避免不必要的重新计算。也就说是MLIR中的分析需要先定义一个类，用于描述分析过程和分析结果，在MLIR显示的生成分析对象以及调用分析过程。为了使用方便，MLIR引入了AnalysisManager，它仅仅管理分析对象。
MLIR中的分析必须遵守以下规定：
提供一个Operation*或者Operation*和AnalysisManager&为参数的构造函数，其中参数AnalysisManager用于查询分析依赖；
不得对操作进行修改。

分析可能会提供额外的钩子来控制各种行为：bool isInvalidated(const AnalysisManager::PreservedAnalyses &)

给定一个保留的分析集，如果它确实应该失效，则分析将返回true。这允许在分析未明确标记为保留的情况处理何时失效情况，例如可以根据其他属性或其他分析集的结果对当前分析集设置保存或失效。如果一个分析依赖另外一个分析，它必须检查依赖的分析是否无效。

查询分析：基类OperationPass提供了针对当前操作查询和保存分析结果的API，
getAnalysis<>：对当前操作进行分析，在必要时构建它，通常在构建分析对象时会进行分析。
getCachedAnalysis<>：获取当前操作的分析（如果已存在）。
getCachedParentAnalysis<>：获取给定父操作的分析（如果存在）。
getCachedChildAnalysis<>：获取给定子操作的分析（如果存在）。
getChildAnalysis<>：获取给定子操作的分析，在必要时构建它。
markAllAnalysesPreserved
markAnalysesPreserved<>

保留分析：通过查询后构建的分析会被缓存，以避免稍后再次请求时不必要的计算。为了避免过时的分析，所有分析都被假定为通过一次而无效。为了避免无效，传递必须专门标记已知要保留的分析。提供的API有：
markAllAnalysesPreserved：保存所有的分析结果。
markAnalysesPreserved<>：保存指定类的分析结果。

<!-- more -->
