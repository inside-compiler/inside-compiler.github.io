---
title: Fast指令选择算法介绍
categories:
  - 深入理解LLVM代码生成
tags:
  - LLVM 
date: 2024-03-07 19:17:08
---
SelectionDAGISel算法经过了LLVM IR的DAG化、合法化、匹配表查找等复杂过程，会耗费大量时间。为了提高指令选择的速度，LLVM实现了一个快速指令选择算法（FastISel），这一算法只适用于部分后端的O0阶段，是以牺牲指令选择的质量来换取编译时间。

<!-- more -->

FastISel的原则是：尽可能快速地选择尽可能多的指令。也就意味着，FastISel允许指令选择失败，当发生失败就会进入到SelectionDAGIsel指令选择流程中继续选择，这也使得FastISel可以复用一些SelectionDAGIsel中的逻辑，避免重复实现。
在7.2节提到在指令选择过程中还会涉及一些复杂的工作，例如合法化、优化等，FastIsel并不会处理，它只对数据类型合法的简单操作进行处理，比如常规的加减运算、位运算，其它的数据类型和指令操作，FastIsel都认为是失败场景，都会被自动切换到SelectionDAGIsel中进行。
FastIsel也使用TableGen工具链，将TD文件中的指令描述直接翻译为一个或者几个函数调用。例如将TD中定义的一个ADD指令直接翻译成MIR指令序列。在使用FastIsel时直接基于LLVM IR进行指令匹配，生成MIR。例如Aarch64架构中ADDXrr指令在TD文件定义如代码清单7-12。
Aarch64架构中对ADDXrr指令的TD描述

```multiclass AddSub<bit isSub, string mnemonic, string alias,
                  SDPatternOperator OpNode = null_frag> {
  let hasSideEffects = 0, isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  ...
  def Xrr : BaseAddSubRegPseudo<GPR64, OpNode>;

...
}
defm ADD : AddSub<0, "add", "sub", add>;
```

经过TableGen工具链的第一阶段处理后（参考第6章），得到的记录如代码清单7-13。
ADDXrr记录
```def ADDXrr {	
  string Namespace = "Aarch64";
  dag OutOperandList = (outs GPR64:$Rd);
  dag InOperandList = (ins GPR64:$Rn, GPR64:$Rm);
  ...
//ADDXrr指令对应的匹配模板，两个入参均为64位寄存器类型，返回值也为64位类型
  list<dag> Pattern = [(set GPR64:$Rd, (add GPR64:$Rn, GPR64:$Rm))];
...
```
再经过TableGen工具继续对于记录进行提取，在FastIsel中仍然根据Pattern字段提取匹配信息，最后会生成如代码清单7-14的函数。
ADDXrr模板校验函数实现
```unsigned fastEmit_ISD_ADD_MVT_i64_rr(MVT RetVT, unsigned Op0, unsigned Op1) {
  if (RetVT.SimpleTy != MVT::i64)
    return 0;
  return fastEmitInst_rr(Aarch64::ADDXrr, &Aarch64::GPR64RegClass, Op0, Op1);
}
```
可以看到代码7-14中只对指令的返回值进行了校验，紧接着就指定要生成的MIR指令及使用的寄存器类型，但没有对操作数类型进行判断，这是因为对操作数是否为寄存器类型的判断，可以作为多条指令的共性处理，所以在框架中定义了函数fastEmitInst_rr，然后要求每个后端都实现这一函数，由此也可以看出FastIsel需要更多框架代码的配合，否则很多指令无法映射为MIR。例如Aarch64中fastEmitInst_rr实现如代码清单7-15所示。
fastEmitInst_rr函数的实现
```Register FastISel::fastEmitInst_rr(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC, unsigned Op0,
                                   unsigned Op1) {
  const MCInstrDesc &II = TII.get(MachineInstOpcode);
  ...
  return ResultReg;
}
```
目前，并不是所有的架构都支持FastISel这种指令选择模式，除了上述提到的Aarch64外，当前支持该模式的架构还有Arm、Mips、PPC、X86、WebAssembly。
