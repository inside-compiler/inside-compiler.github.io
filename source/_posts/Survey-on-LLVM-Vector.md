---
title: Survey-on-LLVM-Vector
categories:
  - null
tags:
  - null
date: 2023-12-30 12:24:41
---

<!-- more -->

## 用例
```
#include <arm_neon.h>
uint32x4_t test(uint32x4_t n) {

  return n;
}
```
```
#include <arm_neon.h>
uint32x4_t test(uint32x4_t a, uint32x4_t b) {

  return a + b;
}
```

## 代码

```
typedef __attribute__((neon_vector_type(4))) uint32_t uint32x4_t;
```

```
AST:
  ASTContext.h
  Type.h
  TypeProperties.td
 
Basic:
  RISCVVTypes.def
  TargetInfo.h
  RISCV.h
  
  
Serialization:
  ASTBitCodes.h
  ASTReader.cpp
  
CodeGen:
  CGDebugInfo.cpp
  CodeGenTypes.cpp
  ItaniumCXXABI.cpp
  
Index:
  USRGeneration.cpp
  
Sema:
  Sema.cpp 
  SemaExpr.cpp
  
module.modulemap

test:
  riscv-types.c  
```

```
InitBuiltinTypes
getTypeInfoImpl
getBuiltinVectorTypeInfo
getScalableVectorType
getObjCEncodingForPrimitiveType

VisitBuiltinType
EvaluateBuiltinClassifyType
```

```
clang  -ast-print
```

