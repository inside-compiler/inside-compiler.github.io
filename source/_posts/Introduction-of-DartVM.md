---
title: Introduction_of_DartVM
categories:
  - virtual machine
tags:
  - DartVM
date: 2024-07-02 20:37:23
---

Dart是Google在2011年10月10号发布的一种用于客户端（web或者移动应用）开发的编程语言，它是为了解决Javascript语言相关问题而设计。它有个基于VM思想实现的运行环境，即DartVM。本文通过阅读DartVM的早期代码和相关文章，了解DartVM的设计思路和整体框架结构。

<!-- more -->

## Dart语言简介

Dart是一种面向对象的语言，语法类似于C语言的语法，有C等编程语言基础的比较容易入门。

它基本类型如下。

```dart
int、double、String、bool、List、Map、Set
```

```
int x = 2;
bool flag = true;
String str = "Dart";

List list = new List(3);
list[0] = 'dart';

Map map = new Map();
map['F'] = 'Dart';
```

Dart可以声明动态类型的变量，具体关键字如下。

```
var: 需要类型推导的变量；
final：编译时确定类型；
const：运行时确定类型；
```

Dart语言的一些特性：

1. 可以通过“？”保证null安全，即变量没有标记“？”就是默认不为空的。

2. Dart里的Function也是一种Object，它有类型。
3. class里可以给属性定义Setter和Getter函数，用于获取和设置属性值。
4. 具有异常处理机制。
5. 并发机制。

## DartVM整体框架

DartVM早期代码量如下，总体上有71kLoc。

![image-20240705114904916](./Introduction-of-DartVM/image-20240705114904916.png)

## 代码流程

主流程里有

1. 参数解析
2. DartVM初始化
3. isolate创建
4. 进入scope（独立的handle管理和内存分配）
5. dart文件编译
6. dart库链接
7. dart文件执行（从main开始）
8. 执行结果处理
9. 退出scope
10. 消息循环执行
11. profiler符号记录
12. isolate关闭

```c++
int main(int argc, char** argv) {
  ParseArguments(argc, argv, &vm_options, &script_name, &dart_options);
  Dart_Initialize(vm_options.count(), vm_options.arguments(), MainIsolateInitCallback);
  Dart_Isolate isolate = Dart_CreateIsolate(snapshot_buffer, script_name);
  
  Dart_EnterScope();
  Dart_CompileAll();
  Dart_Handle script_url = Dart_NewString(script_name);
  Dart_Result result = Dart_LookupLibrary(script_url);
  Dart_Handle result_obj = Dart_GetResult(result);
  Dart_ExitScope();
  
  Dart_RunLoop();
  DumpPprofSymbolInfo();
  Dart_ShutdownIsolate();
}
```

dart文件加载。

```c++
static void* MainIsolateInitCallback(void* data) {
  result = LoadScript(script_name);
  Dart_Handle library = Dart_GetResult(result);
}
```

```c++
// runtime/bin/process_script.cc
Dart_Result LoadScript(const char* script_name) {
  Dart_Result result = ReadStringFromFile(script_name);
  Dart_Handle source = Dart_GetResult(result);
  Dart_Handle url = Dart_NewString(script_name);

  return Dart_LoadScript(url, source, LibraryTagHandler);
}
```

将代码以字符串的形式读到内存中。

```c++
static Dart_Result ReadStringFromFile(const char* filename) {
  File* file = File::OpenFile(filename, false);
  intptr_t len = file->Length();
  char* text_buffer = reinterpret_cast<char*>(malloc(len + 1));
  file->ReadFully(text_buffer, len);
  Dart_Handle str = Dart_NewString(text_buffer);
  return Dart_ResultAsObject(str);
}
```

生成Library

```c++
DART_EXPORT Dart_Result Dart_LoadScript(Dart_Handle url,
                                        Dart_Handle source,
                                        Dart_LibraryTagHandler handler) {
  isolate->set_library_tag_handler(handler);
  library = Library::New(url_str);
  library.Register();
  Dart_Result result;
  CompileSource(library, url_str, source_str, RawScript::kScript, &result);
  return result;  
}                                        
```

```c++
// runtime/vm/parser.cc
static void CompileSource(const Library& lib,
                          const String& url,
                          const String& source,
                          RawScript::Kind kind,
                          Dart_Result* result) {
  Compiler::Compile(lib, script);
}  
```

```c++
void Compiler::Compile(const Library& library, const Script& script) {
  const String& library_key = String::Handle(library.private_key());
  script.Tokenize(library_key);
  Parser::ParseCompilationUnit(library, script);  
}
```

前端解析

```c++
// runtime/vm/parser.cc
void Parser::ParseCompilationUnit(const Library& library,
                                  const Script& script) {
  Parser parser(script, library);
  parser.ParseTopLevel();
}
```

```c++
void Parser::ParseTopLevel() {
  while (true) {
    set_current_class(Class::Handle());  // No current class.
    if (CurrentToken() == Token::kCLASS) {
      ParseClassDefinition(&classes);
    } else if (CurrentToken() == Token::kTYPEDEF) {
      ParseFunctionTypeAlias(&classes);
    } else if (CurrentToken() == Token::kINTERFACE) {
      ParseInterfaceDefinition(&classes);
    } else if (IsVariableDeclaration()) {
      set_current_class(toplevel_class);
      ParseTopLevelVariable(&top_level);
    } else if (IsTopLevelFunction()) {
      set_current_class(toplevel_class);
      ParseTopLevelFunction(&top_level);
    } else if (IsTopLevelAccessor()) {
      set_current_class(toplevel_class);
      ParseTopLevelAccessor(&top_level);
    } else if (CurrentToken() == Token::kEOS) {
      break;
    } else {
      UnexpectedToken();
    }
  }
}
```

编译Dart代码，依次遍历每个库文件，在每个库文件里依次编译class中的函数和匿名class中的函数。

```c++
// runtime/vm/dart_api_impl.cc
DART_EXPORT Dart_Result Dart_CompileAll() {
  CompileAll(&result);
}
```

```c++
// runtime/vm/object.cc
void Library::CompileAll() {
  Library& lib = Library::Handle(
      Isolate::Current()->object_store()->registered_libraries());
  Class& cls = Class::Handle();
  while (!lib.IsNull()) {
    ClassDictionaryIterator it(lib);
    while (it.HasNext()) {
      Compiler::CompileAllFunctions(cls);
    }
    
    Array& anon_classes = Array::Handle(lib.raw_ptr()->anonymous_classes_);
    for (int i = 0; i < lib.raw_ptr()->num_anonymous_; i++) {
      Compiler::CompileAllFunctions(cls);
    }
    lib = lib.next_registered();
  }
}
```

```c++
// runtime/vm/compiler.cc
void Compiler::CompileAllFunctions(const Class& cls) {
  Array& functions = Array::Handle(cls.functions());
  Function& func = Function::Handle();
  for (int i = 0; i < functions.Length(); i++) {
    CompileFunction(func);
  }
}

void Compiler::CompileFunction(const Function& function) {
  CompileFunctionHelper(function, false);
}
```

有三种编译过程：

1. 首次编译，直接会编译成未优化代码；

2. 在有未优化代码的情况下，编译成优化代码；

   这个场景下会根据退优化的次数判断是否使用类型反馈信息。

3. 将优化代码切换成未优化代码。

编译完成后，函数指针会被添加到CodeIndexTable中。

```c++
static void CompileFunctionHelper(const Function& function, bool optimized) {
  ParsedFunction parsed_function(function);
  Parser::ParseFunction(&parsed_function);
  CodeIndexTable* code_index_table = Isolate::Current()->code_index_table();
  Assembler assembler;
  
  if (optimized) {
    ExtractTypeFeedback(Code::Handle(parsed_function.function().code()),  
                        parsed_function.node_sequence());  
    OptimizingCodeGenerator code_gen(&assembler, parsed_function);
    code_gen.GenerateCode();
    function.SetCode(code);
    code_index_table->AddFunction(function);
  } else {
    // Unoptimized code.
    if (Code::Handle(function.unoptimized_code()).IsNull()) {
      ASSERT(Code::Handle(function.code()).IsNull());
      // Compiling first time.
      CodeGenerator code_gen(&assembler, parsed_function);
      code_gen.GenerateCode();
      function.SetCode(code);
      code_index_table->AddFunction(function);
    } else {
      // Disable optimized code.
      const Code& optimized_code = Code::Handle(function.code());
      ASSERT(optimized_code.is_optimized());
      CodePatcher::PatchEntry(Code::Handle(function.code()));
      // Use previously compiled code.
      function.SetCode(Code::Handle(function.unoptimized_code()));
      CodePatcher::RestoreEntry(Code::Handle(function.unoptimized_code()));
    }
  }
}
```

不优化的CodeGen--CodeGenerator。

通过宏隔离区分不同Target的实现。

```
TARGET_ARCH_IA32
TARGET_ARCH_ARM
TARGET_ARCH_X64
```

```c++
// NOTE: First 5 bytes of the code may be patched with a jump instruction. Do
// not emit any objects in the first 5 bytes.
void CodeGenerator::GenerateCode() {
  GeneratePreEntryCode();
  GenerateEntryCode();
  parsed_function_.node_sequence()->Visit(this);
  GenerateDeferredCode();
  pc_descriptors_list_->AddDescriptor(PcDescriptors::kPatchCode,
                                      assembler_->CodeSize(),
                                      AstNode::kInvalidId,
                                      0,
                                      -1);
  __ jmp(&StubCode::FixCallersTargetLabel());  
}
```

函数进入前的栈溢出检查和调用次数上报（用于后续的优化等级提升），此外，当允许优化的时候跳到优化代码上去。

```c++
// Pre entry code is called before the frame has been constructed:
// - check for stack overflow.
// - optionally count function invocations.
// - optionally trigger optimizing compiler if invocation threshold has been
//   reached.
// Note that first 5 bytes may be patched with a jump.
// TODO(srdjan): Add check that no object is inlined in the first
// 5 bytes (length of a jump instruction).
void CodeGenerator::GeneratePreEntryCode() {
  if (may_optimize) {
    __ cmpl(EBX, Immediate(FLAG_optimization_invocation_threshold));
    __ j(GREATER, &StubCode::OptimizeInvokedFunctionLabel());
  }
}
```

栈帧创建和参数处理。

```c++
// Call to generate entry code:
// - compute frame size and setup frame.
// - allocate local variables on stack.
// - optionally check if number of arguments match.
// - initialize all non-argument locals to null.
//
// Input parameters:
//   ESP : points to return address.
//   ESP + 4 : address of last argument (arg n-1).
//   ESP + 4*n : address of first argument (arg 0).
//   EDX : arguments descriptor array.
void CodeGenerator::GenerateEntryCode() {

}
```

遍历节点进行指令生成。

```c++
#define DEFINE_VISIT_FUNCTION(type, name)                                      \
  void type::Visit(AstNodeVisitor* visitor) {                                  \
    visitor->Visit##type(this);                                                \
  }
NODE_LIST(DEFINE_VISIT_FUNCTION)
#undef DEFINE_VISIT_FUNCTION
```

```c++
// runtime/vm/code_generator_ia32.cc
void CodeGenerator::VisitSequenceNode(SequenceNode* node_sequence) {
}

void CodeGenerator::VisitLoadLocalNode(LoadLocalNode* node) {
  // Load the value of the local variable and push it onto the expression stack.
  if (IsResultNeeded(node)) {
    GeneratePushVariable(node->local(), EAX);
  }
}

oid CodeGenerator::GeneratePushVariable(const LocalVariable& variable,
                                         Register scratch) {
  if (variable.is_captured()) {
    // The variable lives in the context.
    int delta = state()->context_level() - variable.owner()->context_level();
    ASSERT(delta >= 0);
    Register base = CTX;
    while (delta-- > 0) {
      __ movl(scratch, FieldAddress(base, Context::parent_offset()));
      base = scratch;
    }
    __ pushl(FieldAddress(base, Context::variable_offset(variable.index())));
  } else {
    // The variable lives in the current stack frame.
    __ pushl(Address(EBP, variable.index() * kWordSize));
  }
}
```

优化代码生成器--OptimizingCodeGenerator

继承自CodeGenerator，在一些Node实现上进行扩展，生成更高效的代码。

```
void OptimizingCodeGenerator::VisitLoadLocalNode(LoadLocalNode* node) {
  if (!IsResultNeeded(node)) return;
  if (IsResultInEaxRequested(node)) {
    GenerateLoadVariable(EAX, node->local());
    node->info()->set_result_returned_in_eax(true);
  } else {
    GeneratePushVariable(node->local(), EAX);
  }
}
```

在优化生成器中会有退优化的操作。首先是生成检查点。

```
```



然后在生成退优化代码，即DeferredCode。

```c++
void OptimizingCodeGenerator::GenerateDeferredCode() {
  CodeGenerator::GenerateDeferredCode();
  for (int i = 0; i < deoptimization_blobs_.length(); i++) {
    deoptimization_blobs_[i]->Generate(this);
  }
}

// Code that calls the deoptimizer, emitted as deferred code (out of line).
// Specify the corresponding 'node' and the registers that need to
// be pushed for the deoptimization point in unoptimized code.
class DeoptimizationBlob : public ZoneAllocated {
  void Generate(OptimizingCodeGenerator* codegen) {
    codegen->assembler()->Bind(&label_);
    for (int i = 0; i < registers_.length(); i++) {
      codegen->assembler()->pushl(registers_[i]);
    }
    codegen->CallDeoptimize(node_->id(), node_->token_index());
  }  
};

void OptimizingCodeGenerator::CallDeoptimize(intptr_t node_id,
                                             intptr_t token_index) {
  __ call(&StubCode::DeoptimizeLabel());
  AddCurrentDescriptor(PcDescriptors::kOther, node_id, token_index);
}
```









## 参考

1. https://medium.com/@author2000.1225/the-history-and-rules-of-dart-language-f25e09a58530

