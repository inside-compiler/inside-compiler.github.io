---
title: Build LLVM
date: 2023-12-14 23:11:23
categories:
  - 项目编译
tags:
  - 编译配置
---

## download code
~~~sh
git clone git@github.com:chenghanpeng/llvm-project.git
cd llvm-project
git switch release/15.x
~~~
<!-- more -->
## Linux
~~~sh
$ touch build.sh
~~~
build.sh 内容如下：

~~~sh
#!/bin/bash

mkdir -p build
cd build

if [ ! -f bin/llvm-lto ]; then
  CORE_NUM=$(nproc --all)
  if [ -z "${CORE_NUM}" ]; then
    CORE_NUM=1
  fi

  echo "Build llvm with" ${CORE_NUM} "cores"

  cmake /mnt/Compiler/llvm-project/llvm \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE="Debug" \
    -DLLVM_TARGETS_TO_BUILD="X86;AArch64;BPF;Hexagon" \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi"

  cmake --build .
else
  echo "llvm has already been built"
fi
~~~

~~~sh
$ chmod +x build.sh
$ ./build.sh
~~~

## MacOs
构建LLVM依赖git和cmake，可使用brew安装git和cmake
### 安装依赖
~~~sh
$ which brew
brew not found

# 安装brew
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
$ which brew

#安装git
$ brew install git
$ git --version

#安装cmake
$ brew install cmake
$ which cmake
~~~

### 构建LLVM
~~~sh
$ cd llvm-project
$ mkdir build && cd build
$ cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug DLLVM_TARGETS_TO_BUILD="X86;AArch64;BPF;Hexagon" -DLLVM_ENABLE_PROJECTS="clang" ../llvm
$ make -j32
$ cd bin & ls
$ clang -v
clang version 15.0.6 ...
~~~

### macOS上使用自己源码构建的clang编译源文件
~~~sh
# 在macOS上可能会报错 fatal error: 'stdio.h' file not found
$ clang -emit-llvm -S -isysroot `xcrun --show-sdk-path` hello.c -o hello.ll
# 使用debug版本llc打印log
$ llc hello.ll --march=bpf -o hello.s -debug >hello.log 2>&1
~~~



