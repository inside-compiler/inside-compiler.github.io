---
title: ADT-算法
categories:
  - LLVM
tags:
  - [LLVM, MLIR, ADT]
date: 2024-03-16 18:41:50
---

ADT提供了一些算法，主要用于补充std缺失部分，用于简化开发，最为典型的算法是zip和enumerate

# zip

```std::vector<int> v1 = {0, 4, 2, 1};
  std::vector<int> v2 = {1, 4, 3, 6};
  EXPECT_TRUE(all_of_zip(v1, v2, [](int v1, int v2) { return v1 <= v2; }));
  EXPECT_FALSE(all_of_zip(v1, v2, [](int L, int R) { return L < R; }));
```

# enumerate

遍历容器的元素，

``` std::vector<char> foo = {'a', 'b', 'c'};

  for (auto X : llvm::enumerate(foo)) {
    ++X.value();
  }
  EXPECT_THAT(foo, ElementsAre('b', 'c', 'd'));
  
  
  std::vector<PairType> Results;

  for (auto X : llvm::enumerate(std::vector<int>{1, 2, 3})) {
    Results.emplace_back(X.index(), X.value());
  }
```

<!-- more -->
