hero: 摘要
path: docs
source: Material Theme Tutorial.md

# Material Theme Tutorial

[更加完整的Tutorial](<https://cyent.github.io/markdown-with-mkdocs-material/>)

## 警告

### 用法
警告块遵循一个简单的语法:每个块都以`!!!`，然后使用一个关键字作为块的类型限定符。然后，块的内容跟随到下一行，由四个空格缩进。

Example:

!!! note

    Text
    
    Text


#### 改变标题

Example:

!!! note "title"

   	Text
   	
    Text

#### 删除标题

Example:

!!! note ""

   	Text
   	
    Text

#### 折叠块

Example:

??? note "title"

   	Text	

    Text

### 类型

- `note/seealso`

!!! note "note"

   	Text
   	
    Text

- `abstract/summary/tldr`

!!! summary "summary"

   	Text
   	
    Text

- `info/todo`

!!! info "info"

   	Text
   	
    Text

- `tip/hint/important`

!!! tip "tip"

   	Text
   	
    Text

- `success/check/done`

!!! done "done"

   	Text
   	
    Text

- `question/help/faq`

!!! help "help"

   	Text
   	
    Text

- `failure/fail/missing`

!!! fail "fail"

   	Text
   	
    Text

- `danger/error`

!!! error "error"

   	Text
   	
    Text

- `bug`

!!! bug "bug"

   	Text
   	
    Text

- `example/snippet`

!!! example "example"

   	Text
   	
    Text

- `quote/cite`

!!! cite "cite"

   	Text
   	

    Text

## CodeHilite

### 用法

#### `markdown`用法

``` python
import tensorflow as tf
```
#### `Shebang`用法

``` python
#!/usr/bin/python
import tensorflow as tf
```
#### 三个冒号

    :::python
    import tensorflow as tf

#### 多选项卡的代码块

``` bash tab="Bash"
#!/bin/bash

echo "Hello world!"
```

``` c tab="C"
#include <stdio.h>

int main(void) {
  printf("Hello world!\n");
}
```

``` c++ tab="C++"
#include <iostream>

int main() {
  std::cout << "Hello world!" << std::endl;
  return 0;
}
```

``` c# tab="C#"
using System;

class Program {
  static void Main(string[] args) {
    Console.WriteLine("Hello world!");
  }
}
```

#### 特定行高亮

    #!python hl_lines="3 4"
        """ Bubble sort """
        def bubble_sort(items):
            for i in range(len(items)):
                for j in range(len(items) - 1 - i):
                    if items[j] > items[j + 1]:
                        items[j], items[j + 1] = items[j + 1], items[j]

## 脚注

单行 [^1] 多行[^2]

[^1]: 单行脚注
[^2]: 多行脚注

      多行脚注
      
      多行脚注      

## 元数据

- 设置hero文本
```
hero: 支持摘要文本
```
- 给出下载链接

```
path: docs
source: Material Theme Tutorial.md
```
- 给出跳转链接

```
redirect: /new/url
```

### 覆盖

- 页面标题
```
title: 页面标题
```
- 页面描述
```
description: 页面描述
```

## Pymdown

#### 块

``` markdown
$$

$$
```

$$
\frac{n!}{k!(n-k)!} = \binom{n}{k}
$$


#### 内联
`$ $`
Lorem ipsum dolor sit amet: $p(x|y) = \frac{p(y|x)p(x)}{p(y)}$

#### 粗体斜体

**AA**  `****`
*BB*  `**`

#### 插入符

 ^^inserted text^^ 

<u>下划线</u>	
 `^^...^^` `<u></u>`

#### 评论

- 删除 添加 高亮 

{--deleted--}  {++added++} {==Highlighting==}

{~~one~>a single~~} 

- 内联注释

{>>and comments can be added inline<<}.

- 块高亮

{==

Formatting can also be applied to blocks, by putting the opening and closing
tags on separate lines and adding new lines between the tags and the content.

==}

#### 表情
`:shit: :smile: :heart: :thumbsup:`
:shit: :smile: :heart: :thumbsup:

#### 内联代码高亮
- `使用 #!js`
```markdown
`#!js var test = 0;`
```
`#!js var test = 0;`

#### 突出
`==...==`
==AAA==

#### 智能符号
- 箭头 (<--, -->, <-->)
- 商标和版权符号 ((c), (tm), (r)) 
- 分数 (1/2, 1/4, ...).

#### 任务列表

- [x] Curabitur elit nibh, euismod et ullamcorper at, iaculis feugiat est
- [ ] Vestibulum convallis sit amet nisi a tincidunt
    - [x] In hac habitasse platea dictumst

#### 波浪线
`~~...~~`
~~AAA~~