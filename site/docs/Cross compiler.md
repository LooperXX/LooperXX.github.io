# Cross Compiler

目的： 了解交叉编译的概念，学会使用交叉编译工具 `toolchain`

## 简介

### 什么是交叉编译

##### 本地编译

本地编译可以理解为，在当前编译平台下，编译出来的程序只能放到当前平台下运行。平时我们常见的软件开发，都是属于本地编译。比如，我们在 x86 平台上，编写程序并编译成可执行程序。这种方式下，我们使用 x86 平台上的工具，开发针对 x86 平台本身的可执行程序，这个编译过程称为本地编译。

##### 交叉编译

交叉编译可以理解为，在当前编译平台下，编译出来的程序能运行在体系结构不同的另一种目标平台上，但是编译平台本身却不能运行该程序。比如，我们在 x86 平台上，编写程序并编译成能运行在 ARM 平台的程序，编译得到的程序在 x86 平台上是不能运行的，必须放到 ARM 平台上才能运行。

### 为什么要交叉编译

-   **Speed：** 目标平台的运行速度往往比主机慢得多，许多专用的嵌入式硬件被设计为低成本和低功耗，没有太高的性能
-   **Capability：** 整个编译过程是非常消耗资源的，嵌入式系统往往没有足够的内存或磁盘空间
-   **Availability：** 即使目标平台资源很充足，可以本地编译，但是第一个在目标平台上运行的本地编译器总需要通过交叉编译获得。
-   **Flexibility：** 一个完整的Linux编译环境需要很多支持包，交叉编译使我们不需要花时间将各种支持包移植到目标板上。交叉编译的重点是构建要部署的目标包，而不是花时间获得只构建目标系统的先决条件。

### 为什么交叉编译比较困难

**不同的体系架构拥有不同的机器特性**

-   **Word size：** 是64位还是32位系统
-   **Endianness：** 是大端还是小端系统
-   **Alignment：** 是否必修按照4字节对齐方式进行访问
-   **Default signedness：** 默认数据类型是有符号还是无符号
-   **NOMMU：** 是否支持MMU

**交叉编译时的主机环境与目标环境不同**

-   **Configuration issues：**在主机系统上运行配置测试会给出错误的答案。配置还可以检测主机上是否存在包，并在目标没有包或版本不兼容时提供对包的支持。
-   **HOSTCC vs TARGETCC：**许多构建过程需要编译要在主机系统上运行的东西，比如上面的配置测试，或者生成代码的程序(比如一个C程序，它创建一个.h文件，然后在主构建过程中include)。只要用目标编译器替换宿主编译器，就会破坏需要构建在构建过程中运行的东西的包。这样的包需要同时访问主机和目标编译器，并且需要学习何时使用它们。
-   **Toolchain Leaks：**配置不当的交叉编译工具链可能会将主机系统的一些信息泄漏到编译后的程序中，导致通常很容易检测到的故障，但是很难诊断和纠正。工具链可能包含错误的头文件，或者在链接时搜索错误的库路径。共享库通常依赖于其他共享库，而这些共享库也可能在引入对主机系统的意外链接。
-   **Libraries：**动态链接的程序必须在编译时访问适当的共享库。目标系统的共享库需要添加到交叉编译工具链中，以便程序可以链接到它们。
-   **Testing：**在本地构建上，开发系统提供了一个方便的测试环境。在交叉编译时，确认成功构建的“hello world”可能需要配置(至少)引导加载程序、内核、根文件系统和共享库

## 交叉工具链

![20161024170548772](imgs/20161024170548772.jpg)

**交叉工具链**就是为了编译跨平台体系结构的程序代码而形成的由多个子工具构成的一套完整的工具集。同时，它隐藏了预处理、编译、汇编、链接等细节，当我们指定了源文件(.c)时，它会自动按照编译流程调用不同的子工具，自动生成最终的二进制程序映像(.bin)。 

交叉工具链是一个由**编译器、连接器和解释器**组成的综合开发环境，交叉编译工具链主要由**binutils、gcc和glibc**三个部分组成。有时出于减小 libc 库大小的考虑，也可以用别的 c 库来代替 glibc，例如 uClibc、dietlibc 和 newlib。

目标是为了：**生成（可以运行的）程序或库文件**，内部的执行过程和逻辑主要包含了：

-   编译
    -   编译的输入（对象）是：程序代码
    -   编译输出（目标）是：目标文件
    -   编译所需要的工具是：编译器
    -   编译器：常见的编译器，即为gcc
-   链接
    -   链接的输入（对象）是：（程序运行时所依赖的，或者某个库所依赖的另外一个）库（文件）
    -   链接的输出（目标）是：程序的可执行文件，或者是可以被别人调用的完整的库文件
    -   链接所需要的工具是：链接器
    -   链接器，即ld

为了将程序代码，编译成可执行文件，涉及到编译，链接（等其他步骤），要依赖到很多相关的工具，最核心的是编译器gcc，链接器ld。而所谓的工具，主要指的就是：和程序编译链接等相关的gcc，ld等工具

>   实际上，上面所说的ld，只是处理操作目标文件，二进制文件的最主要的一个工具
>   而和操作目标等文件相关的，还有其他很多工具的：as，objcopy，strip，ar等等工具的
>   所以，对此，GNU官网，弄出一个binutils，即binary utils，二进制工具（包），集成了这些，和操作二进制相关的工具集合，叫做binutils。
>
>   所以，之后你所见到的，常见的工具，就是那个著名的GNU Binutils了。

链：不止一个东西，然后，按照对应的逻辑链在一起。

-   不止一个东西：指的是就是前面所说的那个工具，即：和程序编译链接等相关的gcc，binutils等工具
-   按照对应的逻辑：指的就是，按照程序本身编译链接的先后顺序，即：先编译，后链接，再进行后期其他的处理等等，比如用objcopy去操作相应的目标文件等等。
-   如此的，将和程序编译链接等相关的gcc，binutils等工具按照先编译后链接等相关的编译程序的内在逻辑串起来，就成了我们所说工具链

普通所说的，工具链指的是当前自己的本地平台的工具链。用于交叉编译的工具链，就叫做交叉工具链。即那些工具，即编译的gcc，链接的ld，以及相关的工具，用于交叉编译的工具链，叫做交叉工具链。

交叉工具链，是用来交叉编译，跨平台的程序所用的。交叉工具链，和（本地）工具链类似，也是包含了很多对应的工具，交叉编译版本的gcc，ld，as等等。但是，由于其中最主要的是用于编译的gcc，所以，我们也常把交叉工具链，简称为交叉编译器

即严格意义上来说，交叉编译器，只是指的是交叉编译版本的gcc。但是实际上为了叫法上的方便，我们常说的交叉编译器，都是指的是交叉工具链。常说的交叉编译版本的gcc，比如arm-linux-gcc，实际上指代了，包含一系列交叉编译版本的交叉工具链（arm-linux-gcc，arm-linux-ld，arm-linux-as等等）

## 包含工具

### [Binutils](https://sourceware.org/binutils/)

Binutils是GNU工具之一，它包括链接器、汇编器和其他用于目标文件和档案的工具，它是二进制代码的处理维护工具。

Binutils工具包含的子程序主要有:：

-   **ld** GNU连接器the GNU linker.
-   **as** GNU汇编器the GNU assembler.

-   **gold** a new, faster, ELF only linker.

还有：

-   **addr2line** 把地址转换成文件名和所在的行数
-   **ar** A utility for creating, modifying and extracting from archives.
-   **c++filt** Filter to demangle encoded C++ symbols.
-   **dlltool** Creates files for building and using DLLs.
-   **gprof** Displays profiling information.
-   **nlmconv** Converts object code into an NLM.
-   **nm** Lists symbols from object files.
-   **objcopy** Copys and translates object files.
-   **objdump** Displays information from object files.
-   **ranlib** Generates an index to the contents of an archive.
-   **readelf** Displays information from any ELF format object file.
-   **size** Lists the section sizes of an object or archive file.
-   **strings** Lists printable strings from files.
-   **strip** Discards symbols

binutils介绍

### GCC

GNU编译器套件，支持C, C++, Java, Ada, Fortran, Objective-C等众多语言。

### GLibc

Linux上通常使用的C函数库为glibc。glibc是linux系统中最底层的api，几乎其它任何运行库都会依赖于glibc。glibc除了封装linux操作系统所提供的系统服务外，它本身也提供了许多其它一些必要功能服务的实现。

glibc 各个库作用介绍

因为嵌入式环境的资源及其紧张，所以现在除了glibc外，还有uClibc和eglibc可以选择，三者的关系可以参见这两篇文章：

uclibc eglibc glibc之间的区别和联系

Glibc vs uClibc Differences

### GDB

GDB用于调试程序

## 命名规则

```
arch-core-kernel-system
```

-   **arch：** 用于哪个目标平台
-   **core：** 使用的是哪个CPU Core，如Cortex A8，但是这一组命名好像比较灵活，在其它厂家提供的交叉编译链中，有以厂家名称命名的，也有以开发板命名的，或者直接是none或cross的
-   **kernel：** 所运行的OS，Linux，uclinux，bare（无OS）
-   **system：**交叉编译链所选择的库函数和目标映像的规范，如gnu，gnueabi等。其中gnu等价于glibc+oabi；gnueabi等价于glibc+eabi

## 如何使用已有交叉工具链

使用其他人针对某些CPU平台已经编译好的交叉编译链。我们只需要找到合适的，下载下来使用即可。

常见的交叉编译链下载地址：

1.  在 <http://ftp.arm.linux.org.uk/pub/armlinux/toolchain/> 下载已经编译好的交叉编译链
2.  在 <http://www.denx.de/en/Software/WebHome> 下载已经编译好的交叉编译链
3.  在<https://launchpad.net/gcc-arm-embedded>下载已经编译好的交叉编译链
4.  一些制作交叉编译链的工具中，包含了已经制作好的交叉编译链，可以直接拿来使用。如crosstool-NG
5.  如果购买了某个芯片或开发板，一般厂商会提供对应的整套开发软件，其中就包含了交叉编译链。

**厂家提供的工具一般是经过了严格的测试，并打入了一些必要的补丁，所以这种方式往往是最可靠的工具来源。**

| 方式       | 使用已有交叉编译链                                | 自己制作交叉编译链                                           |
| :--------- | ------------------------------------------------- | ------------------------------------------------------------ |
| 安装       | 一般提供压缩包                                    | 需要自己打包                                                 |
| 源码版本   | 一般使用较老的稳定版本，对于一些新的GCC特性不支持 | 可以使用自己需要的GCC特性的版本                              |
| 补丁       | 一般都会打上修复补丁                              | 普通开发者很难辨别需要打上哪些补丁，资深开发者可以针对自己的需求合入补丁 |
| 源码溯源   | 可能不清楚源码版本和补丁情况                      | 一切都可以定制                                               |
| 升级       | 一般不会升级                                      | 可以随时升级                                                 |
| 优化       | 一般已经针对特定CPU特性和性能进行优化             | 一般无法做到比厂家优化的更好，除非自己设计的CPU              |
| 技术支持   | 可以通过FAE进行支持，可能需要收费                 | 只能通过社区支持，免费                                       |
| 可靠性验证 | 已经通过了完善的验证                              | 自己验证，肯定没有专业人士验证的齐全                         |

## 引用

[Introduction to cross-compiling for Linux](http://landley.net/writing/docs/cross-compiling.html)

[交叉编译详解](https://www.crifan.com/files/doc/docbook/cross_compile/release/html/cross_compile.html#what_is_toolchain)

[交叉编译详解 一 概念篇](https://blog.csdn.net/pengfei240/article/details/52912833)