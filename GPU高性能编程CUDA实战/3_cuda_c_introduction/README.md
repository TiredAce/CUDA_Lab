## 第3章 CUDA C简介

### 3.1 Target

* 编写一段`cuda`代码
* 了解`Host`和`Device`编写代码的区别
* 如何从主机上运行设备代码
* 了解如何在支持`CUDA`的设备上使用设备内存
* 了解如何查询系统中支持`CUDA`的设备信息

### 3.2 第一个程序

```c++

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void kernel() {
}

int main() {
	kernel << <1, 1 >> > ();
	printf("Hello, world!\n");
	return 0;
}
```

与标准的`C++`相比，多了两个地方：

* 一个空的函数`kernel()`，并带有修饰符`__global__`。
* 对这个空函数的调用，并且带有修饰字符`<<<1, 1>>>`。

`__global__`修饰符告诉编译器，函数应该编译在设备而不是主机上运行。函数`kernel()`将被交给编译设备代码的编译器，`main()`函数交给主机编译器。

调用设备的代码，所用的尖括号表示要将一些参数给运行时系统。这些参数并不是传递给设备代码的参数，而是告诉运行时如何启动设备代码。

```c++

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void add(int a, int b, int* c) {
	*c = a + b;
}

int main() {
	int c;
	int* dev_c;
	cudaMalloc((void**)&dev_c, sizeof(int));   // 在cuda上创建内存

	add << <1, 1 >> > (2, 7, dev_c);
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost); 
	printf("2 + 7 = %d\n", c);
	cudaFree(dev_c);   // 释放cuda上的内存
	
	return 0;
}
```

以上代码包含两个概念：

* 可以像调用`C`函数那样将参数传递给核函数
* 当设备执行任何有用的操作时，都需要分配内存。

需要注意的是通过`cudaMalloc()`来分配内存。这个函数调用的行为非常类似于标准的`malloc()`函数，但该函数的作用是告诉`CUDA`运行时在设备上分配内存。第一个参数是一个指针，指向用于保存新分配内存地址的变量，第二个参数是分配内存的大小。

主机指针只能访问主机代码中的内存，而设备指针也只能访问设备代码中的内存。

主机通过调用`cudaMemcpy()`来访问设备上的内存。`cudaMemcpyDeviceToHost`表示的是传递参数源指针位于设备上，而且目标指针为主机。同理得到`cudaMemcpyHostToDevice`与`cudaMemcpyDeviceToDevice`.

### 3.3 查询设备

```c++
int count;
cudaGetDeviceCount(&count);
```

在调用`cudaGetDeviceCount()`后，得到设备的数量，可以对每个设备进行迭代，使用`cudaGetDeviceProperties(&prop, i)`函数得到，并查询各个设备的相关属性。

### 3.4 设备属性的使用

使用迭代的方式查找设备有些繁琐，`CUDA`提供自动方式来执行这个迭代操作。

```c++
cudaDeviceProp prop;
memset($prop, 0, sizeof(cudaDeviceProp));
prop.major = 1;
prop.minor = 3;
```

填充完后，将其传递给`cudaChooseDevice()`，这样`CUDA`运行时将查找是否存在某个设备满足这些条件，并返回一个设备`ID`。然后通过`cudaSetDevice()`就可将操作都在这个设备上执行。
