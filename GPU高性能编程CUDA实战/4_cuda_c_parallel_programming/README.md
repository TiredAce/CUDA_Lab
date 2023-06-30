## 第4章 CUDA C并行编程

### 4.1 Target

* 了解`CUDA`在实现并行性时采用的一种重要方式
* 用`CUDA C`编写第一段并行代码

### 4.2 CUDA并行编程

将标准`C`放到`GPU`设备上运行是很容易的。只需在函数定义前加上`__global__`修饰符。并通过尖括号语法去调用它，就可以在`GPU`上执行这个函数。但是这只是调用了一个和函数，并且该函数在`GPU`上以串行方式运行，也是很低效的，`NVIDIA`早已对图形处理器进行优化，使其可以通知并行执行数百次的计算。

#### 4.2.1 矢量求和运算

1. 基于`CPU`的矢量求和

```c++

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 10

void add(int* a, int* b, int* c) {
	int tid = 0; // 这是第0个CPU， 因此索引从0开始
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid++;  // 由于只有一个CPU，因此每次递增1
	}
}

int main() {
	int a[N], b[N], c[N];

	for (int i = 0; i < N; i++) {
		a[i] = -1;
		b[i] = i * i;
	}
	
	add(a, b, c);

	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	return 0;
}
```

在`add`函数中之所以采用`while`循环，是为了使代码能够在拥有多个`CPU`或者`CPU`核的系统上并行运算。例如双核处理器每次递增的大小改为`2`。

2. 基于`GPU`的矢量求和

```c++

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 10

__global__ void add(int* a, int* b, int* c) {
	int tid = blockIdx.x;   //计算该索引处的数据
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	// 在GPU上分配内存
	cudaMalloc(&dev_a, N * sizeof(int));
	cudaMalloc(&dev_b, N * sizeof(int));
	cudaMalloc(&dev_c, N * sizeof(int));

	// GPU上赋值
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	// 将数组 a 和 b 复制到GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	
	add << <N, 1 >> > (dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	// 展示结果
	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
```

以上代码使用了一些通用模式：

* 调用`cudaMalloc()`在设备上为三个数组分配内存
* 用完`GPU`通过`cudaFree()`释放它们
* 通过`cudaMemcpy()`将输入数据复制到设备中，同时指定参数传递数据。
* 通过尖括号语法，在主机代码`main()`中执行`add()`中的设备代码。

不同的是函数`add()`，当你调用这个内核函数时，你会指定执行内核函数的线程块和线程网格的维度。在这段代码中，每个线程使用 `blockIdx.x` 的值作为索引来访问数组元素。因为每个线程都有唯一的索引，所以它们会并行地访问不同的数组元素并执行相加操作。

还有两个值得注意的地方，尖括号中的参数以及和函数中包含的代码，这两处地方都引入了新的概念。第一个参数表示设备在执行核函数时使用的并行线程块的数量，第二个参数表示线程块中线程的数量。

如果想编写更大规模的并行应用程序，那么也是非常容易的。只需要改动`N`的值，这样可以启动数万个并行线程块。然而，数组的每一维不能操作`65535`。这是硬件的限制，如果启动的线程块数量超过这个限值，那么程序将运行失败。

### 4.2.2 一个有趣的示例

生成一个`Julia`集，该算法非常简单，通过一个简单的迭代等式对复平面中的点求值。如果在计算某个点时，迭代灯饰的计算结果是发散的，那么这个点就不属于`Julia`集合。

这个迭代等式非常简单：
$$
Z_{n+1} = Z^2_{n} + C
$$

1. 基于`CPU`的`Julia`集

