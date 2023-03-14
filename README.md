# CUDA_Spring2023

The companion git repo for my Spring 2023 CUDA course. The first week work is largely getting started with the two most common machine and deeplearning platforms Google Colab and Kaggle. Both are equally good but Kaggle provides more recent GPUs and provides a range of GPUs to pick from. Google Colab is limited to one GPU or a TPU. I have an excellent batch of students with me this term who have already started extending the course codes to alternate frameworks. Thanks [Meesum Ali Qazalbash](https://github.com/Qazalbash) for the OpenCL port and [Akeel Medina](https://www.github.com/AkeelMedina22) for the [NVIDIA warp](https://github.com/nvidia/warp) and [TAICHI](https://github.com/taichi-dev/taichi) conversion of codes.

## Week 1

If you examine the Week1 folder your will see 6 jupyter notebooks. The first three do hello world in Google Colab while the later 3 do hello world in Kaggle. Feel free to get comfortable with both of these environments. [Meesum Ali Qazalbash](https//github.com/Qazalbash) has shared OpenCL version of Week 1 code. Those of you who do not have an NVIDIA GPU, you can use the OpenCL version of codes. Thanks [Meesum Ali Qazalbash](https//github.com/Qazalbash).

## Week 2

We extend our knowledge of CUDA slightly by getting to know how to initialize data on the device (GPU). We also see how to check for errors in our CUDA code. We also see how to add two numbers on the GPU and return the result from device (GPU) back to the host (CPU). [Meesum Ali Qazalbash](https//github.com/Qazalbash) has shared OpenCL version of Week 2 codes. Those of you who do not have an NVIDIA GPU, you can use the OpenCL version of codes. Thanks [Meesum Ali Qazalbash](https//github.com/Qazalbash).

## Week 3

We extend our knowledge of CUDA by getting to know how to pass a large array of data on the device (GPU). We also see how to time code on the CPU as well as on the GPU. [Meesum Ali Qazalbash](https//github.com/Qazalbash) has shared OpenCL version of Week 3 codes. Those of you who do not have an NVIDIA GPU, you can use the OpenCL version of codes. Thanks [Meesum Ali Qazalbash](https//github.com/Qazalbash).
[Akeel Medina](https://www.github.com/AkeelMedina22) has shared Week1 to Week3 codes converted to [NVIDIA Warp library](https://github.com/nvidia/warp) if you want to create code in python and call CUDA kernels. Thanks [Akeel Medina](https://www.github.com/AkeelMedina22).

## Week 4

We extend our knowledge by starting work in 2D. We first start with creating a simple [image viewer in python](Week4/ImageViewer.ipynb). Next we see how to [lauch threads in 2D](Week4/Simple2D.ipynb). Then, we look at how to do [raw image processing in CUDA](Week4/RawImageProcessing.ipynb) and Finally, we wrap up this week with a small task where students are asked to finish an implementation of [matrix multiplication in CUDA](Week4/MatrixMultiplication_Task.ipynb).

## Week 5

We now see how we can use shared memory to efficienlty optimize writing to device memory across multple threads concurrently. The first example [Shared Memory Bitmap](Week5/SharedMemoryBitmap.ipynb) demonstrates a simple example of creating a bitmap image using shared memory to store data in a bitmap using multiple parallel threads. The examples demonstrates the general dataflow pattern of working with the shared memory. We create the shared memory buffer first. Then we ask each thread to fill up its data from the global memory to shared memory. Next, we put the __syncthreads() call which inserts a synchronization barrier. This barrier ensures that the following execution is halted until all threads in the block have finished writing to their shared memory location. After __syncthreads call, each threads reads data from its own shared memory location for processing. If we comment the __syncthreads() call we see noticable garbage values in the output as there is no guarantee that all threads have written to their shared memory location. The second example [MatrixMultiplicationTiled](Week5/MatrixMultiplicationTiled.ipynb) optimizes the MatrixMultiplication example through tiliing whereby tile of data is copied from device memory into shared memory and then used for matrix multiplication. We do comparison of the naive matrixmultiplication against CPU as well as the optimized matrix multiplication on the GPU.

## Week 6

We start Week 6 with understanding parallel reduction which is usually used to parallelize sum calculation on the GPU. The first example [ReductionSumGPU](Week6/ReductionSumGPU.ipynb) demonstrates how to implement naive reduction on the CPU and then on the GPU. Finally, we see an implementation of reduction which is optimized by avoiding branch divergence. The second example [DotProductGPU](Week6/DotProductGPU.ipynb) calculates dot product on the GPU using shared memory and parallel reduction. In the first phase, the vector products are calculated and stored in shared memory. In the second phase, the product pair are summed using parallel reduction. Finally, we cover how to do basic profiling of our code using tools that ship with the NVIDIA CUDA Computing SDK namely nvprof and ncu in [TestProfilers_nvprof_ncu](Week6/TestProfilers_nvprof_ncu.ipynb).

## Week 7

Week 7 starts with understanding of global static memory, unified memory and zero copy memory. We give three examples on these including [globalStaticMemory](Week7/globalStaticMemory.ipynb), [SumArrayZeroCopy](Week7/SumArrayZeroCopy.ipynb) and [DotProductGPU_UnifiedMemory](Week7/DotProductGPU_UnifiedMemory.ipynb). 

## Week 8

With Week 8, we start working on parallel patterns. We talk about two patterns in Week 8. Convolution and Prefix Sum. The first pattern (Convolution) which helps us filter a given set of data with some set of coefficients (kernel or mask). The first example [Conv1D](Week8/Conv1D.ipynb) shows how to carry out 1D convolution in CUDA. The next example [Conv1D_Modified](Week8/Conv1D_Modified.ipynb) shows how to move the filter mask into constant memory to optimize read accessses inside the CUDA kernel. Finally, we wrap the dicussion on convolution with implementing tiling whereby halo elements are moved from global memory into shared memory to increase the memory acces bandwidth. This is shown in [Conv1D_Tiled](Week8/Conv1D_Tiled.ipynb) example. Finally, we give an example of 2D convolution on the host [Conv2D](Week8/Conv2D.ipynb) and request the students to implement the 2D convolution on the GPU and then optimize it using tiling as an exercise. After convolution, we move to prefix sum. Two examples are given: [PrefixSum_Correct](Week8/PrefixSum_Correct.ipynb) and [PrefixSum_WorkEfficient](Week8/PrefixSum_WorkEfficient.ipynb).

## Week 9

Week 9 focuses on the third parallel pattern that is histogram. We talk about four different strategies for computing the histogram. These strategies are given in four examples: [Strategy 1](Week9/Histogram_Strategy_1.ipynb), [Strategy 2](Week9/Histogram_Strategy_2.ipynb), [Strategy 3](Week9/Histogram_Strategy_3.ipynb) and [Strategy 4](Week9/Histogram_Strategy_4.ipynb).

# Contributors

<a href="https://github.com/mmmovania/CUDA_Spring2023/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mmmovania/CUDA_Spring2023" />
</a>
