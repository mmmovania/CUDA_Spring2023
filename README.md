# CUDA_Spring2023

The companion git repo for my Spring 2023 CUDA course. The first week work is largely getting started with the two most common machine and deeplearning platforms Google Colab and Kaggle. Both are equally good but Kaggle provides more recent GPUs and provides a range of GPUs to pick from. Google Colab is limited to one GPU or a TPU. I have an excellent batch of students with me this term who have already started extending the course codes to alternate frameworks. Thanks [Meesum Ali Qazalbash](https://github.com/Qazalbash) for the OpenCL port and [Akeel Medina](https://www.github.com/AkeelMedina22) for the [NVIDIA warp](https://github.com/nvidia/warp) conversion of codes.

## Week 1

If you examine the Week1 folder your will see 6 jupyter notebooks. The first three do hello world in Google Colab while the later 3 do hello world in Kaggle. Feel free to get comfortable with both of these environments. [Meesum Ali Qazalbash](https//github.com/Qazalbash) has shared OpenCL version of Week 1 code. Those of you who do not have an NVIDIA GPU, you can use the OpenCL version of codes. Thanks [Meesum Ali Qazalbash](https//github.com/Qazalbash).

## Week 2

We extend our knowledge of CUDA slightly by getting to know how to initialize data on the device (GPU). We also see how to check for errors in our CUDA code. We also see how to add two numbers on the GPU and return the result from device (GPU) back to the host (CPU). [Meesum Ali Qazalbash](https//github.com/Qazalbash) has shared OpenCL version of Week 2 codes. Those of you who do not have an NVIDIA GPU, you can use the OpenCL version of codes. Thanks [Meesum Ali Qazalbash](https//github.com/Qazalbash).

## Week 3

We extend our knowledge of CUDA by getting to know how to pass a large array of data on the device (GPU). We also see how to time code on the CPU as well as on the GPU. [Meesum Ali Qazalbash](https//github.com/Qazalbash) has shared OpenCL version of Week 3 codes. Those of you who do not have an NVIDIA GPU, you can use the OpenCL version of codes. Thanks [Meesum Ali Qazalbash](https//github.com/Qazalbash).
[Akeel Medina](https://www.github.com/AkeelMedina22) has shared Week1 to Week3 codes converted to [NVIDIA Warp library](https://github.com/nvidia/warp) if you want to create code in python and call CUDA kernels. Thanks [Akeel Medina](https://www.github.com/AkeelMedina22).

## Week 4

We extend our knowledge by starting work in 2D. We first start with creating a simple [image viewer in python](Week4/ImageViewer.ipynb). Next we see how to [lauch threads in 2D](Week4/Simple2D.ipynb). Then, we look at how to do [raw image processing in CUDA](Week4/RawImageProcessing.ipynb) and Finally, we wrap up this week with a small task where students are asked to finish an implementation of [matrix multiplication in CUDA](Week4/MatrixMultiplication_Task.ipynb).

# Contributors

<a href="https://github.com/mmmovania/CUDA_Spring2023/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mmmovania/CUDA_Spring2023" />
</a>
