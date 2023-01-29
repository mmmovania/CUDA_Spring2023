import warp as wp
import numpy as np

wp.init()


@wp.kernel
def HelloWorldKernel():
    print("Hello World from the GPU!")


# launch kernel
wp.launch(kernel=HelloWorldKernel, dim=8, inputs=[])

# Note that synchronization is not always required, like in CUDA.
# Check the FAQ's for more details: https://github.com/NVIDIA/warp
wp.synchronize()

print("Hello World from the CPU after kernel execution!")