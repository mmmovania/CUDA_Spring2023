import warp as wp
import numpy as np

N = 10

wp.init()

# Note that all kernel parameters must be strongly typed
@wp.kernel
def AddTwoNumbersOnGPU(data: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    data[tid] = tid

# int32 and float32 are considered compute types, 
# all other dtypes are storage types and cannot be used for arithmetic operations
data = wp.array(np.array([0 for i in range(N)]), dtype=wp.int32, device="cuda")

# launch kernel
wp.launch(kernel=AddTwoNumbersOnGPU,
            dim=N,
            inputs=[data])

wp.synchronize()

data = np.array(data.to("cpu"))

print("After GPU execution:")
for i in range(N):
    print("data[{}] = {}".format(i, data[i]))