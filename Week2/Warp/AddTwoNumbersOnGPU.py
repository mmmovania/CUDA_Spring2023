import warp as wp
import numpy as np

wp.init()

# Note that all kernel parameters must be strongly typed
@wp.kernel
def AddTwoNumbersOnGPU(c: wp.array(dtype=wp.int32), a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    c[tid] = a[tid]+b[tid]

# int32 and float32 are considered compute types, 
# all other dtypes are storage types and cannot be used for arithmetic operations
c = wp.array(np.array([0]), dtype=wp.int32, device="cuda")
a = wp.array(np.array([10]), dtype=wp.int32, device="cuda")
b = wp.array(np.array([20]), dtype=wp.int32, device="cuda")

# launch kernel
wp.launch(kernel=AddTwoNumbersOnGPU,
            dim=len(c),
            inputs=[c, a, b])

wp.synchronize()

c = c.to("cpu")
a = a.to("cpu")
b = b.to("cpu")

# Array slicing is not allowed on warp arrays, convert to np to using [] notation
print("{} + {} = {}".format(np.array(a)[0], np.array(b)[0], np.array(c)[0]))
