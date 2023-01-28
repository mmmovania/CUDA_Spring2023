import warp as wp
import numpy as np

N = wp.constant(10)
X = wp.constant(10)
Y = wp.constant(10)

wp.init()

c = np.zeros(N.val)
a = np.ones(N.val)
b = np.ones(N.val)

b = b*2 

# Sum = (1+2)*10 = 30

cpu_sum = np.sum(a+b)

print("Sum on CPU = {}".format(cpu_sum))

@wp.kernel
def AddTwoNumbersOnGPU(c: wp.array(dtype=wp.int32), a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32)):
    i,j = wp.tid()
    tid = (i*X)+j
    if tid < N:
        c[tid] = a[tid]+b[tid]

gpu_c = wp.from_numpy(c, dtype=wp.int32)
gpu_a = wp.from_numpy(a, dtype=wp.int32)
gpu_b = wp.from_numpy(b, dtype=wp.int32)

# launch kernel
wp.launch(kernel=AddTwoNumbersOnGPU,
            dim=(X.val, Y.val),
            inputs=[gpu_c, gpu_a, gpu_b])

wp.synchronize()

gpu_c = np.array(gpu_c.to("cpu"))
gpu_a = np.array(gpu_a.to("cpu"))
gpu_b = np.array(gpu_b.to("cpu"))

gpu_sum = np.sum(gpu_c)

print("Sum on GPU = {}".format(gpu_sum))
