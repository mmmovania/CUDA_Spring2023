# Shared memory is not a feature of Warp, so Tiled Matrix Multiplication is impossible

import warp as wp
import numpy as np
import time

N = wp.constant(10000)

wp.init()

c = np.zeros(shape=(N.val, N.val))
a = np.random.random(size=(N.val, N.val))
b = np.random.random(size=(N.val, N.val))

start = time.time()
cpu_mult = a@b
end = time.time()
print("CPU Time: {}".format(end-start))

@wp.kernel
def NaiveMatrixMultiplication(c: wp.array2d(dtype=wp.int32),
                       a: wp.array2d(dtype=wp.int32),
                       b: wp.array2d(dtype=wp.int32)):
    i, j = wp.tid()
    if i < N and j < N:
        sum = wp.int32(0)
        for x in range(N):
            sum += a[i, j]*b[i, j]
        c[i, j] = sum


gpu_c = wp.from_numpy(c, dtype=wp.int32)
gpu_a = wp.from_numpy(a, dtype=wp.int32)
gpu_b = wp.from_numpy(b, dtype=wp.int32)

start = time.time()
# launch kernel
wp.launch(kernel=NaiveMatrixMultiplication,
          dim=(N.val, N.val),
          inputs=[gpu_c, gpu_a, gpu_b])

end = time.time()

print("GPU Time: {}".format(end-start))

gpu_c = np.array(gpu_c.to("cpu"))


print("Equal = {}".format(np.array_equal(gpu_c,c)))
