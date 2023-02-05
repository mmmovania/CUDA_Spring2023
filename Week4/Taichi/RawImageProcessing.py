'''
Taichi has a nice GUI that makes image processing convenient and GPU accelerated.
However, it does not share much similarity to CUDA syntax-wise, and most simple 
concepts are abstracted from the user. Taichi achieves huge performance speedups 
over traditional Python automatically, and can be comparable to CUDA: 
https://docs.taichi-lang.org/docs/performance. As you can see, using Taichi to its 
full potential can be incredibly complex, and there are many interesting uses for it.
'''

import os

import cv2 as cv
import numpy as np
import taichi as ti
import wget

ti.init(arch=ti.gpu)

filename = "https://upload.wikimedia.org/wikipedia/commons/3/32/Recursive_raytrace_of_a_sphere.png"
if not os.path.exists("Week4/Taichi/img.jpg"):
    img = wget.download(filename, out="Week4/Taichi/img.jpg")

I = cv.imread("Week4/Taichi/img.jpg")
# Taichi GUI pixel max value is 1.0
I = cv.normalize(I,
                 None,
                 alpha=0.00001,
                 beta=1,
                 norm_type=cv.NORM_MINMAX,
                 dtype=cv.CV_32F)

HEIGHT, WIDTH, CHANNELS = I.shape
pixels = ti.field(dtype=float, shape=(WIDTH, HEIGHT, CHANNELS))

# Taichi coordinate system doesn't match OpenCV/Numpy
pixels.from_numpy(np.rot90(np.rot90(np.rot90(I))))


@ti.kernel
def brighten(coeff: float):
    # The outer for loop only is Taichi accelerated
    for i, j in ti.ndrange(WIDTH, HEIGHT):
        pixels[i, j, 0] *= coeff
        pixels[i, j, 1] *= coeff
        pixels[i, j, 2] *= coeff


if __name__ == "__main__":

    gui = ti.GUI("RawImageProcessing", res=(WIDTH, HEIGHT))
    coeff = 0.99
    sign = 1.0
    count = 0

    while gui.running:

        if count == 100:
            coeff += 0.02 * sign
            sign *= -1.0
            count = 0
        count += 1

        brighten(coeff)
        gui.set_image(pixels)
        gui.show()