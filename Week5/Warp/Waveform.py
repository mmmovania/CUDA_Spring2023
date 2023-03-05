import warp as wp
import numpy as np
import cv2 as cv

N = wp.constant(256)
PI = wp.constant(np.pi)

wp.init()

pixels= np.zeros(shape=(N.val, N.val))

@wp.kernel
def Waveform(pixels: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    if i < N and j < N:
        # pixels[i, j] = 255.0 * (np.sin(i*2.0*PI*128.0) + 1.0) * (np.sin(j*2.0*PI/128.0) + 1.0) / 4.0
        
        dx = np.float32(i - N)
        dy = np.float32(j - N)

        pixels[i, j] = 255.0 * np.sin(np.sqrt(dx*dx + dy*dy)*2.0*PI/256.0)


pixels = wp.from_numpy(pixels, dtype=wp.float32)

# launch kernel
wp.launch(kernel=Waveform,
          dim=(N.val, N.val),
          inputs=[pixels])

pixels = np.array(pixels.to("cpu"))

cv.imshow("Waveform", pixels)
cv.waitKey(0)
cv.destroyAllWindows()