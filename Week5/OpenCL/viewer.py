import numpy as np
from matplotlib import pyplot as plt
#lets view our image
fd = open('OutputImage.raw', 'rb')
rows = 512
cols = 512
f = np.fromfile(fd, dtype=np.uint8, count=rows * cols * 4)
im = f.reshape((rows, cols, 4))  #notice row, column format
# fd.close()
plt.imsave('OutputImage.png', im)