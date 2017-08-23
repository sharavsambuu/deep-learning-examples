from skimage import data
from matplotlib import pyplot as plt
import numpy as np

img = data.camera().astype(np.float32)

plt.imshow(img, cmap='gray')
print(img.shape)
plt.show()
