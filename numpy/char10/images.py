#encoding=utf8

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage

image = misc.lena().astype(np.float32)

plt.subplot(221)
plt.title('Original Image')
img = plt.imshow(image, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(222)
plt.title('Meidan Filter')
filtered = ndimage.median_filter(image, size=(42,42))
img = plt.imshow(filtered, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(223)
plt.title('Rotated')
rotated = ndimage.rotate(image, 90)
img = plt.imshow(rotated, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(224)
plt.title('Prewitt Filter')
ff = ndimage.prewitt(image)
img = plt.imshow(ff, cmap=plt.cm.gray)

plt.savefig('images/image.png', format='png')
