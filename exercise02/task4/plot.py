import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('out.txt', delimiter=' ', unpack=True)
data = np.log(data)
plt.imshow(data, cmap='plasma', interpolation='nearest')
plt.show()