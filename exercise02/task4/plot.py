import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

data = np.loadtxt('out.txt', delimiter=' ', unpack=True)
# take log
data = np.log(data)
plt.imshow(data, cmap='plasma', interpolation='nearest')
plt.show()
