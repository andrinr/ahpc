import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('out_projected.txt', delimiter=' ', unpack=True)
data = np.log(data)
plt.imshow(data, cmap='plasma', interpolation='nearest')
plt.savefig('projected.png')
plt.show()

data = np.loadtxt('out_power.txt', delimiter=' ', unpack=True)
print(data[2:,0])
plt.plot(data[2:, 1], data[2:,0])
plt.show()
plt.savefig('power.png')