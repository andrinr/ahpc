import matplotlib.pyplot as plt
import numpy as np
data100 = np.loadtxt('out_power1000000.txt', delimiter=' ', unpack=True)
data200 = np.loadtxt('out_power8000000.txt', delimiter=' ', unpack=True)
data500 = np.loadtxt('out_power125000000.txt', delimiter=' ', unpack=True)


# log log plot
plt.loglog(data100[:, 1], data100[:,0], label='1000000')
plt.loglog(data200[:, 1], data200[:,0], label='8000000')
plt.loglog(data500[:, 1], data500[::,0], label='125000000')

legend = ['100^2', '200^2', '500^2']
plt.legend(legend, loc='upper right')
plt.show()
plt.savefig('power.png')