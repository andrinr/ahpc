import matplotlib.pyplot as plt
import numpy as np
data100 = np.loadtxt('out_power1000000.txt', delimiter=' ', unpack=True)
data200 = np.loadtxt('out_power8000000.txt', delimiter=' ', unpack=True)
data500 = np.loadtxt('out_power125000000.txt', delimiter=' ', unpack=True)


# log log plot
plt.loglog(data100[2:, 1], data100[2:,0], label='1000000')
plt.loglog(data200[2:, 1], data200[2:,0], label='8000000')
plt.loglog(data500[2:, 1], data500[2:,0], label='125000000')

plt.show()
plt.savefig('power.png')