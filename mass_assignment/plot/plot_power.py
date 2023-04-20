import matplotlib.pyplot as plt
import numpy as np
data100 = np.loadtxt('out_power1000000.txt', delimiter=' ', unpack=True)


# log log plot
plt.loglog(data100[:, 1], data100[:,0], label='1000000')

plt.show()
plt.savefig('power.png')