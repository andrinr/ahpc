import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

data100_serial = np.loadtxt('out_power1000000_serial.txt', delimiter=' ', unpack=True)
data100_par = np.loadtxt('out_power1000000.txt', delimiter=' ', unpack=True)

diff = data100_par[:,0] / data100_serial[:,0]

# plot
ax.plot(data100_par[:, 1], diff)

ax.set_yscale('log')

ax.set_xlabel('k')
ax.set_ylabel('P(k)')

plt.show()