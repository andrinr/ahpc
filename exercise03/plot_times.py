import matplotlib.pyplot as plt
import numpy as np

reading_times = np.array([0, 0, 1.0])
mass_assignment_times = np.array([12, 31, 397])
projection_times = np.array([1, 0, 1])

total = reading_times + mass_assignment_times + projection_times
n_particles = [100*3, 200**3, 500**3]

# linear linear figure
plt.figure()
plt.plot(n_particles, total)
plt.title('linear-linear plot')
plt.xlabel('number of particles')
plt.ylabel('time (ms)')
plt.savefig('linear_linear.png')

# log log figure
plt.figure()
plt.loglog(n_particles, total)
plt.title('log-log plot')
plt.xlabel('number of particles')
plt.ylabel('time (ms)')
plt.savefig('log_log.png')

# log linear figure
plt.figure()
plt.semilogx(n_particles, total)
plt.title('log-linear plot')
plt.xlabel('number of particles')
plt.ylabel('time (ms)')
plt.savefig('log_linear.png')
