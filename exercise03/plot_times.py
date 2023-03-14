import matplotlib.pyplot as plt
import numpy as np

reading_times = np.array([13, 675, 6412])
mass_assignment_times = np.array([247, 1583, 23842])
projection_times = np.array([1, 1, 1])

total = reading_times + mass_assignment_times + projection_times
n_particles = [100*3, 200**3, 500**3]

# linear linear figure
plt.figure()
plt.plot(n_particles, reading_times, label='reading')
plt.plot(n_particles, mass_assignment_times, label='mass assignment')
plt.plot(n_particles, projection_times, label='projection')
plt.plot(n_particles, total, label='total')
plt.legend()
plt.title('linear-linear plot')
plt.xlabel('number of particles')
plt.ylabel('time (ms)')
plt.savefig('linear_linear.png')

# log log figure
plt.figure()
plt.loglog(n_particles, reading_times, label='reading')
plt.loglog(n_particles, mass_assignment_times, label='mass assignment')
plt.loglog(n_particles, projection_times, label='projection')
plt.loglog(n_particles, total, label='total')
plt.legend()
plt.title('log-log plot')
plt.xlabel('number of particles')
plt.ylabel('time (ms)')
plt.savefig('log_log.png')

# log linear figure
plt.figure()
plt.semilogx(n_particles, reading_times, label='reading')
plt.semilogx(n_particles, mass_assignment_times, label='mass assignment')
plt.semilogx(n_particles, projection_times, label='projection')
plt.semilogx(n_particles, total, label='total')
plt.legend()
plt.title('log-linear plot')
plt.xlabel('number of particles')
plt.ylabel('time (ms)')
plt.savefig('log_linear.png')


