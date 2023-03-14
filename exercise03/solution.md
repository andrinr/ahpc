# Exercise 3

## 2

**100^3**  
Reading file took: 0 ms
Mass assignment took: 12 ms
Projection took: 1 ms

**200^3**  
Reading file took: 0 ms
Mass assignment took: 31 ms
Projection took: 0 ms

**400^3**  
Reading file took: 1 ms
Mass assignment took: 397 ms
Projection took: 1 ms

![linear linear](linear_linear.png)

![linear log](log_log.png)

![log linear](log_linear.png)


The linear linear plot is not very readbale as the first two measurements are very close by on the x-axis. 

The log-log tells the story the best, as its clear how the runtime increase more than linearly with the number of particles.

## 3

*ngp* kernel

![ngp](mass_assignment/ngp.png)

*cic* kernel

![cic](mass_assignment/cic.png)

*tsc* kernel

![tsc](mass_assignment/tsc.png)

*psc* kernel

![psc](mass_assignment/psc.png)


## 4

psc kernel mass assignment with openmp took 24888 ms
with openmp and simd took 16065 ms

Therfore openmp does actually slow down the assignment. 

