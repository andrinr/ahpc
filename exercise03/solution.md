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

The log-log tells the story the best, as the runtime is linear in the number of elements. (todo)

## 3

*ngp* kernel

![ngp](mass_assignment/ngp.png)

*cic* kernel

![cic](mass_assignment/cic.png)

*tsc* kernel

![tsc](mass_assignment/tsc.png)

*psc* kernel

![psc](mass_assignment/psc.png)


