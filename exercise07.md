# Exercise 07

## Task 1

```{bash}
mpirun -n 2 main ../data/B100.00100 100 ngp
```
It does not parallelize the work but simply runs everything twice.

All the relevant code can be found in the [main.cxx](mass_assignment/src/main.cxx) file.

## Task 2

Table of results:

| # of processes | read file [ms] | mass assignment [ms] |
|----------------|----------------|----------------------|
| 1              | 11             | 233                  |
| 2              | 6              | 127                  |

All the relevant code can be found in the [main.cxx](mass_assignment/src/main.cxx) file.

## Task 3

![compare.png](mass_assignment/out/compare.png)

The line is as straight as numerical precision allows it to be. 