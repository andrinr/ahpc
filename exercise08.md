# Exercise 08

## Task 1 

Look at the `loadParticles` function found in [helpers.cxx](mass_assignment/src/helpers.cxx).

## Task 2

Look at the `loadParticles` function found in [helpers.cxx](mass_assignment/src/helpers.cxx).

## Task 3

Look at the `sortParticles` function found in [helpers.cxx](mass_assignment/src/helpers.cxx).


## Task 4

Look at the `reshuffleParticles` function found in [helpers.cxx](mass_assignment/src/helpers.cxx).

## Task 5

Look at the `assign` function found in [helpers.cxx](mass_assignment/src/helpers.cxx).

## Task 6

Found in [main.cxx](mass_assignment/src/main.cxx).

## Task 7

Look at the function `bin` in [helpers.cxx](mass_assignment/src/main.cxx).

## Task 8

The implementation I used differs a bit from the one explained in the lectures. It uses non blocking I send and I recv to "reduce" the ghost regions in the corresponding cells. The advantage is that is simpler and still very efficent as it takes only a around 20ms on my machine. There is a slight data overhead as I create additional arrays to temporarily store the data. The implementation can be found in [main.cxx](mass_assignment/src/main.cxx).


## Current state
Note that with my current there is still an error with the ghost cell reduction, as the ghost regions have invlaid values in them. 

Furthermore there is another segfault after the ghost cell reduction. 

Unfortunately I was not able to fix these issues in time.