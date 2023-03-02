# Exercise 1


## 1.1 

- ``blitz`` is the namespace
- ``blitz::Array`` invokes the constructor whereas 10, 8 and 6 are the arguments
- ``<float, 3>`` corresponds to the template parameters

## 1.2

This line creates a tensor of size three where the corresponding dimensions are 10, 8 and 6 with the float dataype. According to the blitz++ documentation, the tensor is initialized with zeros.

```cpp
#include "blitz/array.h"

int  main() {
    blitz::Array <float ,3> data (10,8,6);

    std::cout  << data << std::endl;
}
```

where the ouput is:

```bash
(0,9) x (0,7) x (0,5)
[ 0 0 0 0 0 0 
  0 0 0 0 0 0 
    ...
  0 0 0 0 0 0 
  0 0 0 0 0 0 ]

```

## 2.

- number of particles: 4096^3 = 2^36 

- number of bytes of all particles: 2^36 * 56 bytes 

- bytes per gigabyte 1GB = 2^30 bytes

- number of gigabytes for all particles: 2^36 * 56 / 2^30 = 3584 GB

- 3584 GB / 64 $\frac{GB}{node}$ = 56 nodes

## 3. (1 - 3)

- ``od -j 0 -N 8 -t f8 --endian=big b0-final.std`` => duration 1.0000000000000107
- ``od -j 8 -N 4 -t u4 --endian=big b0-final.std`` => 158095 particles

## 4. 

total per particle byte offset: 32 + 4 = 36
header offset 28 + 4 = 32

- For the mass we use the offset: 32 + 99 * 36 = 3640. The command is `` od -j 3596 -N 4 -t f4 --endian=big b0-final.std `` and the result: 2.0393363e-09

- X positions: offset 32 + 99 * 36 + 4. The command is `` od -j 3600 -N 4 -t f4  --endian=big b0-final.std `` and the result: -0.008689348

- Y positions: offset 32 + 99 * 36 + 4 * 2. The command is `` od -j 3604 -N 4 -t f4 --endian=big b0-final.std `` and the result: -0.03393134

- Z positions: offset 32 + 99 * 36 + 4 * 3. The command is `` od -j 3608 -N 4 -t f4 --endian=big b0-final.std `` and the result:  -0.03598262

## 5.

solution in ``analysis.py``

- max X dark: 0.49999925
- max Y dark: 0.499731
- max Z dark: 0.4999042
- min X dark: -0.4999119
- min Y dark: -0.49984062
- min Z dark: -0.49995995
- total mass: 0.23699994

