# Exercise 1


## 1.1 

- ``blitz`` is the namespace
- ``blitz::Array`` invokes the constructor whereas 10, 8 and 6 are the arguments
- ``<float, 3>`` corresponds to the template parameters

## 1.2

This line creates a tensor of size three where the corresponding dimensions are 10, 8 and 6 where the dataype of each entry is float. According to the blitz++ documentation, the tensor is initialized with zeros.

```cpp
#include  <iostream 

int  main() {
    std::cout  << "data" << std::endl;
}
```

## 2.



4096^3 = 2^36 

2^36 * 56 bytes = 2^4 * 56 Gbytes
1GB = 2^32 bytes

2^4 * 56 / 64 = 14 nodes

## 3. (1 - 3)

- ``od -j 0 -N 8 -t f8 --endian=big b0-final.std`` => duration 1.0000000000000107
- ``od -j 8 -N 4 -t u4 --endian=big b0-final.std`` => 158095 particles

## 4. 

total per particle byte offset: 32 + 4 = 36
header offset 28 + 4 = 32

- For the mass we use the offset: 32 + 100 * 36 = 3640. The command is `` od -j 3640 -N 4 -t f4 --endian=big b0-final.std `` and the result: -0.031605203

- X positions: offset 32 + 100 * 36 + 4. The command is `` od -j 3644 -N 4 -t f4  --endian=big b0-final.std `` and the result: 2.9017775e+18

- Y positions: offset 32 + 100 * 36 + 4 * 2. The command is `` od -j 3648 -N 4 -t f4 --endian=big b0-final.std `` and the result: -0.036164634

- Z positions: offset 32 + 100 * 36 + 4 * 3. The command is `` od -j 3652 -N 4 -t f4 --endian=big b0-final.std `` and the result: -0.047118336

## 5.


max X dark 0.49999925
max Y dark 0.499731
max Z dark 0.4999042
min X dark -0.4999119
min Y dark -0.49984062
min Z dark -0.49995995
total mass 0.23699994

