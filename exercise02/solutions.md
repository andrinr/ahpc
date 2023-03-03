# Sol

## 1

Adding two arrays using the ``+`` operator results in a component-wise addition of the two arrays. The result is a new array with the same shape as the two operands. 


## 2

- Line 6 sets all entires in the array to seven.

- Line 9 will set all elements in the given ranges of array A to seven, as B is a subarray of A and not a copy.


## 3

```cpp
#include "blitz/array.h"

int  main() {
    blitz::GeneralArrayStorage <3> storage;
    //  modify  storage  here so that  the  first  dimension  starts  at 10
    // the  second  and  third  dimension  should  start  at 0
    storage.base() = 10, 0, 0;

    blitz::Array <int ,3> A(5, 20, 20, storage ); 
    std::cout << A(11, 0, 0) << std::endl;
    std::cout << A(0, 0, 0) << std::endl;
}
```

## 4

```cpp