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