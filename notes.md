# Advanced High Performance Computing

## Files

HDF5 is a file format designed to store and organize large amounts of data. It is a hierarchical data format, meaning that it can store data in a tree-like structure. HDF5 is a binary format, meaning that it is not human-readable. HDF5 files are typically given the extension `.h5` or `.hdf5`.

Use od to view the contents of any file:

    od -j <offset> -N <bytes> -t <type> <file>

where type could be f8 or d8 for double precision floating point numbers, f4 or d4 for single precision floating point numbers, i4 for 32-bit integers, i8 for 64-bit integers, etc.

- big endian: most significant byte first
- little endian: least significant byte first (most common)

add hpccourse23@gmail.com to repo


## Stack vs. Heap Memory

In c and c++, the stack is managed by the compiler, whereas the heap is managed by the programmer using the `malloc` and `free` functions. 

When we run a function in c or c++, the compiler allocates a block of memory on the stack for the function to use. This block of memory is called the stack frame. The stack frame contains the function's local variables, as well as the return address of the function. The stack frame is automatically deallocated when the function returns.

Usually the stack runs from the highest memory address to the lowest memory address, whereas the heap runs from the lowest abailble memory address to the highest memory address. Multiple threads do not share the same stack, but they do share the same heap.

## Testing

MTBF (mean time between failures) is the average time between failures of a system. MTBF is a measure of reliability. The higher the MTBF, the more reliable the system.

Given a program that runs on a laptop error free for a year, it will only do so for a couple hours on a supercomputer.

