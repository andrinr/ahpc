# Advanced High Performance Computing

## Files

HDF5 is a file format designed to store and organize large amounts of data. It is a hierarchical data format, meaning that it can store data in a tree-like structure. HDF5 is a binary format, meaning that it is not human-readable. HDF5 files are typically given the extension `.h5` or `.hdf5`.

Use od to view the contents of any file:

    od -j <offset> -N <bytes> -t <type> <file>

where type could be f8 or d8 for double precision floating point numbers, f4 or d4 for single precision floating point numbers, i4 for 32-bit integers, i8 for 64-bit integers, etc.

- big endian: most significant byte first
- little endian: least significant byte first (most common)

add hpccourse23@gmail.com to repo


