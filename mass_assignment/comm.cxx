#include <mpi.h> // MPI
#include <fftw3-mpi.h> // MPI FFTW
#include "comm.h"

Communicator::Communicator() {
    errs = 0;

    MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided);
    
    MPI_Is_thread_main( &flag );
    if (!flag) {
        errs++;
        printf( "This thread called init_thread but Is_thread_main gave false\n" );
        fflush(stdout);
    }

    MPI_Query_thread( &claimed );
    if (claimed != provided) {
        errs++;
        printf( "Query thread gave thread level %d but Init_thread gave %d\n", claimed, provided );
        fflush(stdout);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &np );
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

Communicator::~Communicator() {
    MPI_Finalize();
}