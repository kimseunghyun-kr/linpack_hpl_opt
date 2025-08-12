#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = -1, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[256];
    if (gethostname(host, sizeof(host)) != 0) {
        std::snprintf(host, sizeof(host), "unknown");
    }

    std::printf("Hello from rank %d/%d on %s\n", rank, size, host);
    std::fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

