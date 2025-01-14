#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "papi.h"
#include "papi_test.h"
#include "do_loops.h"

int main(int argc, char **argv) {
  int retval, i;
  int quiet = 0;
  char *region_name;
  int world_size, world_rank;

  /* Set TESTS_QUIET variable */
  quiet = tests_quiet(argc, argv);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  region_name = "do_flops";

//#pragma omp parallel
//#pragma omp for
  for (i = 1; i <= 2; ++i) {
    int tid;
    tid = omp_get_thread_num();

    if (!quiet) {
      printf("\nRank %d, Thread %d: instrument flops\n", world_rank, tid);
    }

    retval = PAPI_hl_region_begin(region_name);
    if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_hl_region_begin", retval);
    }

    do_flops(NUM_FLOPS);

    retval = PAPI_hl_region_end(region_name);
    if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_hl_region_end", retval);
    }

  }

  MPI_Finalize();
  test_hl_pass(__FILE__);

  return 0;
}