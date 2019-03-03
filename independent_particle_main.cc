#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "bunch.h"
#include "bunch_data_paths.h"
#include "gsvector.h"
#include "independent_particle.h"
#include "populate.h"

const int    particles_per_rank   = 100000;
const double real_particles       = 1.0e12;

double
do_timing(prop_function*     propagator,
          const char*        name,
          Bunch&             bunch,
          libff_drift const& thelibff_drift,
          double             reference_timing,
          const int          rank)
{
  constexpr const int          num_runs  = 100;
  auto                         best_time = std::numeric_limits<double>::max();
  std::array<double, num_runs> times;
  for (size_t i = 0; i < num_runs; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    (*propagator)(bunch, thelibff_drift);
    const auto end  = std::chrono::high_resolution_clock::now();
    const auto time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    if (time < best_time) {
      best_time = time;
    }
    times[i] = time;
  }
  if (rank == 0) {
    std::cout << name << " best time = " << 1000 * best_time << std::endl;
    if (reference_timing > 0.0) {
      std::cout << name << " speedup = " << reference_timing / best_time << std::endl;
    }
  }
  return best_time;
}

void
run_check(prop_function* propagator, const char* name, libff_drift const& thelibff_drift, int size, int rank)
{
  const double tolerance = 1.0e-14;
  const int    num_test  = 104 * size;
  const double real_num  = 1.0e12;
  Bunch        b1(num_test * size, real_num, size, rank);
  Bunch        b2(num_test * size, real_num, size, rank);
  propagate_orig(b1, thelibff_drift);
  propagator(b2, thelibff_drift);
  if (!check_equal(b1, b2, tolerance)) {
    std::cerr << "run_check failed for " << name << std::endl;
  }
}

void
run(int rank, int size)
{
  Bunch bunch(particles_per_rank, real_particles, 1, 0);
  populate_gaussian(bunch);
  libff_drift thelibff_drift;

  run_check(&propagate_orig, "orig", thelibff_drift, size, rank);
  auto reference_timing = do_timing(&propagate_orig, "orig", bunch, thelibff_drift, 0.0, rank);

  run_check(&propagate_double, "optimized", thelibff_drift, size, rank);
  auto opt_timing = do_timing(&propagate_double, "optimized", bunch, thelibff_drift, reference_timing, rank);

  if (rank == 0) {
    std::cout << "GSVector::implementation = " << GSVector::implementation << std::endl;
  }

  run_check(&propagate_gsv, "vectorized", thelibff_drift, size, rank);
  do_timing(&propagate_gsv, "vectorized", bunch, thelibff_drift, opt_timing, rank);

  run_check(&propagate_double_simpler, "not manually vectorized", thelibff_drift, size, rank);
  do_timing(&propagate_double_simpler, "not manually vectorized", bunch, thelibff_drift, opt_timing, rank);

  run_check(&propagate_omp_simd, "omp simd", thelibff_drift, size, rank);
  do_timing(&propagate_omp_simd, "omp simd", bunch, thelibff_drift, opt_timing, rank);

  run_check(&propagate_omp_simd2, "omp simd2", thelibff_drift, size, rank);
  do_timing(&propagate_omp_simd2, "omp simd2", bunch, thelibff_drift, opt_timing, rank);

  run_check(&propagate_omp_simd3_nosimd, "omp simd3_nosimd", thelibff_drift, size, rank);
  do_timing(&propagate_omp_simd3_nosimd, "omp simd3_nosimd", bunch, thelibff_drift, opt_timing, rank);

  run_check(&propagate_omp_simd3, "omp simd3", thelibff_drift, size, rank);
  do_timing(&propagate_omp_simd3, "omp simd3", bunch, thelibff_drift, opt_timing, rank);

  run_check(&propagate_omp_simd3_2, "omp simd3_2", thelibff_drift, size, rank);
  do_timing(&propagate_omp_simd3_2, "omp simd3_2", bunch, thelibff_drift, opt_timing, rank);

  run_check(&propagate_omp_simd3_4, "omp simd3_4", thelibff_drift, size, rank);
  do_timing(&propagate_omp_simd3_4, "omp simd3_4", bunch, thelibff_drift, opt_timing, rank);

  run_check(&propagate_omp_simd3_8, "omp simd3_8", thelibff_drift, size, rank);
  do_timing(&propagate_omp_simd3_8, "omp simd3_8", bunch, thelibff_drift, opt_timing, rank);

  //  run_check(&propagate_omp_simd3_16, "omp simd3_16", thelibff_drift, size, rank);
  //  do_timing(&propagate_omp_simd3_16, "omp simd3_16", bunch, thelibff_drift, opt_timing, rank);

  //  run_check(&propagate_omp_simd3_32, "omp simd3_32", thelibff_drift, size, rank);
  //  do_timing(&propagate_omp_simd3_32, "omp simd3_32", bunch, thelibff_drift, opt_timing, rank);

  run_check(&propagate_lambda1, "lambda1", thelibff_drift, size, rank);
  do_timing(&propagate_lambda1, "lambda1", bunch, thelibff_drift, opt_timing, rank);

  run_check(&propagate_lambda2, "lambda2", thelibff_drift, size, rank);
  do_timing(&propagate_lambda2, "lambda2", bunch, thelibff_drift, opt_timing, rank);
}

int
main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int error, rank, size;
  error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  error = MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (error) {
    std::cerr << "MPI error" << std::endl;
    exit(error);
  }

  run(rank, size);
  MPI_Finalize();
  return 0;
}
