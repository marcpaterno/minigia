#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "bunch.h"
#include "bunch_data_paths.h"
// We define GSV_AVX in order to get the GSVector implementation that uses AVX.
#define GSV_AVX 1
#include "gsvector.h"
#include "populate.h"
#include "independent_particle.h"
#include "benchmark/benchmark.h"

//constexpr int particles_per_rank = 10 * 1000;
constexpr int min_particles = 64;
constexpr int max_particles = min_particles * 64 * 1024;

constexpr double real_particles = 1.e12;

#define BMARK(name) BENCHMARK(name)->RangeMultiplier(4)->Range(min_particles, max_particles)

static void BM_propagate_orig(benchmark::State &state) {
    Bunch bunch(state.range(0), real_particles, 1, 0);
    populate_gaussian(bunch);
    libff_drift thelibff_drift;
    for (auto _ : state) {
        propagate_orig(bunch, thelibff_drift);
        benchmark::DoNotOptimize(bunch);
        benchmark::DoNotOptimize(thelibff_drift);
    }
}
BMARK(BM_propagate_orig);

static void BM_propagate_gsv(benchmark::State &state) {
    Bunch bunch(state.range(0), real_particles, 1, 0);
    populate_gaussian(bunch);
    libff_drift thelibff_drift;
    for (auto _ : state) {
        propagate_gsv(bunch, thelibff_drift);
        benchmark::DoNotOptimize(bunch);
        benchmark::DoNotOptimize(thelibff_drift);
    }
}
BMARK(BM_propagate_gsv);

static void BM_propagate_lambda2(benchmark::State &state) {
    Bunch bunch(state.range(0), real_particles, 1, 0);
    populate_gaussian(bunch);
    libff_drift thelibff_drift;
    for (auto _ : state) {
        propagate_lambda2(bunch, thelibff_drift);
        benchmark::DoNotOptimize(bunch);
        benchmark::DoNotOptimize(thelibff_drift);
    }
}
BMARK(BM_propagate_lambda2);

static void BM_propagate_omp_simd3_4(benchmark::State& state) {
    Bunch bunch(state.range(0), real_particles, 1, 0);
    populate_gaussian(bunch);
    libff_drift thelibff_drift;
    for (auto _ : state) {
        propagate_omp_simd3_4(bunch, thelibff_drift);
        benchmark::DoNotOptimize(bunch);
        benchmark::DoNotOptimize(thelibff_drift);
    }
}
BMARK(BM_propagate_omp_simd3_4);

BENCHMARK_MAIN();
