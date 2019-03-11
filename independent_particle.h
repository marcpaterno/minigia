#ifndef INDEPENDENT_PARTICLE_H
#define INDEPENDENT_PARTICLE_H
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
#include "populate.h"

class libff_drift {
public:
  double
  Length() const
  {
    return 2.1;
  }

  double
  getReferenceTime() const
  {
    return 0.345;
  }
};

// This is the type for propagator functions.
using prop_function = void(Bunch&, libff_drift const&);

inline double
invsqrt(double x)
{
  return 1.0 / sqrt(x);
}

// Use param_type<T>::type inside of a template when you want to
// pass arguments by const& when size of that type is large, and
// by value when the size is small.
// This is the most primitive of implementations: pass by value only
// for types that have explicit specializations.
template <typename T> struct param_type { using type = T const&; };
template <> struct param_type<double> { using type = double; };

template <typename T>
void
libff_drift_unit(T&      x,
                 T&      y,
                 T&      cdt,
                 typename param_type<T>::type xp,
                 typename param_type<T>::type yp,
                 typename param_type<T>::type dpop,
                 double length,
                 double reference_momentum,
                 double m,
                 double reference_time)
{
    T inv_npz   = invsqrt((dpop + 1.0) * (dpop + 1.0) - xp * xp - yp * yp);
    T lxpr      = xp * length * inv_npz;
    T lypr      = yp * length * inv_npz;
    T D2        = lxpr * lxpr + length * length + lypr * lypr;
    T p         = dpop * reference_momentum + reference_momentum;
    T E2        = p * p + m * m;
    T inv_beta2 = E2 / (p * p);
    x += lxpr;
    y += lypr;
    cdt += sqrt(D2 * inv_beta2) - reference_time;
}

void
propagate_orig(Bunch& bunch, libff_drift const& thelibff_drift)
{
  auto const        local_num          = bunch.get_local_num();
  Bunch::Particles& particles          = bunch.get_local_particles();
  auto              length             = thelibff_drift.Length();
  auto              reference_momentum = bunch.get_reference_particle().get_momentum();
  auto              m                  = bunch.get_mass();
  auto              reference_time     = thelibff_drift.getReferenceTime();

  for (Eigen::Index part = 0; part < local_num; ++part) {
    auto dpop(particles(part, Bunch::dpop));
    auto xp(particles(part, Bunch::xp));
    auto yp(particles(part, Bunch::yp));
    auto inv_npz = 1.0 / sqrt((dpop + 1.0) * (dpop + 1.0) - xp * xp - yp * yp);
    auto lxpr    = xp * length * inv_npz;
    auto lypr    = yp * length * inv_npz;
    auto D       = sqrt(lxpr * lxpr + length * length + lypr * lypr);
    auto p       = reference_momentum + dpop * reference_momentum;
    auto E       = sqrt(p * p + m * m);
    auto beta    = p / E;
    auto x(particles(part, Bunch::x));
    auto y(particles(part, Bunch::y));
    auto cdt(particles(part, Bunch::cdt));
    x += lxpr;
    y += lypr;
    cdt += D / beta - reference_time;
    particles(part, Bunch::x)   = x;
    particles(part, Bunch::y)   = y;
    particles(part, Bunch::cdt) = cdt;
  }
}

void
propagate_double(Bunch& bunch, libff_drift const& thelibff_drift)
{
  auto       local_num          = bunch.get_local_num();
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();
  double *RESTRICT xa, *RESTRICT xpa, *RESTRICT ya, *RESTRICT ypa, *RESTRICT cdta, *RESTRICT dpopa;
  bunch.set_arrays(xa, xpa, ya, ypa, cdta, dpopa);

  for (int part = 0; part < local_num; ++part) {
    auto x(xa[part]);
    auto xp(xpa[part]);
    auto y(ya[part]);
    auto yp(ypa[part]);
    auto cdt(cdta[part]);
    auto dpop(dpopa[part]);

    libff_drift_unit(x, y, cdt, xp, yp, dpop, length, reference_momentum, m, reference_time);

    xa[part]   = x;
    ya[part]   = y;
    cdta[part] = cdt;
  }
}

void
propagate_omp_simd(Bunch& bunch, libff_drift const& thelibff_drift)
{
  auto       local_num          = bunch.get_local_num();
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();
  double *RESTRICT xa, *RESTRICT xpa, *RESTRICT ya, *RESTRICT ypa, *RESTRICT cdta, *RESTRICT dpopa;
  bunch.set_arrays(xa, xpa, ya, ypa, cdta, dpopa);

#pragma omp simd
  for (int part = 0; part < local_num; ++part) {
    auto x(xa[part]);
    auto xp(xpa[part]);
    auto y(ya[part]);
    auto yp(ypa[part]);
    auto cdt(cdta[part]);
    auto dpop(dpopa[part]);

    libff_drift_unit(x, y, cdt, xp, yp, dpop, length, reference_momentum, m, reference_time);

    xa[part]   = x;
    ya[part]   = y;
    cdta[part] = cdt;
  }
}

void
propagate_omp_simd2(Bunch& bunch, libff_drift const& thelibff_drift)
{
  auto       local_num          = bunch.get_local_num();
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();
  double *RESTRICT xa, *RESTRICT xpa, *RESTRICT ya, *RESTRICT ypa, *RESTRICT cdta, *RESTRICT dpopa;
  bunch.set_arrays(xa, xpa, ya, ypa, cdta, dpopa);

#pragma omp simd
  for (int part = 0; part < local_num; ++part) {
    libff_drift_unit(
      xa[part], ya[part], cdta[part], xpa[part], ypa[part], dpopa[part], length, reference_momentum, m, reference_time);
  }
}

void
propagate_omp_simd3_nosimd(Bunch& bunch, libff_drift const& thelibff_drift)
{
  auto       local_num          = bunch.get_local_num();
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();
  auto&      particles(bunch.get_local_particles());

  for (Eigen::Index part = 0; part < local_num; ++part) {
    libff_drift_unit(particles(part, Bunch::x),
                     particles(part, Bunch::y),
                     particles(part, Bunch::cdt),
                     particles(part, Bunch::xp),
                     particles(part, Bunch::yp),
                     particles(part, Bunch::dpop),
                     length,
                     reference_momentum,
                     m,
                     reference_time);
  }
}

void
propagate_omp_simd3(Bunch& bunch, libff_drift const& thelibff_drift)
{
  auto       local_num          = bunch.get_local_num();
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();
  auto&      particles(bunch.get_local_particles());

#pragma omp simd
  for (Eigen::Index part = 0; part < local_num; ++part) {
    libff_drift_unit(particles(part, Bunch::x),
                     particles(part, Bunch::y),
                     particles(part, Bunch::cdt),
                     particles(part, Bunch::xp),
                     particles(part, Bunch::yp),
                     particles(part, Bunch::dpop),
                     length,
                     reference_momentum,
                     m,
                     reference_time);
  }
}

void
propagate_omp_simd3_2(Bunch& bunch, libff_drift const& thelibff_drift)
{
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();

  auto& particles(bunch.get_local_particles());
  auto  local_num = bunch.get_local_num();
#pragma omp parallel for simd num_threads(2)
  for (Eigen::Index part = 0; part < local_num; ++part) {
    libff_drift_unit(particles(part, Bunch::x),
                     particles(part, Bunch::y),
                     particles(part, Bunch::cdt),
                     particles(part, Bunch::xp),
                     particles(part, Bunch::yp),
                     particles(part, Bunch::dpop),
                     length,
                     reference_momentum,
                     m,
                     reference_time);
  }
}

void
propagate_omp_simd3_4(Bunch& bunch, libff_drift const& thelibff_drift)
{
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();

  auto& particles(bunch.get_local_particles());
  auto  local_num = bunch.get_local_num();
#pragma omp parallel for simd // num_threads(4)
  for (Eigen::Index part = 0; part < local_num; ++part) {
    libff_drift_unit(particles(part, Bunch::x),
                     particles(part, Bunch::y),
                     particles(part, Bunch::cdt),
                     particles(part, Bunch::xp),
                     particles(part, Bunch::yp),
                     particles(part, Bunch::dpop),
                     length,
                     reference_momentum,
                     m,
                     reference_time);
  }
}

void
propagate_omp_simd3_8(Bunch& bunch, libff_drift const& thelibff_drift)
{
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();

  auto& particles(bunch.get_local_particles());
  auto  local_num = bunch.get_local_num();
#pragma omp parallel for simd num_threads(8)
  for (Eigen::Index part = 0; part < local_num; ++part) {
    libff_drift_unit(particles(part, Bunch::x),
                     particles(part, Bunch::y),
                     particles(part, Bunch::cdt),
                     particles(part, Bunch::xp),
                     particles(part, Bunch::yp),
                     particles(part, Bunch::dpop),
                     length,
                     reference_momentum,
                     m,
                     reference_time);
  }
}

void
propagate_omp_simd3_16(Bunch& bunch, libff_drift const& thelibff_drift)
{
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();

  auto& particles(bunch.get_local_particles());
  auto  local_num = bunch.get_local_num();
#pragma omp parallel for simd num_threads(16)
  for (Eigen::Index part = 0; part < local_num; ++part) {
    libff_drift_unit(particles(part, Bunch::x),
                     particles(part, Bunch::y),
                     particles(part, Bunch::cdt),
                     particles(part, Bunch::xp),
                     particles(part, Bunch::yp),
                     particles(part, Bunch::dpop),
                     length,
                     reference_momentum,
                     m,
                     reference_time);
  }
}
void
propagate_omp_simd3_32(Bunch& bunch, libff_drift const& thelibff_drift)
{
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();

  auto& particles(bunch.get_local_particles());
  auto  local_num = bunch.get_local_num();
#pragma omp parallel for simd num_threads(32)
  for (Eigen::Index part = 0; part < local_num; ++part) {
    libff_drift_unit(particles(part, Bunch::x),
                     particles(part, Bunch::y),
                     particles(part, Bunch::cdt),
                     particles(part, Bunch::xp),
                     particles(part, Bunch::yp),
                     particles(part, Bunch::dpop),
                     length,
                     reference_momentum,
                     m,
                     reference_time);
  }
}

inline void
looper1(long local_num, std::function<void(long part)> f)
{
#pragma omp simd
  for (Eigen::Index part = 0; part < local_num; ++part) {
    f(part);
  }
}

void
propagate_lambda1(Bunch& bunch, libff_drift const& thelibff_drift)
{
  auto       local_num          = bunch.get_local_num();
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();
  auto&      particles(bunch.get_local_particles());

  looper1(local_num, [&](long part) {
    libff_drift_unit(particles(part, Bunch::x),
                     particles(part, Bunch::y),
                     particles(part, Bunch::cdt),
                     particles(part, Bunch::xp),
                     particles(part, Bunch::yp),
                     particles(part, Bunch::dpop),
                     length,
                     reference_momentum,
                     m,
                     reference_time);
  });
}

template <typename F>
void
looper2(long local_num, F&& f)
{
#pragma omp parallel for simd // num_threads(4)
  for (Eigen::Index part = 0; part < local_num; ++part) {
    f(part);
  }
}

void
propagate_lambda2(Bunch& bunch, libff_drift const& thelibff_drift)
{
  auto       local_num          = bunch.get_local_num();
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();
  auto&      particles(bunch.get_local_particles());

  auto propagate_through_drift_unit = [=,&particles](long part) {
      libff_drift_unit(particles(part, Bunch::x),
                       particles(part, Bunch::y),
                       particles(part, Bunch::cdt),
                       particles(part, Bunch::xp),
                       particles(part, Bunch::yp),
                       particles(part, Bunch::dpop),
                       length,
                       reference_momentum,
                       m,
                       reference_time);
  };

  looper2(local_num, propagate_through_drift_unit);
}

void
propagate_gsv(Bunch& bunch, libff_drift const& thelibff_drift)
{
  auto local_num = bunch.get_local_num();
  if (local_num % GSVector::size != 0) {
    throw std::runtime_error("local number of particles must be a multiple of GSVector::size");
  }
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();
  double *RESTRICT xa, *RESTRICT xpa, *RESTRICT ya, *RESTRICT ypa, *RESTRICT cdta, *RESTRICT dpopa;
  bunch.set_arrays(xa, xpa, ya, ypa, cdta, dpopa);
#pragma omp parallel for
  for (int part = 0; part < local_num; part += GSVector::size) {
    GSVector x(&xa[part]);
    GSVector xp(&xpa[part]);
    GSVector y(&ya[part]);
    GSVector yp(&ypa[part]);
    GSVector cdt(&cdta[part]);
    GSVector dpop(&dpopa[part]);

    libff_drift_unit(x, y, cdt, xp, yp, dpop, length, reference_momentum, m, reference_time);

    x.store(&xa[part]);
    y.store(&ya[part]);
    cdt.store(&cdta[part]);
  }
}

void
propagate_double_simpler(Bunch& bunch, libff_drift const& thelibff_drift)
{
  auto local_num = bunch.get_local_num();
  if (local_num % GSVector::size != 0) {
    throw std::runtime_error("local number of particles must be a multiple of GSVector::size");
  }
  const auto length             = thelibff_drift.Length();
  const auto reference_momentum = bunch.get_reference_particle().get_momentum();
  const auto m                  = bunch.get_mass();
  const auto reference_time     = thelibff_drift.getReferenceTime();
  double *RESTRICT xa, *RESTRICT xpa, *RESTRICT ya, *RESTRICT ypa, *RESTRICT cdta, *RESTRICT dpopa;
  bunch.set_arrays(xa, xpa, ya, ypa, cdta, dpopa);

  for (int part = 0; part < local_num; ++part) {
    libff_drift_unit(
      xa[part], ya[part], cdta[part], xpa[part], ypa[part], dpopa[part], length, reference_momentum, m, reference_time);
  }
}

#endif 
