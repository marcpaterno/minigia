#ifndef SPACE_CHARGE_3D_OPEN_HOCKNEY_H_
#define SPACE_CHARGE_3D_OPEN_HOCKNEY_H_

#include "bunch.h"
#include "rectangular_grid_domain.h"
#include "rectangular_grid.h"
#include "distributed_rectangular_grid.h"
#include "commxx.h"
#include "commxx_divider.h"
#include "distributed_fft3d.h"

/// Note: internal grid is stored in [z][y][x] order, but
/// grid shape expects [x][y][z] order.
class Space_charge_3d_open_hockney
{
private:
    std::vector<int > grid_shape, doubled_grid_shape, padded_grid_shape;
    Rectangular_grid_domain_sptr domain_sptr, doubled_domain_sptr;
    Distributed_fft3d_sptr distributed_fft3d_sptr;
    Commxx_divider_sptr commxx_divider_sptr;
    Commxx_sptr comm2_sptr, comm1_sptr;
    std::vector<int > lowers1, lengths1;
    int real_lower, real_upper, real_length;
    std::vector<int > real_lengths;
    int doubled_lower, doubled_upper;
    int real_doubled_lower, real_doubled_upper;
    double n_sigma;
    bool domain_fixed;
    bool have_domains;
    void
    constructor_common(std::vector<int > const& grid_shape);
    void
    set_doubled_domain();
public:
    Space_charge_3d_open_hockney(Commxx_divider_sptr commxx_divider_sptr,
            std::vector<int > const & grid_shape,
            double n_sigma = 8.0);
    /// Note: Use Space_charge_3d_open_hockney::get_internal_grid_shape for
    /// Distributed_fft3d.
    void
    setup_communication(Commxx_sptr const& bunch_comm_sptr);
    void
    setup_derived_communication();
    double
    get_n_sigma() const;
    void
    update_domain(Bunch const& bunch);
    Rectangular_grid_domain const&
    get_domain() const
    {
        return *domain_sptr;
    }
    /// Returns global charge density on doubled grid in [C/m^3]
    Distributed_rectangular_grid_sptr
    get_global_charge_density2_allreduce(
            Rectangular_grid const& local_charge_density, Commxx_sptr comm_sptr);
    /// Returns local charge density on original grid in [C/m^3]
    Rectangular_grid_sptr
    get_local_charge_density(Bunch const& bunch);
    /// Returns global charge density on doubled grid in [C/m^3]
    Distributed_rectangular_grid_sptr
    get_global_charge_density2(Rectangular_grid const& local_charge_density,
            Commxx_sptr comm_sptr);
    /// Returns Green function on the doubled grid in [1/m^3]
    Distributed_rectangular_grid_sptr
    get_green_fn2_pointlike();
    Distributed_rectangular_grid_sptr
    get_scalar_field2(Distributed_rectangular_grid & charge_density22,
            Distributed_rectangular_grid & green_fn2);
    Distributed_rectangular_grid_sptr
    extract_scalar_field(Distributed_rectangular_grid const& scalar_field2);
    /// Returns component of electric field [V/m]
    /// @param scalar_field the scalar field [V]
    /// @param component which component (0=x, 1=y, 2=z)
    Distributed_rectangular_grid_sptr
    get_electric_field_component(
            Distributed_rectangular_grid const& scalar_field, int component);
    Rectangular_grid_sptr
    get_global_electric_field_component_allreduce(
            Distributed_rectangular_grid const& dist_field);
    Rectangular_grid_sptr
    get_global_electric_field_component(
            Distributed_rectangular_grid const& dist_field);
    void
    apply_kick(Bunch & bunch, Rectangular_grid const& En, double delta_tau,
            int component);
    void
    apply(Bunch & bunch, double time_step, int verbosity);
    virtual
    ~Space_charge_3d_open_hockney();
};

#endif /* SPACE_CHARGE_3D_OPEN_HOCKNEY_H_ */
