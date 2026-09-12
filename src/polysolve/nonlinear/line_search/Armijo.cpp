

#include "Armijo.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

namespace polysolve::nonlinear::line_search
{
    Armijo::Armijo(const json &params, spdlog::logger &logger)
        : Superclass(params, logger)
    {
        c = params["line_search"]["Armijo"]["c"];
        roundoff_tolerance = params["line_search"]["Armijo"]["roundoff_tolerance"];
    }

    void Armijo::init_compute_descent_step_size(
        const TVector &delta_x,
        const TVector &old_grad)
    {
        armijo_criteria = c * delta_x.dot(old_grad);
        assert(armijo_criteria <= 0);
    }

    bool Armijo::criteria(
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const TVector &old_grad,
        const TVector &new_x,
        const double new_energy,
        const double step_size) const
    {
        if (new_energy <= old_energy + step_size * armijo_criteria)
            return true;

        // The Armijo decrease is below what the energy evaluation can
        // resolve; measure progress on the gradient instead.
        if (energy_at_roundoff(objFunc, use_grad_norm, old_energy, new_energy))
            return gradient_decreased(objFunc, old_grad, new_x);

        return false;
    }

    bool Armijo::energy_at_roundoff(
        const Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const double new_energy) const
    {
        if (!use_grad_norm && !(roundoff_tolerance > 0))
            return false;

        // A small gradient alone does not make an uphill step safe: it can
        // decrease on the way to a local maximum. Keep a finite energy bound.
        // Near stationarity, cancellation can leave a tiny total energy even
        // though the terms being summed have the problem's characteristic
        // energy scale. Use that existing scale only for the small-gradient
        // regime; retain the energy-value bound elsewhere.
        const double energy_scale = use_grad_norm ? objFunc.energy_norm_rescaling(norm_type) : 1.;
        if (!std::isfinite(energy_scale) || !(energy_scale > 0))
            return false;
        const double bound = roundoff_tolerance * std::max(1., energy_scale)
                             + roundoff_tolerance * std::abs(old_energy);
        return std::isfinite(bound)
               && std::abs(new_energy - old_energy) <= bound;
    }

    bool Armijo::gradient_decreased(
        Problem &objFunc,
        const TVector &old_grad,
        const TVector &new_x) const
    {
        TVector new_grad;
        objFunc.gradient(new_x, new_grad);
        if (!new_grad.array().isFinite().all())
            return false;
        const double old_norm = objFunc.grad_norm(old_grad, norm_type);
        const double new_norm = objFunc.grad_norm(new_grad, norm_type);
        if (new_norm < old_norm)
        {
            m_logger.trace("ls it: {} energy at roundoff; accepted on gradient norm {} -> {}", cur_iter, old_norm, new_norm);
            return true;
        }
        return false;
    }

}; // namespace polysolve::nonlinear::line_search
