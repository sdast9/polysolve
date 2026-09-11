#pragma once

#include "Backtracking.hpp"

namespace polysolve::nonlinear::line_search
{
    class Armijo : public Backtracking
    {
    public:
        using Superclass = Backtracking;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        Armijo(const json &params, spdlog::logger &logger);

        virtual std::string name() const override { return "Armijo"; }

    protected:
        virtual void init_compute_descent_step_size(
            const TVector &delta_x,
            const TVector &old_grad) override;

        virtual bool criteria(
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const TVector &old_grad,
            const TVector &new_x,
            const double new_energy,
            const double step_size) const override;

        /// @brief Whether the energy cannot resolve the Armijo decrease.
        ///
        /// True when the caller flagged use_grad_norm (‖∇f‖ already below
        /// use_grad_norm_tol times the problem's gradient scale, the switch
        /// Backtracking honors) or when the measured change is within
        /// roundoff of the energy value itself, |ΔE| ≤ ε(1 + |E|).
        bool energy_at_roundoff(
            const bool use_grad_norm,
            const double old_energy,
            const double new_energy) const;

        /// @brief Fallback acceptance when energy_at_roundoff: the gradient
        /// norm must decrease. Never accepts a step along which the gradient
        /// grows, so ascent that the energy cannot see is still rejected.
        bool gradient_decreased(
            Problem &objFunc,
            const TVector &old_grad,
            const TVector &new_x) const;

        double c;
        double armijo_criteria;    ///< cached value: c * delta_x.dot(old_grad)
        double roundoff_tolerance; ///< ε in |ΔE| ≤ ε(1 + |E|); 0 disables the roundoff regime
    };
} // namespace polysolve::nonlinear::line_search
