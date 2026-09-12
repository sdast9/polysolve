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
        /// Requires |ΔE| <= ε (max(1, S) + |E|), where S is the problem's
        /// characteristic energy scale only when use_grad_norm is flagged,
        /// and 1 otherwise. With ε=0, the flagged regime permits equal energy
        /// only; disabling both gates disables this fallback.
        bool energy_at_roundoff(
            const Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const double new_energy) const;

        /// @brief Fallback acceptance when energy_at_roundoff: the gradient
        /// norm must decrease. This is a progress measure within the energy
        /// bound, not a proof of descent for a nonconvex objective.
        bool gradient_decreased(
            Problem &objFunc,
            const TVector &old_grad,
            const TVector &new_x) const;

        double c;
        double armijo_criteria;    ///< cached value: c * delta_x.dot(old_grad)
        double roundoff_tolerance; ///< ε in the energy-change bound; 0 permits only flagged equal-energy fallback
    };
} // namespace polysolve::nonlinear::line_search
