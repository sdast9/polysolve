#include "Backtracking.hpp"

#include <polysolve/Utils.hpp>

#include <spdlog/spdlog.h>

#include <limits>

namespace polysolve::nonlinear::line_search
{

    Backtracking::Backtracking(const json &params, spdlog::logger &logger)
        : Superclass(params, logger)
    {
    }

    double Backtracking::compute_descent_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const TVector &old_grad,
        const double starting_step_size)
    {
        double step_size = starting_step_size;

        init_compute_descent_step_size(delta_x, old_grad);

        // A strategy may hand over a direction that is not one of descent
        // (ADAM makes no descent promise, so the solver does not screen its
        // directions). No step along it satisfies a sufficient-decrease
        // condition, and trying each step size only spends evaluations and
        // risks the roundoff fallback accepting an uphill step: fail the
        // search and let the solver's strategy fallback take over.
        if (!admits_direction())
        {
            const double slope = delta_x.dot(old_grad);
            m_logger.log(final_strategy() ? spdlog::level::warn : spdlog::level::debug,
                         "[{}] Line search refused a direction with slope {}={:g} (not a descent direction)",
                         name(), log::delta("x") + log::dot() + "g", slope);
            mutable_diagnostics()["rejected_direction"] = {{"reason", std::isfinite(slope) ? "ascent" : "nonfinite_slope"},
                                                           {"slope", std::isfinite(slope) ? json(slope) : json(nullptr)}};
            return std::numeric_limits<double>::quiet_NaN();
        }

        for (; step_size > current_min_step_size() && cur_iter < current_max_step_size_iter(); step_size *= step_ratio, ++cur_iter)
        {
            const TVector new_x = x + step_size * delta_x;

            try
            {
                POLYSOLVE_SCOPED_STOPWATCH("solution changed - constraint set update in LS", constraint_set_update_time, m_logger);
                objFunc.solution_changed(new_x);
            }
            catch (const std::runtime_error &e)
            {
                m_logger.warn("Failed to take step due to \"{}\", reduce step size...", e.what());
                continue;
            }

            if (!objFunc.is_step_valid(x, new_x))
            {
                continue;
            }

            const double new_energy = objFunc(new_x);

            if (!std::isfinite(new_energy))
            {
                continue;
            }

            m_logger.trace("ls it: {} {}: {}, {}: {}", cur_iter, log::delta("E"), new_energy - old_energy, log::delta("x"), objFunc.step_norm(delta_x, norm_type));

            if (criteria(delta_x, objFunc, use_grad_norm, old_energy, old_grad, new_x, new_energy, step_size))
            {
                break; // found a good step size
            }
        }

        return step_size;
    }

    bool Backtracking::criteria(
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const TVector &old_grad,
        const TVector &new_x,
        const double new_energy,
        const double step_size) const
    {
        if (use_grad_norm)
        {
            TVector new_grad;
            objFunc.gradient(new_x, new_grad);
            return objFunc.grad_norm(new_grad, norm_type) < objFunc.grad_norm(old_grad, norm_type); // TODO cache old grad norm
        }
        return new_energy < old_energy;
    }

} // namespace polysolve::nonlinear::line_search
