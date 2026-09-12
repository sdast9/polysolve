#include "RobustArmijo.hpp"

#include <polysolve/Utils.hpp>

#include <spdlog/spdlog.h>

namespace polysolve::nonlinear::line_search
{
    RobustArmijo::RobustArmijo(const json &params, spdlog::logger &logger)
        : Superclass(params, logger)
    {
        delta_relative_tolerance = params.at(
            "/line_search/RobustArmijo/delta_relative_tolerance"_json_pointer);
    }

    bool RobustArmijo::criteria(
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const TVector &old_grad,
        const TVector &new_x,
        const double new_energy,
        const double step_size) const
    {
        if (new_energy <= old_energy + step_size * this->armijo_criteria) // Try Armijo first
            return true;

        if (std::abs(new_energy - old_energy) <= delta_relative_tolerance * std::abs(old_energy))
        {
            TVector new_grad;
            objFunc.gradient(new_x, new_grad);

            const double deltaE_approx = step_size / 2 * delta_x.dot(new_grad + old_grad);
            const double abs_eps_est = step_size / 2 * std::abs(delta_x.dot(new_grad - old_grad));

            //std::cout << "delta E: " << deltaE_approx << std::endl;
            //std::cout << "abs eps: " << abs_eps_est << std::endl;
            //std::cout << "armijo crit: " << this->armijo_criteria << std::endl;

            if (deltaE_approx + abs_eps_est <= step_size * this->armijo_criteria)
                return true;
        }

        // Neither the energy nor its gradient-integral estimate can resolve
        // the decrease (e.g. a full Newton step, where the estimate's error
        // term cancels its decrease exactly); fall back to the gradient norm.
        if (this->energy_at_roundoff(objFunc, use_grad_norm, old_energy, new_energy))
            return this->gradient_decreased(objFunc, old_grad, new_x);

        return false;
    }

}; // namespace polysolve::nonlinear::line_search
