#pragma once

#include <polysolve/Utils.hpp>

#include <polysolve/nonlinear/Problem.hpp>

namespace polysolve::nonlinear
{

    class DescentStrategy
    {
    public:
        using TVector = Problem::TVector;
        using Scalar = Problem::Scalar;

        /// @brief Constructor
        /// @param solver_params_ JSON of solver parameters
        /// @param characteristic_length
        /// @param logger
        DescentStrategy(const json &solver_params_,
                        const double characteristic_length,
                        spdlog::logger &logger)
            : m_logger(logger)
        {
        }
        virtual ~DescentStrategy() {}
        virtual std::string name() const = 0;

        virtual void reset(const int ndof) {}
        virtual void reset_times() {}

        /// @brief Discard whatever the strategy holds from a superseded objective
        ///
        /// Called when the problem reports a different objective generation
        /// than the one the strategy's stored iterate and gradient belong to.
        /// A strategy that keeps no such history does nothing.
        virtual void objective_changed(const int ndof) {}

        /// @brief Update solver info after finding descent direction
        /// @param solver_info JSON of solver parameters
        /// @param per_iteration Number of iterations (used to normalize timings)
        virtual void update_solver_info(json &solver_info, const double per_iteration) {}
        virtual void log_times() const {}

        /// @brief Observational state of the direction most recently computed.
        ///
        /// Used only by the opt-in nonlinear iteration diagnostics. Strategies
        /// return an empty object when they have no strategy-specific state to
        /// report; the solver supplies the common direction and line-search
        /// quantities.
        virtual json diagnostics() const { return json::object(); }

        virtual bool is_direction_descent() { return true; }
        virtual bool handle_error() { return false; }

        /// @brief Compute descent direction along which to do line search
        /// @param objFunc Problem to be minimized
        /// @param x Current input (n x 1)
        /// @param grad Current gradient (n x 1)
        /// @param direction Current descent direction (n x 1)
        /// @return True if a descent direction was successfully found
        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) = 0;

    protected:
        spdlog::logger &m_logger;
    };
} // namespace polysolve::nonlinear
