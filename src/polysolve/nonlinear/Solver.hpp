#pragma once

#include "Criteria.hpp"
#include "descent_strategies/DescentStrategy.hpp"
// Line search methods
#include "line_search/LineSearch.hpp"

#include <limits>
#include <map>

namespace spdlog
{
    class logger;
}

namespace polysolve::nonlinear
{
    enum class FiniteDiffStrategy
    {
        NONE = 0,
        DIRECTIONAL_DERIVATIVE = 1,
        FULL_FINITE_DIFF = 2
    };

    class Solver
    {
    public:
        using Scalar = typename Problem::Scalar;
        using TVector = typename Problem::TVector;
        using THessian = typename Problem::THessian;

    public:
        // --- Static methods -------------------------------------------------

        // Static constructor
        //
        // @param[in]  solver   Solver type
        // @param[in]  precond  Preconditioner for iterative solvers
        //
        static std::unique_ptr<Solver> create(
            const json &solver_params,
            const json &linear_solver_params,
            const double characteristic_length,
            spdlog::logger &logger,
            const bool strict_validation = true,
            const NormType norm_type = NormType::EUCLIDEAN);

        /// @brief List available solvers
        static std::vector<std::string> available_solvers();

    public:
        /// @brief Construct a new Nonlinear Solver object
        /// @param solver_params JSON of solver parameters
        /// @param characteristic_length used to scale tolerances
        /// @param logger
        Solver(const json &solver_params,
               const double characteristic_length,
               spdlog::logger &logger);

        virtual ~Solver() = default;

        /// @brief Add a descent strategy to the solver
        /// @param s Descent strategy
        void add_strategy(const std::shared_ptr<DescentStrategy> &s) { m_strategies.push_back(s); }

        /// @brief Minimize the objective function
        /// @param objFunc Objective function
        /// @param x Initial guess
        void minimize(Problem &objFunc, TVector &x);

        // ====================================================================
        // Getters and setters
        // ====================================================================

        Criteria &stop_criteria() { return m_stop; }
        const Criteria &stop_criteria() const { return m_stop; }
        const Criteria &current_criteria() const { return m_current; }
        Status status() const { return m_status; }

        /// @brief Whether the strategy active at the last iteration solves with the Hessian
        ///
        /// See DescentStrategy::direction_solves_with_hessian: the
        /// direction-based stopping criteria apply only when this is true.
        bool direction_solves_with_hessian() const
        {
            return !m_strategies.empty() && m_strategies[m_descent_strategy]->direction_solves_with_hessian();
        }

        void set_strategies_iterations(const json &solver_params);
        void set_line_search(const json &params);
        const json &info() const { return solver_info; }

        void set_iteration_callback(std::function<bool(const Criteria &)> callback) { m_iteration_callback = callback; }

        /// @brief Set a filter applied to every computed update direction
        ///        before it is vetted and line-searched (e.g. one-sided
        ///        projection of constraint-violating components). The filter
        ///        does not redefine the objective or its gradient: descent and
        ///        slope tolerances use gradient.dot(filtered_direction).
        ///        This callback alone does not define a constrained merit
        ///        function or a constrained-stationarity stopping criterion.
        void set_direction_filter(std::function<void(const TVector &, TVector &)> filter) { m_direction_filter = filter; }

        /// @brief If true the solver will not throw an error if the maximum number of iterations is reached
        bool allow_out_of_iterations = false;

        /// @brief If true, converging via a non-gradient criterion (e.g., x/f delta tolerance) is logged as a success instead of an error
        bool allow_non_grad_convergence = false;

        /// @brief Get the line search object
        const std::shared_ptr<line_search::LineSearch> &line_search() const { return m_line_search; };

        virtual double compute_grad_norm(const Problem &objFunc, const TVector &x, const TVector &grad) const
        {
            return objFunc.grad_norm(grad, m_norm_type);
        }

    protected:
        /// @brief Compute direction in which the argument should be updated
        /// @param objFunc Problem to be minimized
        /// @param x Current input (n x 1)
        /// @param grad Gradient at current step (n x 1)
        /// @param[out] direction Current update direction (n x 1)
        /// @return True if update direction was found, False otherwises
        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction)
        {
            return m_strategies[m_descent_strategy]->compute_update_direction(objFunc, x, grad, direction);
        }

        /// @brief Whether a step longer than the direction itself can be admissible.
        ///
        /// A line search that grows the step (alpha > 1) relies on the problem's
        /// feasibility callbacks to bound it. A solver whose directions end at a
        /// boundary the problem does not report -- a box bound built into the
        /// direction -- must say so here.
        virtual bool direction_admits_growth() const { return true; }

        /// @brief Whether the direction-based stopping criteria (the slope
        /// tolerance, x_delta_tol, rel_x_delta_tol) may end the solve now
        ///
        /// Only while a strategy that solves with the Hessian is active (see
        /// DescentStrategy::direction_solves_with_hessian).
        virtual bool direction_based_stops_allowed() const { return direction_solves_with_hessian(); }

        void reset_stopping_criteria(Problem &objFunc, NormType norm_type)
        {
            m_stop_rescaled.reset();
            m_stop_rescaled.iterations = m_stop.iterations;
            m_stop_rescaled.xDelta = m_stop.xDelta * objFunc.step_norm_rescaling(norm_type);
            m_stop_rescaled.fDelta = m_stop.fDelta * objFunc.energy_norm_rescaling(norm_type);
            m_stop_rescaled.gradNorm = m_stop.gradNorm * objFunc.grad_norm_rescaling(norm_type);
            m_stop_rescaled.firstGradNorm = m_stop.firstGradNorm * objFunc.grad_norm_rescaling(norm_type);
            m_stop_rescaled.xDeltaDotGrad = m_stop.xDeltaDotGrad * objFunc.energy_norm_rescaling(norm_type);
            m_stop_rescaled.relGradNorm = m_stop.relGradNorm;
            m_stop_rescaled.relXDelta = m_stop.relXDelta;
            m_stop_rescaled.newtonDecrement = m_stop.newtonDecrement * objFunc.energy_norm_rescaling(norm_type);
        }

        /// @brief Stopping criteria
        Criteria m_stop;

        Criteria m_stop_rescaled;

        /// @brief Current criteria

        Criteria m_current;
        /// @brief Current status
        Status m_status = Status::NotStarted;

        /// @brief Index into m_strategies
        int m_descent_strategy;

        /// @brief Objective generations the problem went through in this
        ///        minimization, reported in the solver info
        int m_objective_changes = 0;

        /// @brief Logger to use
        spdlog::logger &m_logger;

        NormType m_norm_type;

        // ====================================================================
        //                        Finite Difference Utilities
        // ====================================================================

        FiniteDiffStrategy gradient_fd_strategy = FiniteDiffStrategy::NONE;
        double gradient_fd_eps = 1e-7;
        /// @brief Check gradient versus finite difference results
        /// @param objFunc Problem defining relevant objective function
        /// @param x Current input (n x 1)
        /// @param grad Current gradient (n x 1)
        virtual void verify_gradient(Problem &objFunc, const TVector &x, const TVector &grad) final;

    private:
        // ====================================================================
        //                        Solver parameters
        // ====================================================================

        const double characteristic_length;

        // ====================================================================
        //                           Solver state
        // ====================================================================

        /// @brief Reset the solver at the start of a minimization
        /// @param ndof number of degrees of freedom
        void reset(const int ndof);

        /// Record a change between configured descent strategies. Internal
        /// L-BFGS steepest-descent restarts are strategy diagnostics instead;
        /// keeping the two separate is essential when diagnosing fallback
        /// scale jumps.
        void record_strategy_transition(
            const std::string &from,
            const std::string &to,
            const std::string &reason,
            const json &details = json::object());

        std::string descent_strategy_name() const { return m_strategies[m_descent_strategy]->name(); };

        std::shared_ptr<line_search::LineSearch> m_line_search;
        std::vector<std::shared_ptr<DescentStrategy>> m_strategies;

        std::vector<int> m_iter_per_strategy;

        std::function<bool(const Criteria &)> m_iteration_callback = nullptr;

        std::function<void(const TVector &, TVector &)> m_direction_filter = nullptr;

        bool m_iteration_diagnostics_enabled = false;
        json m_last_iteration_diagnostics = nullptr;
        json m_strategy_transition_events = json::array();
        json m_pending_strategy_transitions = json::array();
        std::map<std::string, int> m_strategy_transition_counts;
        double m_previous_accepted_direction_norm = std::numeric_limits<double>::quiet_NaN();

        // ====================================================================
        //                            Solver info
        // ====================================================================

        /// @brief Update solver info JSON object
        /// @param energy
        void update_solver_info(const double energy);

        /// @brief Reset timing members to 0
        void reset_times();

        /// @brief Log time taken in different phases of the solve
        void log_times() const;

        json solver_info;

        // Timers
        double total_time;
        double obj_fun_time;
        double grad_time;
        double update_direction_time;
        double line_search_time;
        double constraint_set_update_time;
        double diagnostics_time;

        // ====================================================================
        //                                 END
        // ====================================================================
    };
} // namespace polysolve::nonlinear
