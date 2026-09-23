// L-BFGS solver (Using the LBFGSpp under MIT License).

#pragma once

#include "CurvatureGuard.hpp"
#include "DescentStrategy.hpp"
#include <polysolve/Utils.hpp>

#include <LBFGSpp/BFGSMat.h>

#include <polysolve/linear/Solver.hpp>

#include <deque>
#include <map>
#include <memory>

namespace polysolve::nonlinear
{
    class LBFGS : public DescentStrategy
    {
    public:
        using Superclass = DescentStrategy;

        LBFGS(const json &solver_params,
              const double characteristic_length,
              spdlog::logger &logger);

        /// EXPERIMENT (qn-contact): with linear solver settings, so that the
        /// opt-in "Hessian" preconditioner can factorize a lagged Hessian.
        LBFGS(const json &solver_params,
              const json &linear_solver_params,
              const double characteristic_length,
              spdlog::logger &logger);

        std::string name() const override { return "L-BFGS"; }

    public:
        void reset(const int ndof) override;

        bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;

        /// The stored iterate and gradient were taken from the objective
        /// that has just been superseded, so no pair may be formed across it.
        void objective_changed(const int ndof) override
        {
            m_guard.note_reset(ResetReason::OBJECTIVE_CHANGED);
            reset(ndof);
        }

        void reset_times() override
        {
            m_guard.reset_counts();
            m_direction_sources.clear();
        }
        void update_solver_info(json &solver_info, const double per_iteration) override
        {
            m_guard.update_solver_info(solver_info);
            solver_info["direction_sources"][name()] = m_direction_sources;
            if (m_preconditioner != Preconditioner::NONE)
                solver_info["lbfgs_preconditioner"] = {
                    {"kind", preconditioner_name()},
                    {"refreshes", m_precond_refreshes},
                    {"factorization_failures", m_precond_failures},
                    {"assembly_seconds", m_precond_assembly_time},
                    {"factorization_seconds", m_precond_factor_time},
                    {"apply_seconds", m_precond_apply_time}};
        }
        json diagnostics() const override { return m_last_diagnostics; }

    private:
        // ---- EXPERIMENT (qn-contact): preconditioned two-loop recursion ----
        enum class Preconditioner
        {
            NONE,     ///< the unchanged LBFGSpp path, scalar initial matrix
            DIAGONAL, ///< H0 = diag(projected Hessian)^-1, lagged
            HESSIAN   ///< H0 = (projected Hessian)^-1, factorized, lagged
        };
        Preconditioner m_preconditioner = Preconditioner::NONE;
        int m_precond_refresh = 0; ///< iterations between refreshes; 0 = on reset only
        /// refresh when the previous accepted step was shorter than this
        /// fraction of its direction (truncated or backtracked); 0 = off
        double m_refresh_short_step = 0;
        double m_prev_direction_norm = -1;
        int m_short_step_refreshes = 0;
        int m_iters_since_refresh = 0;
        bool m_precond_valid = false;
        TVector m_diag_inv;
        std::unique_ptr<polysolve::linear::Solver> m_linear_solver;
        std::deque<std::pair<TVector, TVector>> m_pairs; ///< (s, y), oldest first
        int m_precond_refreshes = 0;
        int m_precond_failures = 0;
        double m_precond_assembly_time = 0;
        double m_precond_factor_time = 0;
        double m_precond_apply_time = 0;

        std::string preconditioner_name() const;
        void refresh_preconditioner(Problem &objFunc, const TVector &x);
        void apply_initial(const TVector &q, TVector &r);
        /// out = H * v with the two-loop recursion over m_pairs
        void apply_inverse_hessian(const TVector &v, TVector &out);
        bool compute_preconditioned_direction(
            Problem &objFunc, const TVector &x, const TVector &grad, TVector &direction);
        // ---------------------------------------------------------------------

        LBFGSpp::BFGSMat<Scalar> m_bfgs; // Approximation to the Hessian matrix

        /// Validates the secant pairs before they reach the history
        CurvatureGuard m_guard;

        /// The number of corrections to approximate the inverse Hessian matrix.
        /// The L-BFGS routine stores the computation results of previous \ref m
        /// iterations to approximate the inverse Hessian matrix of the current
        /// iteration. This parameter controls the size of the limited memories
        /// (corrections). The default value is \c 6. Values less than \c 3 are
        /// not recommended. Large values will result in excessive computing time.
        int m_history_size;

        TVector m_prev_x;    // Previous x
        TVector m_prev_grad; // Previous gradient

        /// Source and scale of the last direction, plus cumulative source
        /// counts for distinguishing an L-BFGS history direction from either
        /// kind of steepest-descent fallback.
        json m_last_diagnostics = json::object();
        std::map<std::string, int> m_direction_sources;
    };
} // namespace polysolve::nonlinear
