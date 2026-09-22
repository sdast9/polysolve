// L-BFGS solver (Using the LBFGSpp under MIT License).

#include "BFGS.hpp"

#include <cmath>

namespace polysolve::nonlinear
{

    BFGS::BFGS(const json &solver_params,
               const json &linear_solver_params,
               const double characteristic_length,
               spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger),
          m_guard(name(), SecantForm::DIRECT, solver_params, logger)
    {
        linear_solver = polysolve::linear::Solver::create(linear_solver_params, logger);
        if (!linear_solver->is_dense())
            log_and_throw_error(
                logger, "BFGS linear solver must be dense, instead got {}: BFGS factorizes a dense n x n "
                        "approximation of the Hessian; use a dense solver such as Eigen::LDLT (or L-BFGS, which needs none)",
                linear_solver->name());
    }

    void BFGS::reset(const int ndof)
    {
        Superclass::reset(ndof);
        reset_history(ndof);
        m_last_diagnostics = json::object();
    }

    void BFGS::reset_history(const int ndof)
    {
        m_prev_x.resize(0);
        m_prev_grad.resize(ndof);

        hess.setIdentity(ndof, ndof);
    }

    bool BFGS::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        std::string direction_source;
        if (m_prev_x.size() == 0)
        {
            direction = -grad;
            direction_source = "steepest_descent_initial_or_reset";
        }
        else
        {
            TVector y = grad - m_prev_grad;
            TVector s = x - m_prev_x;

            // B*s is both the damping metric and the second denominator.
            TVector Bs = hess * s;

            // The approximation is only positive definite while every stored
            // pair has positive curvature with representable denominators,
            // which an energy-decreasing step does not establish.
            PairVerdict verdict = m_guard.classify(s, y);
            if (verdict == PairVerdict::INSUFFICIENT_CURVATURE
                && m_guard.policy() == CurvaturePolicy::DAMP)
                verdict = m_guard.damp(s, y, Bs);

            Eigen::MatrixXd updated;
            if (verdict == PairVerdict::ACCEPTED || verdict == PairVerdict::DAMPED)
            {
                // Damping changes y, so both denominators are taken from the
                // pair that would actually be stored, and the update is formed
                // aside: a non-finite one must not reach the approximation.
                const double y_s = y.dot(s);
                const double sBs = s.dot(Bs);
                if (!Bs.allFinite() || !std::isfinite(sBs) || sBs <= 0)
                    verdict = PairVerdict::INVALID_SCALE;
                else
                {
                    updated = hess + (y * y.transpose()) / y_s - (Bs * Bs.transpose()) / sBs;
                    if (!updated.allFinite())
                        verdict = PairVerdict::NON_FINITE_UPDATE;
                }
            }

            // Incorporate the latest accepted displacement before computing
            // the direction at x; otherwise the Hessian is one update stale.
            const bool stored = m_guard.store(verdict, s, y);
            bool restarted = false;
            if (stored)
                hess = std::move(updated);
            else if (m_guard.restart_required())
            {
                hess.setIdentity(x.size(), x.size()); // nothing in it is current
                restarted = true;
            }

            try
            {
                linear_solver->analyze_pattern_dense(hess, hess.rows());
                linear_solver->factorize_dense(hess);
                linear_solver->solve(-grad, direction);
            }
            catch (const std::runtime_error &err)
            {
                m_logger.debug("Unable to factorize Hessian: \"{}\";", err.what());
                m_guard.note_reset(ResetReason::FACTORIZATION_FAILED);
                hess.setIdentity(x.size(), x.size());
                m_last_diagnostics = {
                    {"direction_source", "factorization_failed"},
                    {"uses_steepest_descent", false},
                    {"secant_pair", m_guard.last_pair()}};
                ++m_direction_sources["factorization_failed"];
                return false;
            }

            direction_source = restarted ? "steepest_descent_curvature_restart"
                               : stored  ? "dense_bfgs"
                                         : "dense_bfgs_after_pair_skip";

            // A dense factorization of an indefinite matrix succeeds and
            // returns an ascent direction, so the direction itself is checked.
            if (!m_guard.direction_is_usable(direction, grad))
            {
                hess.setIdentity(x.size(), x.size());
                direction = -grad;
                direction_source = "steepest_descent_invalid_hessian_direction";
            }
        }

        m_last_diagnostics = {
            {"direction_source", direction_source},
            {"uses_steepest_descent", direction_source.rfind("steepest_descent", 0) == 0},
            {"secant_pair", m_guard.last_pair()}};
        ++m_direction_sources[direction_source];

        m_prev_x = x;
        m_prev_grad = grad;

        return direction.allFinite();
    }
} // namespace polysolve::nonlinear
