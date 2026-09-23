// L-BFGS solver (Using the LBFGSpp under MIT License).

#include "LBFGS.hpp"

#include <polysolve/Utils.hpp>

#include <chrono>
#include <cmath>

namespace polysolve::nonlinear
{
    LBFGS::LBFGS(const json &solver_params,
                 const double characteristic_length,
                 spdlog::logger &logger)
        : Superclass(solver_params,
                     characteristic_length,
                     logger),
          m_guard(name(), SecantForm::INVERSE, solver_params, logger)
    {
        m_history_size = extract_param("L-BFGS", "history_size", solver_params);
        // EXPERIMENT (qn-contact): with a preconditioner, history_size 0 is the
        // lagged-Hessian (modified Newton) control without secant corrections.
        const bool preconditioned = solver_params.contains("L-BFGS")
                                    && solver_params["L-BFGS"].value("preconditioner", std::string("None")) != "None";
        if (m_history_size < (preconditioned ? 0 : 1))
            log_and_throw_error(logger, "L-BFGS history_size must be >=1, instead got {}", m_history_size);
    }

    LBFGS::LBFGS(const json &solver_params,
                 const json &linear_solver_params,
                 const double characteristic_length,
                 spdlog::logger &logger)
        : LBFGS(solver_params, characteristic_length, logger)
    {
        const json &opts = solver_params.contains("L-BFGS") ? solver_params["L-BFGS"] : json::object();
        const std::string kind = opts.value("preconditioner", std::string("None"));
        if (kind == "None")
            m_preconditioner = Preconditioner::NONE;
        else if (kind == "Diagonal")
            m_preconditioner = Preconditioner::DIAGONAL;
        else if (kind == "Hessian")
            m_preconditioner = Preconditioner::HESSIAN;
        else
            log_and_throw_error(logger, "L-BFGS preconditioner must be None, Diagonal or Hessian, instead got {}", kind);
        m_precond_refresh = opts.value("preconditioner_refresh", 0);
        if (m_preconditioner == Preconditioner::HESSIAN)
        {
            m_linear_solver = polysolve::linear::Solver::create(linear_solver_params, logger);
            if (m_linear_solver->is_dense())
                log_and_throw_error(logger, "L-BFGS Hessian preconditioner needs a sparse linear solver, instead got {}", m_linear_solver->name());
        }
    }

    void LBFGS::reset(const int ndof)
    {
        Superclass::reset(ndof);

        m_bfgs.reset(ndof, m_history_size);
        m_prev_x.resize(0);
        m_last_diagnostics = json::object();
        m_pairs.clear();
        m_precond_valid = false;
    }

    std::string LBFGS::preconditioner_name() const
    {
        switch (m_preconditioner)
        {
        case Preconditioner::DIAGONAL:
            return "Diagonal";
        case Preconditioner::HESSIAN:
            return "Hessian";
        default:
            return "None";
        }
    }

    void LBFGS::refresh_preconditioner(Problem &objFunc, const TVector &x)
    {
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        polysolve::StiffnessMatrix H;
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, H);
        auto t1 = clock::now();
        m_precond_assembly_time += std::chrono::duration<double>(t1 - t0).count();

        bool factorized = false;
        if (m_preconditioner == Preconditioner::HESSIAN)
        {
            try
            {
                m_linear_solver->analyze_pattern(H, H.rows());
                m_linear_solver->factorize(H);
                factorized = true;
            }
            catch (const std::runtime_error &err)
            {
                ++m_precond_failures;
                m_logger.debug("[L-BFGS] Hessian preconditioner factorization failed (\"{}\"); using its diagonal", err.what());
            }
        }
        if (!factorized)
        {
            // Diagonal (also the fallback of a failed factorization)
            TVector d = H.diagonal();
            double mean_pos = 0;
            int npos = 0;
            for (int i = 0; i < d.size(); ++i)
                if (std::isfinite(d[i]) && d[i] > 0)
                {
                    mean_pos += d[i];
                    ++npos;
                }
            mean_pos = npos > 0 ? mean_pos / npos : 1.0;
            m_diag_inv.resize(d.size());
            for (int i = 0; i < d.size(); ++i)
                m_diag_inv[i] = 1.0 / ((std::isfinite(d[i]) && d[i] > 0) ? d[i] : mean_pos);
        }
        else
            m_diag_inv.resize(0);
        m_precond_factor_time += std::chrono::duration<double>(clock::now() - t1).count();
        m_precond_valid = true;
        m_iters_since_refresh = 0;
        ++m_precond_refreshes;
    }

    void LBFGS::apply_initial(const TVector &q, TVector &r)
    {
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        if (m_diag_inv.size() == q.size())
            r = m_diag_inv.cwiseProduct(q);
        else
        {
            r.resize(q.size());
            m_linear_solver->solve(q, r);
        }
        m_precond_apply_time += std::chrono::duration<double>(clock::now() - t0).count();
    }

    void LBFGS::apply_inverse_hessian(const TVector &v, TVector &out)
    {
        const int k = int(m_pairs.size());
        std::vector<double> alpha(k), rho(k);
        TVector q = v;
        for (int i = k - 1; i >= 0; --i)
        {
            const auto &[s, y] = m_pairs[i];
            rho[i] = 1.0 / s.dot(y);
            alpha[i] = rho[i] * s.dot(q);
            q -= alpha[i] * y;
        }
        apply_initial(q, out);
        for (int i = 0; i < k; ++i)
        {
            const auto &[s, y] = m_pairs[i];
            const double beta = rho[i] * y.dot(out);
            out += (alpha[i] - beta) * s;
        }
    }

    bool LBFGS::compute_preconditioned_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        std::string direction_source;
        if (!m_precond_valid || (m_precond_refresh > 0 && m_iters_since_refresh >= m_precond_refresh))
            refresh_preconditioner(objFunc, x);
        ++m_iters_since_refresh;

        bool stored = false;
        if (m_prev_x.size() != 0)
        {
            TVector s = x - m_prev_x;
            TVector y = grad - m_prev_grad;
            PairVerdict verdict = m_guard.classify(s, y);
            if (verdict == PairVerdict::INSUFFICIENT_CURVATURE
                && m_guard.policy() == CurvaturePolicy::DAMP)
            {
                TVector Hy;
                apply_inverse_hessian(y, Hy);
                verdict = m_guard.damp(s, y, Hy);
            }
            stored = m_guard.store(verdict, s, y);
            if (stored)
            {
                m_pairs.emplace_back(s, y);
                while (int(m_pairs.size()) > m_history_size)
                    m_pairs.pop_front();
            }
            else if (m_guard.restart_required())
                m_pairs.clear();
        }

        TVector Hg;
        apply_inverse_hessian(grad, Hg);
        direction = -Hg;
        direction_source = m_pairs.empty() ? "preconditioned_initial" : (stored ? "preconditioned_limited_memory" : "preconditioned_after_pair_skip");

        if (!m_guard.direction_is_usable(direction, grad))
        {
            m_pairs.clear();
            // the preconditioner alone (a PSD-projected Hessian or its diagonal)
            apply_initial(grad, Hg);
            direction = -Hg;
            direction_source = "preconditioned_invalid_history_direction";
            if (!m_guard.direction_is_usable(direction, grad))
            {
                direction = -grad;
                direction_source = "steepest_descent_invalid_preconditioner";
            }
        }

        m_last_diagnostics = {
            {"direction_source", direction_source},
            {"uses_steepest_descent", direction_source.rfind("steepest_descent", 0) == 0},
            {"history_corrections", int(m_pairs.size())},
            {"hessian_initial_scale", nullptr},
            {"inverse_hessian_initial_scale", nullptr},
            {"preconditioner", preconditioner_name()},
            {"preconditioner_refreshes", m_precond_refreshes},
            {"secant_pair", m_guard.last_pair()}};
        ++m_direction_sources[direction_source];

        m_prev_x = x;
        m_prev_grad = grad;
        return direction.allFinite();
    }

    bool LBFGS::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (m_preconditioner != Preconditioner::NONE)
            return compute_preconditioned_direction(objFunc, x, grad, direction);

        std::string direction_source;
        if (m_prev_x.size() == 0)
        {
            // Use gradient descent in the first iteration or if the previous iteration failed
            direction = -grad;
            direction_source = "steepest_descent_initial_or_reset";
        }
        else
        {
            // Update s and y
            // s_{i+1} = x_{i+1} - x_i
            // y_{i+1} = g_{i+1} - g_i
            assert(m_prev_x.size() == x.size());
            assert(m_prev_grad.size() == grad.size());
            TVector s = x - m_prev_x;
            TVector y = grad - m_prev_grad;

            // The approximation is only positive definite while every stored
            // pair has positive curvature with a representable scale, which an
            // energy-decreasing step does not establish.
            PairVerdict verdict = m_guard.classify(s, y);
            if (verdict == PairVerdict::INSUFFICIENT_CURVATURE
                && m_guard.policy() == CurvaturePolicy::DAMP)
            {
                TVector Hy;
                m_bfgs.apply_Hv(y, Scalar(1), Hy);
                verdict = m_guard.damp(s, y, Hy);
            }

            const bool stored = m_guard.store(verdict, s, y);
            bool restarted = false;
            if (stored)
                m_bfgs.add_correction(s, y);
            else if (m_guard.restart_required())
            {
                m_bfgs.reset(x.size(), m_history_size); // nothing in it is current
                restarted = true;
            }

            // Recursive formula to compute d = -H * g
            m_bfgs.apply_Hv(grad, -Scalar(1), direction);
            if (m_bfgs.num_corrections() > 0)
                direction_source = stored ? "limited_memory" : "limited_memory_after_pair_skip";
            else if (restarted)
                direction_source = "steepest_descent_curvature_restart";
            else
                direction_source = "steepest_descent_identity";

            if (!m_guard.direction_is_usable(direction, grad))
            {
                // Keeping a poisoned history would repeat the failure; restart
                // the approximation from the safe steepest-descent direction.
                m_bfgs.reset(x.size(), m_history_size);
                direction = -grad;
                direction_source = "steepest_descent_invalid_history_direction";
            }
        }

        const auto finite_or_null = [](const double value) {
            return std::isfinite(value) ? json(value) : json(nullptr);
        };
        const double theta = m_bfgs.theta();
        m_last_diagnostics = {
            {"direction_source", direction_source},
            {"uses_steepest_descent", direction_source.rfind("steepest_descent", 0) == 0},
            {"history_corrections", m_bfgs.num_corrections()},
            {"hessian_initial_scale", finite_or_null(theta)},
            {"inverse_hessian_initial_scale", finite_or_null(1. / theta)},
            {"secant_pair", m_guard.last_pair()}};
        ++m_direction_sources[direction_source];

        m_prev_x = x;
        m_prev_grad = grad;

        return direction.allFinite();
    }
} // namespace polysolve::nonlinear
