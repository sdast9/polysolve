// Validation of the secant pairs used by the BFGS strategies.

#include "CurvatureGuard.hpp"

#include <cmath>

namespace polysolve::nonlinear
{
    namespace
    {
        /// Parameters may be nested under the strategy name, as in the string
        /// form of the solver, or given directly, as in its list form. A
        /// missing entry keeps the default so the strategies stay constructible
        /// without the injected defaults of the specification.
        json lookup(const json &params, const std::string &name, const std::string &key)
        {
            if (params.contains(name) && params[name].is_object() && params[name].contains(key))
                return params[name][key];
            if (params.contains(key))
                return params[key];
            return json();
        }
    } // namespace

    std::string message(const PairVerdict verdict)
    {
        switch (verdict)
        {
        case PairVerdict::ACCEPTED:
            return "accepted";
        case PairVerdict::DAMPED:
            return "damped";
        case PairVerdict::NON_FINITE_PAIR:
            return "non_finite_pair";
        case PairVerdict::ZERO_DISPLACEMENT:
            return "zero_displacement";
        case PairVerdict::ZERO_GRADIENT_CHANGE:
            return "zero_gradient_change";
        case PairVerdict::INSUFFICIENT_CURVATURE:
            return "insufficient_curvature";
        case PairVerdict::INVALID_SCALE:
            return "invalid_scale";
        case PairVerdict::NON_FINITE_UPDATE:
            return "non_finite_update";
        default:
            return "unknown";
        }
    }

    std::string message(const ResetReason reason)
    {
        switch (reason)
        {
        case ResetReason::NON_FINITE_DIRECTION:
            return "non_finite_direction";
        case ResetReason::NON_DESCENT_DIRECTION:
            return "non_descent_direction";
        case ResetReason::FACTORIZATION_FAILED:
            return "factorization_failed";
        case ResetReason::UNSUPPORTED_HISTORY:
            return "unsupported_history";
        case ResetReason::OBJECTIVE_CHANGED:
            return "objective_changed";
        default:
            return "unknown";
        }
    }

    CurvatureGuard::CurvatureGuard(const std::string &name,
                                   const SecantForm form,
                                   const json &params,
                                   spdlog::logger &logger)
        : m_name(name), m_form(form), m_logger(logger),
          m_policy(CurvaturePolicy::SKIP), m_tolerance(1e-8), m_damping(0.2), m_restart(1)
    {
        const json policy = lookup(params, name, "curvature_policy");
        if (!policy.is_null())
        {
            const std::string value = policy;
            if (value == "Skip")
                m_policy = CurvaturePolicy::SKIP;
            else if (value == "Damp")
                m_policy = CurvaturePolicy::DAMP;
            else
                log_and_throw_error(m_logger, "{} curvature_policy must be Skip or Damp, instead got {}", name, value);
        }

        const json tolerance = lookup(params, name, "curvature_tolerance");
        if (!tolerance.is_null())
            m_tolerance = tolerance;
        if (!(m_tolerance >= 0) || !std::isfinite(m_tolerance))
            log_and_throw_error(m_logger, "{} curvature_tolerance must be finite and >=0, instead got {}", name, m_tolerance);

        const json damping = lookup(params, name, "curvature_damping");
        if (!damping.is_null())
            m_damping = damping;
        if (!(m_damping > 0) || !(m_damping < 1))
            log_and_throw_error(m_logger, "{} curvature_damping must be in (0, 1), instead got {}", name, m_damping);

        const json restart = lookup(params, name, "curvature_restart");
        if (!restart.is_null())
            m_restart = restart;
        if (m_restart < 0)
            log_and_throw_error(m_logger, "{} curvature_restart must be >=0, instead got {}", name, m_restart);
    }

    PairVerdict CurvatureGuard::classify(const TVector &s, const TVector &y) const
    {
        if (!s.allFinite() || !y.allFinite())
            return PairVerdict::NON_FINITE_PAIR;

        // The squared norms are the first intermediates that can overflow.
        const double ss = s.squaredNorm();
        const double yy = y.squaredNorm();
        if (!std::isfinite(ss) || !std::isfinite(yy))
            return PairVerdict::NON_FINITE_PAIR;

        if (ss == 0)
            return PairVerdict::ZERO_DISPLACEMENT;
        if (yy == 0)
            return PairVerdict::ZERO_GRADIENT_CHANGE;

        const double sy = s.dot(y);
        if (!std::isfinite(sy))
            return PairVerdict::NON_FINITE_PAIR;

        // Relative and scale aware: invariant under a rescaling of either the
        // objective or the variables, unlike an absolute curvature bound.
        if (!(sy > m_tolerance * std::sqrt(ss) * std::sqrt(yy)))
            return PairVerdict::INSUFFICIENT_CURVATURE;

        // |y|^2 / s.y is the initial scale of the limited-memory approximation
        // and the magnitude of the dense rank-one term; positive curvature
        // alone does not keep it representable.
        if (!std::isfinite(yy / sy))
            return PairVerdict::INVALID_SCALE;

        return PairVerdict::ACCEPTED;
    }

    PairVerdict CurvatureGuard::damp(TVector &s, TVector &y, const TVector &metric) const
    {
        const PairVerdict verdict = classify(s, y);
        if (verdict != PairVerdict::INSUFFICIENT_CURVATURE)
            return verdict; // damping only repairs the curvature itself

        if (!metric.allFinite())
            return PairVerdict::INVALID_SCALE;

        // s'Bs in the direct form, y'Hy in the inverse form
        const double scale = m_form == SecantForm::DIRECT ? s.dot(metric) : y.dot(metric);
        const double sy = s.dot(y);
        if (!std::isfinite(scale) || scale <= 0)
            return PairVerdict::INVALID_SCALE;
        if (sy >= m_damping * scale)
            return verdict; // already at the damped curvature; damping cannot help

        const double phi = (1 - m_damping) * scale / (scale - sy);
        if (!std::isfinite(phi))
            return PairVerdict::INVALID_SCALE;

        if (m_form == SecantForm::DIRECT)
            y = phi * y + (1 - phi) * metric;
        else
            s = phi * s + (1 - phi) * metric;

        const PairVerdict damped = classify(s, y);
        return damped == PairVerdict::ACCEPTED ? PairVerdict::DAMPED : damped;
    }

    bool CurvatureGuard::store(const PairVerdict verdict, const TVector &s, const TVector &y)
    {
        if (verdict == PairVerdict::ACCEPTED)
        {
            ++m_accepted;
            m_consecutive_refusals = 0;
            return true;
        }
        if (verdict == PairVerdict::DAMPED)
        {
            ++m_damped;
            m_consecutive_refusals = 0;
            m_logger.debug(
                "[{}] damped secant pair ({}={:g}, {}={:g}, {}={:g})", m_name,
                "s" + log::dot() + "y", s.dot(y), log::norm("s"), s.norm(), log::norm("y"), y.norm());
            return true;
        }

        ++m_skipped[message(verdict)];
        ++m_consecutive_refusals;
        m_logger.debug(
            "[{}] secant pair skipped: {} ({}={:g}, {}={:g}, {}={:g}); keeping the previous approximation",
            m_name, message(verdict), "s" + log::dot() + "y", s.dot(y),
            log::norm("s"), s.norm(), log::norm("y"), y.norm());
        return false;
    }

    bool CurvatureGuard::restart_required()
    {
        if (m_restart <= 0 || m_consecutive_refusals < m_restart)
            return false;
        m_logger.debug(
            "[{}] {} consecutive secant pairs carried no usable curvature", m_name, m_consecutive_refusals);
        note_reset(ResetReason::UNSUPPORTED_HISTORY);
        m_consecutive_refusals = 0;
        return true;
    }

    void CurvatureGuard::note_reset(const ResetReason reason)
    {
        ++m_resets[message(reason)];
        m_logger.debug("[{}] discarding the approximation: {}", m_name, message(reason));
    }

    bool CurvatureGuard::direction_is_usable(const TVector &direction, const TVector &grad)
    {
        if (!direction.allFinite())
        {
            note_reset(ResetReason::NON_FINITE_DIRECTION);
            return false;
        }
        // At a stationary point the direction is zero and there is nothing to
        // repair; elsewhere the approximation must produce actual descent.
        if (grad.squaredNorm() != 0 && !(direction.dot(grad) < 0))
        {
            note_reset(ResetReason::NON_DESCENT_DIRECTION);
            return false;
        }
        return true;
    }

    void CurvatureGuard::reset_counts()
    {
        m_consecutive_refusals = 0;
        m_accepted = 0;
        m_damped = 0;
        m_skipped.clear();
        m_resets.clear();
    }

    void CurvatureGuard::update_solver_info(json &solver_info) const
    {
        int skipped = 0;
        for (const auto &[reason, count] : m_skipped)
            skipped += count;
        int resets = 0;
        for (const auto &[reason, count] : m_resets)
            resets += count;

        json report;
        report["policy"] = m_policy == CurvaturePolicy::SKIP ? "Skip" : "Damp";
        report["tolerance"] = m_tolerance;
        report["restart"] = m_restart;
        report["accepted_pairs"] = m_accepted;
        report["damped_pairs"] = m_damped;
        report["skipped_pairs"] = skipped;
        report["skipped"] = m_skipped;
        report["history_resets"] = resets;
        report["resets"] = m_resets;

        solver_info["curvature_guard"][m_name] = report;
    }
} // namespace polysolve::nonlinear
