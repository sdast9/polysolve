// Validation of the secant pairs used by the BFGS strategies.

#pragma once

#include <polysolve/Utils.hpp>

#include <polysolve/nonlinear/Problem.hpp>

#include <map>
#include <string>

namespace polysolve::nonlinear
{
    /// @brief Which approximation a secant pair updates.
    enum class SecantForm
    {
        DIRECT, ///< the Hessian itself, stored densely
        INVERSE ///< the inverse Hessian, stored as limited-memory corrections
    };

    /// @brief What to do with a pair that does not carry positive curvature.
    enum class CurvaturePolicy
    {
        SKIP, ///< discard it and keep the previous approximation
        DAMP  ///< store the nearest pair that does, in the current metric
    };

    /// @brief Outcome of validating one secant pair.
    enum class PairVerdict
    {
        ACCEPTED,               ///< stored unchanged
        DAMPED,                 ///< stored after damping toward the approximation
        NON_FINITE_PAIR,        ///< s, y or one of their norms is not finite
        ZERO_DISPLACEMENT,      ///< s is zero, so the pair carries no curvature
        ZERO_GRADIENT_CHANGE,   ///< y is zero, so the initial scale is undefined
        INSUFFICIENT_CURVATURE, ///< s.y is not above the relative threshold
        INVALID_SCALE,          ///< a scale or denominator is not finite or not positive
        NON_FINITE_UPDATE       ///< the updated approximation is not finite
    };

    /// @brief Why a stored approximation was discarded.
    enum class ResetReason
    {
        NON_FINITE_DIRECTION,  ///< the approximation produced a non-finite direction
        NON_DESCENT_DIRECTION, ///< the approximation produced an ascent direction
        FACTORIZATION_FAILED,  ///< the approximation could not be factorized
        UNSUPPORTED_HISTORY    ///< too many consecutive pairs carried no usable curvature
    };

    std::string message(const PairVerdict verdict);
    std::string message(const ResetReason reason);

    /// @brief Validation and bookkeeping shared by the BFGS descent strategies.
    ///
    /// A BFGS approximation stays positive definite only while every stored
    /// pair satisfies s.y > 0 with finite scaling. An energy-decreasing step
    /// does not establish that, and none of the available line searches tests
    /// the Wolfe curvature condition, so pairs are validated here before they
    /// reach the history. This guard states no opinion on the line search, the
    /// stopping tolerances or the restart budget.
    class CurvatureGuard
    {
    public:
        using TVector = Problem::TVector;

        /// @param name strategy name, used for the parameters and the report
        /// @param form which approximation the pairs update
        /// @param params solver parameters; missing entries keep the defaults
        CurvatureGuard(const std::string &name,
                       const SecantForm form,
                       const json &params,
                       spdlog::logger &logger);

        CurvaturePolicy policy() const { return m_policy; }
        double tolerance() const { return m_tolerance; }

        /// @brief Validate a pair without reference to the current approximation
        /// @return ACCEPTED, or the reason the pair cannot be stored as it is
        PairVerdict classify(const TVector &s, const TVector &y) const;

        /// @brief Move the pair toward the curvature of the current approximation
        ///
        /// Powell damping: the stored pair gets s.y = damping * s'Bs in the
        /// direct form and damping * y'Hy in the inverse form, both positive
        /// while the approximation is.
        /// @param metric B*s in the direct form, H*y in the inverse form
        PairVerdict damp(TVector &s, TVector &y, const TVector &metric) const;

        /// @brief Record and log a verdict
        /// @return True when the pair may be stored
        bool store(const PairVerdict verdict, const TVector &s, const TVector &y);

        /// @brief Whether the retained approximation has gone stale
        ///
        /// A refused pair keeps the previous approximation, which is right for
        /// an isolated one. After a run of them nothing in the approximation
        /// comes from the stretch of the solve being worked on, and the
        /// strategy neither improves nor reports anything: it returns a stale
        /// descent direction the line search accepts, so the caller's own
        /// fallback is never reached. Records the reset and restarts the run.
        bool restart_required();

        /// @brief Record and log a discarded approximation
        void note_reset(const ResetReason reason);

        /// @brief Check the direction the approximation produced
        ///
        /// Records the reason and returns false when the caller must discard
        /// the approximation and fall back to steepest descent.
        bool direction_is_usable(const TVector &direction, const TVector &grad);

        void reset_counts();
        void update_solver_info(json &solver_info) const;

    private:
        const std::string m_name;
        const SecantForm m_form;
        spdlog::logger &m_logger;

        CurvaturePolicy m_policy;
        double m_tolerance;
        double m_damping;
        int m_restart;

        int m_consecutive_refusals = 0;
        int m_accepted = 0;
        int m_damped = 0;
        std::map<std::string, int> m_skipped;
        std::map<std::string, int> m_resets;
    };
} // namespace polysolve::nonlinear
