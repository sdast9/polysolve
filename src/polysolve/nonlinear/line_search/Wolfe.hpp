#pragma once

#include "RobustArmijo.hpp"

#include <map>

namespace polysolve::nonlinear::line_search
{
    /// @brief Strong Wolfe line search that respects the problem's feasibility.
    ///
    /// Bracket-and-zoom on phi(alpha) = f(x + alpha dx) (Nocedal & Wright,
    /// Numerical Optimization, Algorithms 3.5 and 3.6), accepting a step with
    /// the Armijo decrease phi(alpha) <= phi(0) + c alpha phi'(0) (Armijo/c)
    /// and the strong curvature condition |phi'(alpha)| <= c2 |phi'(0)|.
    ///
    /// Feasibility. The search starts from the step the finite-energy and
    /// feasibility caps left. It grows beyond it only when neither cap
    /// shortened it, the solver's direction admits growth, and the problem
    /// validates the longer interval: every growth first rebuilds the swept
    /// interval (line_search_extend) and prices it (max_step_size) and only
    /// then evaluates anything there. No trial ever lies outside an interval
    /// the problem has priced.
    ///
    /// Energy roundoff. Where phi(alpha) is within eps_k = approximate_wolfe_epsilon
    /// |phi(0)| of phi(0) -- or within the roundoff bound RobustArmijo's
    /// fallback uses (Armijo::energy_at_roundoff) -- energies cannot be
    /// compared, so such a trial is judged on its slope alone, as in Hager and
    /// Zhang's approximate Wolfe conditions (SIAM J. Optim. 16(1), 2005): it is
    /// accepted on the strong curvature condition and otherwise placed in the
    /// bracket by the sign of its slope. The energy of a contact problem is a
    /// sum of terms far larger than itself, so its evaluation noise can exceed
    /// the roundoff bound by orders of magnitude.
    ///
    /// When no Wolfe point is reachable the search says which way it ended:
    /// - capped: an Armijo point whose slope is still steeper than c2 allows,
    ///   at a boundary growth cannot cross -- the feasible bound, the growth
    ///   limit, a start the caps shortened, a direction that admits no growth,
    ///   or a longer sweep the problem refused;
    /// - budget / interval collapsed: the evaluation budget ran out or the
    ///   bracket shrank to nothing; the best decrease point is accepted
    ///   (armijo_* or, if its energy was unresolved, flat_*);
    /// - fallback: no trial met the Armijo decrease (or the direction is not
    ///   one of descent), so RobustArmijo backtracking from the same start, on
    ///   its own budget, decides exactly as it would have alone.
    /// A step accepted without the curvature condition carries no guarantee
    /// that its secant pair has positive curvature; the strategies' curvature
    /// safeguard decides whether that pair enters their approximation.
    ///
    /// Objective changes. If the problem's objective generation changes during
    /// the search, every stored energy and slope belongs to a superseded
    /// function. The search restarts from x under the new one, at most
    /// max_objective_restarts times, and fails otherwise.
    class Wolfe : public RobustArmijo
    {
    public:
        using Superclass = RobustArmijo;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        Wolfe(const json &params, spdlog::logger &logger);

        virtual std::string name() const override { return "Wolfe"; }

        void update_solver_info(json &solver_info, const double per_iteration) override;
        void reset_times() override;

    protected:
        double compute_descent_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const TVector &old_grad,
            const double starting_step_size) override;

    private:
        /// One point of phi. valid means the step was taken, is valid and has
        /// a finite energy; the slope is evaluated only where it is needed.
        struct Trial
        {
            double alpha = 0;
            double phi = std::numeric_limits<double>::quiet_NaN();
            double dphi = std::numeric_limits<double>::quiet_NaN();
            bool valid = false;
            bool has_slope = false;
            bool flat = false; ///< phi is within the energy's roundoff of phi(0)
        };

        /// How one bracket-and-zoom pass ended.
        struct Result
        {
            std::string outcome;
            double alpha = std::numeric_limits<double>::quiet_NaN();
            bool objective_changed = false;
            bool fallback = false;
        };

        Result bracket_and_zoom(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double starting_step_size);

        Trial evaluate(const TVector &x, const TVector &delta_x, Problem &objFunc, const double alpha);
        void evaluate_slope(const TVector &x, const TVector &delta_x, Problem &objFunc, Trial &trial);

        /// A trial point for the zoom between lo (an Armijo point with a slope)
        /// and hi, safeguarded to stay inside the interval.
        static double interpolate(const Trial &lo, const Trial &hi);

        /// Leave the problem at the accepted step.
        Result accept(const TVector &x, const TVector &delta_x, Problem &objFunc,
                      const Trial &trial, const std::string &outcome);

        double search(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const TVector &old_grad,
            const double starting_step_size);

        double c2;
        double growth_factor;
        double growth_limit;
        int max_evaluations;
        int max_objective_restarts;
        double approximate_epsilon;

        // State of the current search, anchored at x under one objective.
        double f0;
        double dphi0;
        uint64_t generation;
        double swept;          ///< [0, swept] is the interval the problem has priced
        bool boundary_reached; ///< swept is a feasible bound, not a choice of the search
        std::string no_growth; ///< why growth is unavailable, empty if it is
        double last_evaluated; ///< the step the problem's state is at, NaN if unknown
        int evaluations;
        int gradient_evaluations;
        int growth_sweeps;
        int objective_restarts;
        double accepted_slope_ratio; ///< phi'(alpha)/phi'(0) at the accepted Wolfe-phase step, NaN if unmeasured

        std::map<std::string, int> outcome_counts;
        int total_evaluations = 0;
        int total_growth_sweeps = 0;
    };
} // namespace polysolve::nonlinear::line_search
