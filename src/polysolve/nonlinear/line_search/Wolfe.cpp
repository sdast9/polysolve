#include "Wolfe.hpp"

#include <polysolve/Utils.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cfenv>
#include <cmath>

namespace polysolve::nonlinear::line_search
{
    namespace
    {
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

        json finite_or_null(const double value)
        {
            return std::isfinite(value) ? json(value) : json(nullptr);
        }
    } // namespace

    Wolfe::Wolfe(const json &params, spdlog::logger &logger)
        : Superclass(params, logger)
    {
        const json &p = params.at("line_search").at("Wolfe");
        c2 = p.at("c2");
        growth_factor = p.at("growth_factor");
        growth_limit = p.at("growth_limit");
        max_evaluations = p.at("max_evaluations");
        max_objective_restarts = p.at("max_objective_restarts");
        approximate_epsilon = p.at("approximate_wolfe_epsilon");

        // 0 <= c < c2 < 1 is what makes a strong Wolfe point exist on every
        // interval where phi is smooth and bounded below.
        if (!(c < c2 && c2 < 1))
            log_and_throw_error(m_logger, "Wolfe line search requires Armijo/c < Wolfe/c2 < 1 (c={}, c2={})", c, c2);
        if (!(growth_factor > 1))
            log_and_throw_error(m_logger, "Wolfe line search requires Wolfe/growth_factor > 1 (got {})", growth_factor);
        if (!(approximate_epsilon >= 0))
            log_and_throw_error(m_logger, "Wolfe line search requires Wolfe/approximate_wolfe_epsilon >= 0 (got {})", approximate_epsilon);
        if (!(max_evaluations >= 1))
            log_and_throw_error(m_logger, "Wolfe line search requires Wolfe/max_evaluations >= 1 (got {})", max_evaluations);
    }

    double Wolfe::compute_descent_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const TVector &old_grad,
        const double starting_step_size)
    {
        try
        {
            return search(x, delta_x, objFunc, use_grad_norm, old_energy, old_grad, starting_step_size);
        }
        catch (...)
        {
            // A growth sweep is this search's own; end the interval so nothing
            // downstream mistakes it for a current one, then let the caller
            // decide what the failure means.
            try
            {
                objFunc.line_search_end();
            }
            catch (...)
            {
            }
            throw;
        }
    }

    double Wolfe::search(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const TVector &old_grad,
        const double starting_step_size)
    {
        f0 = old_energy;
        dphi0 = delta_x.dot(old_grad);
        generation = objFunc.objective_generation();
        swept = starting_step_size;
        boundary_reached = false;
        last_evaluated = NaN;
        evaluations = 0;
        gradient_evaluations = 0;
        growth_sweeps = 0;
        objective_restarts = 0;
        accepted_slope_ratio = NaN;

        if (!growth_permitted)
            no_growth = "direction_admits_no_growth";
        else if (starting_step_size < initial_step_size())
            no_growth = "start_capped";
        else if (!(growth_limit > starting_step_size))
            no_growth = "growth_limit";
        else
            no_growth.clear();

        Result result;
        TVector anchor_grad;
        if (!(std::isfinite(dphi0) && dphi0 < 0))
        {
            // No Wolfe point exists along a direction that is not one of
            // descent; leave the decision to the search this one extends.
            result.outcome = "fallback_not_descent";
            result.fallback = true;
        }
        else
        {
            init_compute_descent_step_size(delta_x, old_grad);
            while (true)
            {
                result = bracket_and_zoom(x, delta_x, objFunc, use_grad_norm, starting_step_size);
                if (!result.objective_changed)
                    break;

                // Everything measured so far belongs to a superseded function.
                if (objective_restarts >= max_objective_restarts)
                {
                    result.outcome = "objective_changed";
                    break;
                }
                ++objective_restarts;

                objFunc.solution_changed(x);
                last_evaluated = 0;
                generation = objFunc.objective_generation();
                f0 = objFunc(x);
                objFunc.gradient(x, anchor_grad);
                ++gradient_evaluations;
                dphi0 = delta_x.dot(anchor_grad);
                m_logger.debug("[Wolfe] the objective changed during the search (generation {}); restarting from x", generation);
                if (!std::isfinite(f0) || !(std::isfinite(dphi0) && dphi0 < 0))
                {
                    result.outcome = "objective_changed_not_descent";
                    result.objective_changed = true;
                    break;
                }
                init_compute_descent_step_size(delta_x, anchor_grad);
            }
        }

        double step_size = result.alpha;
        int reported_iterations = std::max(evaluations - 1, 0);
        if (result.fallback)
        {
            // RobustArmijo from the same start, on its own budget, under the
            // objective the search is now anchored to.
            const TVector &grad = objective_restarts > 0 ? anchor_grad : old_grad;
            cur_iter = 0;
            step_size = Backtracking::compute_descent_step_size(
                x, delta_x, objFunc, use_grad_norm, f0, grad, starting_step_size);
            reported_iterations = cur_iter;
        }
        else if (result.objective_changed)
        {
            step_size = NaN;
        }

        cur_iter = reported_iterations;

        ++outcome_counts[result.outcome];
        total_evaluations += evaluations;
        total_growth_sweeps += growth_sweeps;

        mutable_diagnostics()["wolfe"] = {
            {"outcome", result.outcome},
            {"evaluations", evaluations},
            {"gradient_evaluations", gradient_evaluations},
            {"growth_sweeps", growth_sweeps},
            {"swept_step_size", finite_or_null(swept)},
            {"growth_unavailable", no_growth.empty() ? json(nullptr) : json(no_growth)},
            {"objective_restarts", objective_restarts},
            {"accepted_slope_over_initial_slope", finite_or_null(accepted_slope_ratio)},
            {"initial_slope", finite_or_null(dphi0)}};

        m_logger.debug(
            "[Wolfe] {} alpha={:g} after {} evaluations ({} growth sweeps, swept to {:g}, {} objective restarts)",
            result.outcome, step_size, evaluations, growth_sweeps, swept, objective_restarts);

        return step_size;
    }

    Wolfe::Result Wolfe::bracket_and_zoom(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double starting_step_size)
    {
        const int budget = std::min(max_evaluations, current_max_step_size_iter());
        const double armijo_slope = armijo_criteria; // c phi'(0)

        const auto curvature = [&](const Trial &t) {
            return t.has_slope && std::abs(t.dphi) <= -c2 * dphi0;
        };

        // The best decrease point so far; alpha 0 until one is found.
        Trial lo;
        lo.alpha = 0;
        lo.phi = f0;
        lo.dphi = dphi0;
        lo.valid = true;
        lo.has_slope = true;
        lo.flat = true; // phi(0) against itself: only the slopes compare
        Trial hi;

        const auto out_of_budget = [&](const std::string &why) {
            Result r;
            // A decrease point that does not move the iterate in floating
            // point is not a step (measured: alpha = 9e-9 along a direction
            // of norm 2e-10 once energy noise had driven the zoom to zero).
            if (lo.alpha > 0 && (x + lo.alpha * delta_x).cwiseNotEqual(x).any())
                return accept(x, delta_x, objFunc, lo, (lo.flat ? "flat_" : "armijo_") + why);
            r.outcome = "fallback_" + why;
            r.fallback = true;
            return r;
        };

        // Whether t is a decrease point, i.e. may become lo. An energy within
        // eps_k = approximate_wolfe_epsilon |phi(0)| of phi(0), or within the
        // roundoff bound RobustArmijo's fallback uses (RB-19), cannot be
        // compared either way: the energy of a contact problem is a sum of
        // terms far larger than itself. Such a point is judged on its slope
        // alone, as in Hager and Zhang's approximate Wolfe conditions.
        // Evaluates the slope of every decrease point; a trial that is not
        // one is the new hi.
        const double energy_slack = approximate_epsilon * std::abs(f0);
        const auto decrease_point = [&](Trial &t, const bool compare_with_lo) {
            if (!t.valid)
                return false;
            t.flat = std::abs(t.phi - f0) <= energy_slack
                     || energy_at_roundoff(objFunc, use_grad_norm, f0, t.phi);
            if (!t.flat && (t.phi > f0 + t.alpha * armijo_criteria || (compare_with_lo && t.phi >= lo.phi)))
                return false;
            evaluate_slope(x, delta_x, objFunc, t);
            if (!t.has_slope)
            {
                t.valid = false;
                return false;
            }
            return true;
        };
        const auto accepted_outcome = [](const Trial &t) { return t.flat ? "approximate_wolfe" : "wolfe"; };

        // --- Bracketing ------------------------------------------------------
        double alpha = std::min(starting_step_size, swept);
        bool first = true;
        while (true)
        {
            if (evaluations >= budget)
                return out_of_budget("budget");

            Trial t = evaluate(x, delta_x, objFunc, alpha);
            if (objFunc.objective_generation() != generation)
            {
                Result r;
                r.objective_changed = true;
                return r;
            }

            if (!decrease_point(t, !first))
            {
                hi = t;
                break;
            }
            first = false;

            if (curvature(t))
                return accept(x, delta_x, objFunc, t, accepted_outcome(t));

            if (t.dphi >= 0)
            {
                // Past the minimizer along the line: zoom back towards lo.
                hi = lo;
                lo = t;
                break;
            }

            // A decrease point whose slope is still steeper than c2 allows:
            // a longer step is wanted.
            lo = t;

            if (alpha < swept)
            {
                alpha = std::min(alpha * growth_factor, swept);
                continue;
            }
            if (boundary_reached)
                return accept(x, delta_x, objFunc, t, "capped_feasibility");
            if (!no_growth.empty())
                return accept(x, delta_x, objFunc, t, "capped_" + no_growth);
            if (alpha >= growth_limit)
                return accept(x, delta_x, objFunc, t, "capped_growth_limit");

            // Price the longer interval before anything is evaluated in it.
            const double target = std::min(alpha * growth_factor, growth_limit);
            const TVector target_x = x + target * delta_x;
            double fraction;
            try
            {
                POLYSOLVE_SCOPED_STOPWATCH("Wolfe growth sweep", broad_phase_ccd_time, m_logger);
                objFunc.line_search_extend(x, target_x);
                ++growth_sweeps;
                fraction = objFunc.max_step_size(x, target_x);
            }
            catch (const std::exception &e)
            {
                // Growth is optional. A problem that cannot price the longer
                // sweep (a resource limit on the broad phase, say) refuses it;
                // the step already taken was priced by the shorter one, and
                // the problem's state is still that of the last trial.
                m_logger.debug("[Wolfe] growth sweep to alpha={:g} refused: {}", target, e.what());
                no_growth = "sweep_refused";
                return accept(x, delta_x, objFunc, t, "capped_sweep_refused");
            }

            double feasible = NaN;
            if (std::isfinite(fraction) && fraction > 0)
            {
                fraction = std::min(fraction, 1.);
                const int current_round = std::fegetround();
                std::fesetround(FE_DOWNWARD);
                feasible = target * fraction;
                std::fesetround(current_round);
            }
            if (!(feasible > alpha))
            {
                boundary_reached = true;
                swept = alpha;
                // The sweep rebuilt the problem's interval; its state is
                // still that of the last trial, which lies inside it.
                return accept(x, delta_x, objFunc, t, "capped_feasibility");
            }
            if (fraction < 1)
                boundary_reached = true;
            swept = feasible;
            alpha = feasible;
        }

        // --- Zoom ------------------------------------------------------------
        // lo is the best decrease point, and the slope at lo points into
        // [lo, hi]: dphi(lo) (hi - lo) < 0.
        while (true)
        {
            if (evaluations >= budget)
                return out_of_budget("budget");

            const double width = std::abs(hi.alpha - lo.alpha);
            if (!(width > std::max(current_min_step_size(), 1e-12 * std::max(lo.alpha, hi.alpha))))
                return out_of_budget("interval_collapsed");

            Trial t = evaluate(x, delta_x, objFunc, interpolate(lo, hi));
            if (objFunc.objective_generation() != generation)
            {
                Result r;
                r.objective_changed = true;
                return r;
            }

            if (!decrease_point(t, true))
            {
                hi = t;
                continue;
            }

            if (curvature(t))
                return accept(x, delta_x, objFunc, t, accepted_outcome(t));

            if (t.dphi * (hi.alpha - lo.alpha) >= 0)
                hi = lo;
            lo = t;
        }
    }

    Wolfe::Trial Wolfe::evaluate(const TVector &x, const TVector &delta_x, Problem &objFunc, const double alpha)
    {
        ++evaluations;
        Trial t;
        t.alpha = alpha;

        const TVector new_x = x + alpha * delta_x;
        try
        {
            POLYSOLVE_SCOPED_STOPWATCH("solution changed - constraint set update in LS", constraint_set_update_time, m_logger);
            objFunc.solution_changed(new_x);
        }
        catch (const std::runtime_error &e)
        {
            m_logger.warn("Failed to take step due to \"{}\", reduce step size...", e.what());
            last_evaluated = NaN;
            return t;
        }
        last_evaluated = alpha;

        if (objFunc.objective_generation() != generation)
            return t;

        if (!objFunc.is_step_valid(x, new_x))
            return t;

        t.phi = objFunc(new_x);
        t.valid = std::isfinite(t.phi);

        m_logger.trace("[Wolfe] trial {}: alpha={:g} {}={:g}", evaluations, alpha, log::delta("E"), t.phi - f0);
        return t;
    }

    void Wolfe::evaluate_slope(const TVector &x, const TVector &delta_x, Problem &objFunc, Trial &t)
    {
        assert(last_evaluated == t.alpha);
        TVector grad;
        objFunc.gradient(x + t.alpha * delta_x, grad);
        ++gradient_evaluations;
        t.dphi = delta_x.dot(grad);
        t.has_slope = std::isfinite(t.dphi);
    }

    double Wolfe::interpolate(const Trial &lo, const Trial &hi)
    {
        const double left = std::min(lo.alpha, hi.alpha);
        const double right = std::max(lo.alpha, hi.alpha);
        const double width = right - left;

        double a = NaN;
        if (lo.flat && hi.flat && hi.has_slope && hi.dphi != lo.dphi)
        {
            // Energies the evaluation cannot resolve carry no curvature
            // information; the slopes do (a secant step on phi').
            a = lo.alpha - lo.dphi * (hi.alpha - lo.alpha) / (hi.dphi - lo.dphi);
        }
        else if (hi.valid && hi.has_slope)
        {
            // Cubic through both ends' values and slopes (N&W eq. 3.59).
            const double d1 = lo.dphi + hi.dphi - 3 * (lo.phi - hi.phi) / (lo.alpha - hi.alpha);
            const double disc = d1 * d1 - lo.dphi * hi.dphi;
            if (disc >= 0)
            {
                const double d2 = (hi.alpha > lo.alpha ? 1. : -1.) * std::sqrt(disc);
                a = hi.alpha - (hi.alpha - lo.alpha) * (hi.dphi + d2 - d1) / (hi.dphi - lo.dphi + 2 * d2);
            }
        }
        if (!std::isfinite(a) && hi.valid)
        {
            // Quadratic through lo's value and slope and hi's value.
            const double h = hi.alpha - lo.alpha;
            const double curvature = hi.phi - lo.phi - lo.dphi * h;
            if (curvature > 0)
                a = lo.alpha - lo.dphi * h * h / (2 * curvature);
        }

        // Keep every trial well inside the interval, so it shrinks by a fixed
        // fraction whatever the model says; bisect where there is none.
        if (!std::isfinite(a))
            return left + 0.5 * width;
        return std::clamp(a, left + 0.1 * width, right - 0.1 * width);
    }

    Wolfe::Result Wolfe::accept(const TVector &x, const TVector &delta_x, Problem &objFunc,
                                const Trial &trial, const std::string &outcome)
    {
        if (!(last_evaluated == trial.alpha))
        {
            objFunc.solution_changed(x + trial.alpha * delta_x);
            last_evaluated = trial.alpha;
        }

        Result r;
        r.outcome = outcome;
        r.alpha = trial.alpha;
        accepted_slope_ratio = trial.has_slope && dphi0 != 0 ? trial.dphi / dphi0 : NaN;
        return r;
    }

    void Wolfe::update_solver_info(json &solver_info, const double per_iteration)
    {
        Superclass::update_solver_info(solver_info, per_iteration);
        solver_info["wolfe_outcomes"] = outcome_counts;
        solver_info["wolfe_evaluations"] = total_evaluations;
        solver_info["wolfe_growth_sweeps"] = total_growth_sweeps;
    }

    void Wolfe::reset_times()
    {
        Superclass::reset_times();
        outcome_counts.clear();
        total_evaluations = 0;
        total_growth_sweeps = 0;
    }
} // namespace polysolve::nonlinear::line_search
