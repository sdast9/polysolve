#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/BoxConstraintSolver.hpp>
#include <polysolve/nonlinear/line_search/LineSearch.hpp>

#include <catch2/catch.hpp>
#include <spdlog/sinks/null_sink.h>

#include <cmath>
#include <functional>
#include <limits>

using namespace polysolve;
using namespace polysolve::nonlinear;

// ===========================================================================
// Strong Wolfe line search (BFGS audit stage 3)
//
// The search may grow the step beyond the direction, which none of the other
// searches do, so every growth must first have the problem rebuild and price
// the longer interval. These regressions state what it does at each boundary
// the audit named: a direction that needs alpha > 1, a feasible bound below
// the curvature threshold, non-finite and invalid trials, a zero CCD step,
// contacts that appear along the line, energy roundoff, an objective change
// during the search, and an exception in the middle of it.
// ===========================================================================

namespace
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    /// f(x) = k x^2 / 2 on one variable, with every line-search callback the
    /// PolyFEM contact problems implement, each recorded:
    /// - a contact at `contact_at`: a penalty ck (contact_at - x)^2 / 2 that,
    ///   like an IPC collision set, exists only once solution_changed has seen
    ///   a position past it, and is evaluated from that stored set;
    /// - a wall at `wall` that max_step_size (the CCD bound) will not cross;
    /// - a region of infinite energy and a region of invalid steps.
    class LineProblem : public Problem
    {
    public:
        double k = 1;
        double contact_at = -inf, contact_k = 0;
        double wall = -inf;
        double infinite_below = -inf;
        double invalid_below = -inf;
        double throw_gradient_below = -inf;
        bool zero_bound_beyond_direction = false;
        uint64_t generation = 0;
        std::function<void(LineProblem &, double)> on_solution_changed;

        // --- what the line search did ---
        double origin = NaN, direction_length = NaN; ///< set by the test
        double state = NaN;                          ///< last solution_changed position
        bool active = false;                         ///< the stored contact set
        int begins = 0, extends = 0, ends = 0, bound_calls = 0;
        double priced = 0;             ///< longest distance from origin a sweep has priced
        int unpriced_evaluations = 0;  ///< trials beyond the direction before pricing
        int bound_outside_sweep = 0;   ///< max_step_size called beyond the current sweep
        int stale_gradients = 0;       ///< gradients at a position solution_changed has not seen
        double farthest_evaluated = 0; ///< farthest distance from origin evaluated
        double sweep_end = NaN;

        double value(const TVector &x) override
        {
            note_evaluation(x[0]);
            if (x[0] < infinite_below)
                return inf;
            double e = 0.5 * k * x[0] * x[0];
            if (active && x[0] < contact_at)
                e += 0.5 * contact_k * (contact_at - x[0]) * (contact_at - x[0]);
            return e;
        }

        void gradient(const TVector &x, TVector &grad) override
        {
            if (x[0] != state)
                ++stale_gradients;
            if (x[0] < throw_gradient_below)
                throw std::logic_error("gradient refused");
            grad.resize(1);
            grad[0] = k * x[0];
            if (active && x[0] < contact_at)
                grad[0] -= contact_k * (contact_at - x[0]);
        }

        void hessian(const TVector &, THessian &) override { throw std::runtime_error("not used"); }

        void solution_changed(const TVector &x) override
        {
            state = x[0];
            active = x[0] < contact_at;
            if (on_solution_changed)
                on_solution_changed(*this, x[0]);
        }

        bool is_step_valid(const TVector &, const TVector &x1) override { return x1[0] >= invalid_below; }

        void line_search_begin(const TVector &x0, const TVector &x1) override
        {
            ++begins;
            sweep_end = x1[0];
        }

        void line_search_extend(const TVector &x0, const TVector &x1) override
        {
            ++extends;
            line_search_begin(x0, x1);
        }

        void line_search_end() override { ++ends; }

        double max_step_size(const TVector &x0, const TVector &x1) override
        {
            ++bound_calls;
            // CCD prices only the interval the broad phase was built for.
            if (!(std::abs(x1[0] - x0[0]) <= std::abs(sweep_end - x0[0]) * (1 + 1e-15)))
                ++bound_outside_sweep;
            double fraction = 1;
            if (zero_bound_beyond_direction && std::abs(x1[0] - x0[0]) > direction_length * (1 + 1e-12))
                fraction = 0;
            else if (x1[0] < wall)
                fraction = (x0[0] - wall) / (x0[0] - x1[0]);
            priced = std::max(priced, fraction * std::abs(x1[0] - x0[0]));
            return fraction;
        }

        uint64_t objective_generation() const override { return generation; }

        void start(const double x0, const double d)
        {
            origin = x0;
            direction_length = std::abs(d);
            solution_changed(TVector::Constant(1, x0));
        }

    private:
        void note_evaluation(const double x)
        {
            if (!std::isfinite(origin))
                return;
            const double distance = std::abs(x - origin);
            farthest_evaluated = std::max(farthest_evaluated, distance);
            // The base search evaluates the unit step before any CCD, as it
            // always has; only growth beyond it needs a priced interval.
            if (distance > direction_length * (1 + 1e-12) && distance > priced * (1 + 1e-12))
                ++unpriced_evaluations;
        }
    };

    json line_search_params(const std::string &method, const json &wolfe = json::object())
    {
        json p = {
            {"line_search",
             {{"method", method},
              {"use_grad_norm_tol", 0},
              {"min_step_size", 1e-10},
              {"max_step_size_iter", 30},
              {"min_step_size_final", 1e-20},
              {"max_step_size_iter_final", 100},
              {"default_init_step_size", 1.},
              {"step_ratio", .5},
              {"Armijo", {{"c", 1e-4}, {"roundoff_tolerance", std::numeric_limits<double>::epsilon()}}},
              {"RobustArmijo", {{"delta_relative_tolerance", .1}}},
              {"Wolfe",
               {{"c2", .9},
                {"growth_factor", 2.},
                {"growth_limit", 16.},
                {"max_evaluations", 20},
                {"max_objective_restarts", 2},
                {"approximate_wolfe_epsilon", 1e-6}}}}}};
        for (const auto &[key, value] : wolfe.items())
            p["line_search"]["Wolfe"][key] = value;
        return p;
    }

    struct Search
    {
        std::shared_ptr<line_search::LineSearch> ls;
        double alpha;
        json wolfe;
    };

    Search run(LineProblem &f, const double x0, const double d,
               const std::string &method = "Wolfe", const json &wolfe = json::object(),
               const bool growth_permitted = true)
    {
        static spdlog::logger logger("wolfe-tests", std::make_shared<spdlog::sinks::null_sink_mt>());
        auto ls = line_search::LineSearch::create(line_search_params(method, wolfe), logger);
        ls->reset_times();
        ls->set_growth_permitted(growth_permitted);
        f.start(x0, d);
        const double alpha = ls->line_search(Eigen::VectorXd::Constant(1, x0), Eigen::VectorXd::Constant(1, d), f);
        const json &diag = ls->diagnostics();
        return {ls, alpha, diag.contains("wolfe") ? diag["wolfe"] : json(nullptr)};
    }

    /// phi'(alpha) / phi'(0) of the exact objective (contact included when
    /// the accepted point is past it).
    double slope_ratio(LineProblem &f, const double x0, const double d, const double alpha)
    {
        LineProblem exact = f;
        exact.on_solution_changed = nullptr;
        Eigen::VectorXd g0, g1;
        exact.solution_changed(Eigen::VectorXd::Constant(1, x0));
        exact.gradient(Eigen::VectorXd::Constant(1, x0), g0);
        exact.solution_changed(Eigen::VectorXd::Constant(1, x0 + alpha * d));
        exact.gradient(Eigen::VectorXd::Constant(1, x0 + alpha * d), g1);
        return (g1[0] * d) / (g0[0] * d);
    }
} // namespace

TEST_CASE("wolfe-grows-a-direction-that-is-too-short", "[solver][line_search][wolfe]")
{
    // f = x^2/2 from x = 1 along d = -0.04: phi'(a)/phi'(0) = 1 - 0.04 a,
    // so the strong Wolfe condition at c2 = 0.9 needs a >= 2.5.
    LineProblem f;
    const Search s = run(f, 1, -.04);
    CHECK(s.alpha == 4);
    CHECK(s.wolfe["outcome"] == "wolfe");
    CHECK(s.wolfe["growth_sweeps"] == 2);
    CHECK(std::abs(slope_ratio(f, 1, -.04, s.alpha)) <= .9);
    // Each growth was priced before anything was evaluated in it, and the
    // bound was always asked about the interval the sweep had built.
    CHECK(f.begins == 3);
    CHECK(f.extends == 2);
    CHECK(f.unpriced_evaluations == 0);
    CHECK(f.bound_outside_sweep == 0);
    CHECK(f.state == 1 - .04 * s.alpha);
    CHECK(f.ends == 1);

    // RobustArmijo accepts the unit step, which leaves the slope at 96 % of
    // the initial one.
    LineProblem g;
    CHECK(run(g, 1, -.04, "RobustArmijo").alpha == 1);
}

TEST_CASE("wolfe-stops-at-the-feasible-bound", "[solver][line_search][wolfe]")
{
    // A wall at x = 0.912 is alpha = 2.2, below the curvature threshold 2.5:
    // no Wolfe point is feasible, and the search stops at the bound.
    LineProblem f;
    f.wall = 1 - .04 * 2.2;
    const Search s = run(f, 1, -.04);
    CHECK(s.wolfe["outcome"] == "capped_feasibility");
    CHECK(s.alpha <= 2.2);
    CHECK(s.alpha == Approx(2.2).epsilon(1e-12));
    // (to the last ulp of the position arithmetic, as CCD's own bound is)
    CHECK(1 - .04 * s.alpha >= f.wall - 1e-15);
    CHECK(f.farthest_evaluated <= .04 * 2.2 * (1 + 1e-12));
    CHECK(f.unpriced_evaluations == 0);
    CHECK(f.bound_outside_sweep == 0);
    // Armijo holds at the accepted step: it is a decrease, not a Wolfe point.
    CHECK(0.5 * std::pow(1 - .04 * s.alpha, 2) < .5);
    CHECK(slope_ratio(f, 1, -.04, s.alpha) > .9);
}

TEST_CASE("wolfe-does-not-grow-past-a-zero-ccd-step", "[solver][line_search][wolfe]")
{
    LineProblem f;
    f.zero_bound_beyond_direction = true;
    const Search s = run(f, 1, -.04);
    CHECK(s.alpha == 1);
    CHECK(s.wolfe["outcome"] == "capped_feasibility");
    CHECK(f.farthest_evaluated <= .04 * (1 + 1e-12));
    CHECK(f.state == 1 - .04);

    // A zero bound on the direction itself fails the search, as before.
    LineProblem g;
    g.wall = 1;
    const Search t = run(g, 1, -.04);
    CHECK(std::isnan(t.alpha));
    CHECK(t.ls->diagnostics()["failure_stage"] == "feasibility_bound");
}

TEST_CASE("wolfe-starts-no-growth-from-a-capped-step", "[solver][line_search][wolfe]")
{
    // The direction crosses a wall at alpha = 0.5: the start is capped, so
    // the search must not look beyond it even though the slope is steep.
    LineProblem f;
    f.wall = 1 - .04 * .5;
    const Search s = run(f, 1, -.04);
    CHECK(s.alpha <= .5);
    CHECK(s.wolfe["outcome"] == "capped_start_capped");
    CHECK(f.begins == 1);
    CHECK(1 - .04 * s.alpha >= f.wall - 1e-15);
}

TEST_CASE("wolfe-respects-a-direction-that-admits-no-growth", "[solver][line_search][wolfe]")
{
    LineProblem f;
    const Search s = run(f, 1, -.04, "Wolfe", json::object(), /*growth_permitted=*/false);
    CHECK(s.alpha == 1);
    CHECK(s.wolfe["outcome"] == "capped_direction_admits_no_growth");
    CHECK(f.begins == 1);
}

TEST_CASE("wolfe-stops-at-the-growth-limit", "[solver][line_search][wolfe]")
{
    LineProblem f;
    const Search s = run(f, 1, -.001, "Wolfe", {{"growth_limit", 8.}});
    CHECK(s.alpha == 8);
    CHECK(s.wolfe["outcome"] == "capped_growth_limit");
}

TEST_CASE("wolfe-never-accepts-a-nonfinite-or-invalid-trial", "[solver][line_search][wolfe]")
{
    const std::string region = GENERATE("infinite", "invalid");
    CAPTURE(region);
    // Growth reaches alpha = 4 (x = 0.84), past the region's edge at
    // x = 0.85; the search must come back inside it.
    LineProblem f;
    (region == "infinite" ? f.infinite_below : f.invalid_below) = .85;
    const Search s = run(f, 1, -.04);
    REQUIRE(std::isfinite(s.alpha));
    CHECK(1 - .04 * s.alpha >= .85);
    CHECK(s.wolfe["outcome"] == "wolfe");
    CHECK(std::abs(slope_ratio(f, 1, -.04, s.alpha)) <= .9);
    CHECK(f.state == 1 - .04 * s.alpha);
}

TEST_CASE("wolfe-zooms-back-from-an-overshoot", "[solver][line_search][wolfe]")
{
    SECTION("the unit step fails the Armijo decrease")
    {
        // d = -3 overshoots to x = -2; the minimizer along the line is 1/3.
        LineProblem f;
        const Search s = run(f, 1, -3);
        CHECK(s.wolfe["outcome"] == "wolfe");
        CHECK(std::abs(slope_ratio(f, 1, -3, s.alpha)) <= .9);
        CHECK(s.alpha < 1);
        LineProblem g;
        CHECK(run(g, 1, -3, "RobustArmijo").alpha == .5);
    }
    SECTION("the unit step decreases but has passed the minimizer")
    {
        // d = -1.95: alpha = 1 satisfies Armijo, but the slope has turned
        // positive at 95 % of the initial one; the curvature half rejects it.
        LineProblem f;
        const Search s = run(f, 1, -1.95);
        CHECK(s.wolfe["outcome"] == "wolfe");
        CHECK(s.alpha < 1);
        CHECK(std::abs(slope_ratio(f, 1, -1.95, s.alpha)) <= .9);
        LineProblem g;
        CHECK(run(g, 1, -1.95, "RobustArmijo").alpha == 1);
    }
}

TEST_CASE("wolfe-evaluates-contacts-that-appear-along-the-line", "[solver][line_search][wolfe]")
{
    // A stiff contact at x = 0.9 (alpha = 2.5) that only exists once the
    // problem has seen a position past it, like an IPC collision set.
    LineProblem f;
    f.contact_at = .9;
    f.contact_k = 100;
    const Search s = run(f, 1, -.04);
    REQUIRE(std::isfinite(s.alpha));
    CHECK(s.wolfe["outcome"] == "wolfe");
    // The slope test was always made with the contact set of the trial.
    CHECK(f.stale_gradients == 0);
    CHECK(std::abs(slope_ratio(f, 1, -.04, s.alpha)) <= .9);
    // The problem is left at the accepted step, with its contact set.
    CHECK(f.state == 1 - .04 * s.alpha);
    CHECK(f.active == (f.state < f.contact_at));
}

TEST_CASE("wolfe-judges-unresolved-energies-on-their-slope", "[solver][line_search][wolfe]")
{
    // f = C + x^2/2 with C = 1e8: from x = 1e-6 the whole decrease (5e-13) is
    // far below the energy's resolution (one ulp of 1e8 is 1.5e-8), and the
    // measured energy reads one ulp *higher* away from the start, as
    // summation roundoff can. Comparing energies cannot place any trial.
    class Offset : public LineProblem
    {
    public:
        double value(const TVector &x) override
        {
            const double ulp = 1.4901161193847656e-08;
            return 1e8 + LineProblem::value(x) + (x[0] == origin ? 0. : ulp);
        }
    };

    SECTION("the exact step: the slope vanishes there")
    {
        Offset f, g;
        const Search s = run(f, 1e-6, -1e-6);
        CHECK(s.wolfe["outcome"] == "approximate_wolfe");
        CHECK(s.alpha == 1);
        CHECK(s.wolfe["evaluations"] == 1);
        CHECK(run(g, 1e-6, -1e-6, "RobustArmijo").alpha == 1);
    }
    SECTION("noise above the roundoff bound: only the energy slack sees it")
    {
        // Ten ulps is above the roundoff bound (1e8 eps = 2.2e-8), as the
        // summation noise of a contact energy is; eps_k = 1e-6 * 1e8 covers it.
        class Noisy : public LineProblem
        {
        public:
            double value(const TVector &x) override
            {
                return 1e8 + LineProblem::value(x) + (x[0] == origin ? 0. : 1.4901161193847656e-07);
            }
        };
        Noisy f, g;
        const Search s = run(f, 1e-6, -2e-6);
        CHECK(s.wolfe["outcome"] == "approximate_wolfe");
        CHECK(s.alpha == Approx(.5));
        // Without the slack every trial reads as an increase, and the search
        // must not accept a step the energy did not certify.
        const Search t = run(g, 1e-6, -2e-6, "Wolfe", {{"approximate_wolfe_epsilon", 0.}});
        CHECK(t.wolfe["outcome"] != "approximate_wolfe");
        CHECK(t.wolfe["outcome"].get<std::string>().rfind("fallback_", 0) == 0);
    }
    SECTION("twice the exact step: the slope has turned")
    {
        // x = 1e-6 along d = -2e-6: at alpha = 1 the slope is +|phi'(0)|;
        // the minimizer at alpha = 1/2 is found on slopes alone.
        Offset f, g;
        const Search s = run(f, 1e-6, -2e-6);
        CHECK(s.wolfe["outcome"] == "approximate_wolfe");
        CHECK(s.alpha == Approx(.5));
        CHECK(std::abs(slope_ratio(f, 1e-6, -2e-6, s.alpha)) <= .9);
        CHECK(run(g, 1e-6, -2e-6, "RobustArmijo").alpha == .5);
    }
}

TEST_CASE("wolfe-restarts-when-the-objective-changes", "[solver][line_search][wolfe]")
{
    SECTION("once: the search finishes under the new objective")
    {
        // The first trial past alpha = 1.5 retunes the objective: k 1 -> 4.
        LineProblem f;
        bool retuned = false;
        f.on_solution_changed = [&](LineProblem &p, const double x) {
            if (!retuned && x < 1 - .04 * 1.5)
            {
                retuned = true;
                p.k = 4;
                ++p.generation;
            }
        };
        const Search s = run(f, 1, -.04);
        REQUIRE(std::isfinite(s.alpha));
        CHECK(s.wolfe["objective_restarts"] == 1);
        // Measured against the new objective's own initial slope.
        CHECK(s.wolfe["initial_slope"] == Approx(-4 * .04));
        CHECK(std::abs(slope_ratio(f, 1, -.04, s.alpha)) <= .9);
        CHECK(f.state == 1 - .04 * s.alpha);
    }
    SECTION("every time: the restart budget ends it, and it cleans up")
    {
        LineProblem f;
        f.on_solution_changed = [](LineProblem &p, const double x) {
            if (x != p.origin)
                ++p.generation;
        };
        const Search s = run(f, 1, -.04, "Wolfe", {{"max_objective_restarts", 2}});
        CHECK(std::isnan(s.alpha));
        CHECK(s.wolfe["outcome"] == "objective_changed");
        CHECK(s.wolfe["objective_restarts"] == 2);
        CHECK(s.ls->diagnostics()["failure_stage"] == "descent_search");
        CHECK(f.state == 1);
        CHECK(f.ends == 1);
    }
}

TEST_CASE("wolfe-ends-the-interval-when-a-trial-throws", "[solver][line_search][wolfe]")
{
    // The gradient refuses past x = 0.94 (alpha = 1.5), in the first growth.
    LineProblem f;
    f.throw_gradient_below = .94;
    static spdlog::logger logger("wolfe-throw", std::make_shared<spdlog::sinks::null_sink_mt>());
    auto ls = line_search::LineSearch::create(line_search_params("Wolfe"), logger);
    f.start(1, -.04);
    CHECK_THROWS_AS(ls->line_search(Eigen::VectorXd::Ones(1), Eigen::VectorXd::Constant(1, -.04), f), std::logic_error);
    CHECK(f.begins == 2);
    CHECK(f.ends == 1);
}

TEST_CASE("wolfe-refuses-inconsistent-parameters", "[solver][line_search][wolfe]")
{
    spdlog::logger logger("wolfe-params", std::make_shared<spdlog::sinks::null_sink_mt>());
    CHECK_THROWS(line_search::LineSearch::create(line_search_params("Wolfe", {{"c2", 1e-5}}), logger));
    CHECK_THROWS(line_search::LineSearch::create(line_search_params("Wolfe", {{"c2", 1.}}), logger));
    CHECK_THROWS(line_search::LineSearch::create(line_search_params("Wolfe", {{"growth_factor", 1.}}), logger));
    CHECK_NOTHROW(line_search::LineSearch::create(line_search_params("Wolfe"), logger));
}

// ===========================================================================
// Inside the solver
// ===========================================================================

namespace
{
    /// The chained Rosenbrock function with its exact gradient.
    class Rosenbrock : public Problem
    {
    public:
        double value(const TVector &x) override
        {
            double v = 0;
            for (int i = 0; i + 1 < x.size(); ++i)
                v += 100 * std::pow(x[i + 1] - x[i] * x[i], 2) + std::pow(1 - x[i], 2);
            return v;
        }
        void gradient(const TVector &x, TVector &g) override
        {
            g = TVector::Zero(x.size());
            for (int i = 0; i + 1 < x.size(); ++i)
            {
                const double r = x[i + 1] - x[i] * x[i];
                g[i] += -400 * x[i] * r - 2 * (1 - x[i]);
                g[i + 1] += 200 * r;
            }
        }
        void hessian(const TVector &, THessian &) override { throw std::runtime_error("not used"); }
    };
} // namespace

TEST_CASE("wolfe-lbfgs-converges-with-positive-curvature-pairs", "[solver][line_search][wolfe]")
{
    spdlog::logger logger("wolfe-lbfgs", std::make_shared<spdlog::sinks::null_sink_mt>());
    const std::string method = GENERATE("Wolfe", "RobustArmijo");
    CAPTURE(method);
    json params = line_search_params(method);
    params["solver"] = json::array({json{{"type", "L-BFGS"}, {"history_size", 6}}});
    params["grad_norm_tol"] = 1e-8;
    params["rel_grad_norm_tol"] = 0;
    params["max_iterations"] = 2000;
    params["advanced"] = {{"derivative_along_delta_x_tol", 0}};
    auto solver = Solver::create(params, json::object(), 1, logger);

    Rosenbrock f;
    Eigen::VectorXd x(10);
    x << -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1;
    REQUIRE_NOTHROW(solver->minimize(f, x));
    CHECK(solver->status() == Status::GradNormTolerance);
    CHECK((x - Eigen::VectorXd::Ones(10)).norm() < 1e-6);

    const json &guard = solver->info()["curvature_guard"]["L-BFGS"];
    if (method == "Wolfe")
    {
        // Every step met the curvature condition or was an explicit
        // outcome; Wolfe steps give positive curvature, so the safeguard
        // never had anything to refuse.
        const json &outcomes = solver->info()["wolfe_outcomes"];
        CAPTURE(outcomes.dump());
        CHECK(outcomes.value("wolfe", 0) > 0);
        CHECK(guard["skipped_pairs"] == 0);
        CHECK(guard["history_resets"] == 0);
    }
}

TEST_CASE("wolfe-keeps-box-constrained-steps-inside-the-box", "[solver][line_search][wolfe]")
{
    // L-BFGS-B points at the bound x = 4 of f = |x - 10|^2 / 2; with c2 = 0.5
    // the slope there is still too steep, and growth would leave the box.
    class Far : public Problem
    {
    public:
        double value(const TVector &x) override { return .5 * (x.array() - 10).square().sum(); }
        void gradient(const TVector &x, TVector &g) override { g = x.array() - 10; }
        void hessian(const TVector &, THessian &) override { throw std::runtime_error("not used"); }
    } f;

    spdlog::logger logger("wolfe-box", std::make_shared<spdlog::sinks::null_sink_mt>());
    json params = line_search_params("Wolfe", {{"c2", .5}});
    params["solver"] = "L-BFGS-B";
    params["box_constraints"] = {{"bounds", std::vector<double>({0, 4})}, {"max_change", 4}};
    params["max_iterations"] = 100;
    auto solver = BoxConstraintSolver::create(params, json::object(), 1, logger);

    Eigen::VectorXd x = Eigen::VectorXd::Constant(3, 3.);
    REQUIRE_NOTHROW(solver->minimize(f, x));
    CHECK((x.array() <= 4).all());
    CHECK((x.array() - 4).abs().maxCoeff() < 1e-12);
    CHECK(solver->info()["wolfe_outcomes"].value("capped_direction_admits_no_growth", 0) > 0);
    CHECK(solver->info()["wolfe_growth_sweeps"] == 0);
}
