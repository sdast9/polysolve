#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/descent_strategies/BFGS.hpp>
#include <polysolve/nonlinear/descent_strategies/LBFGS.hpp>

#include <catch2/catch.hpp>
#include <spdlog/sinks/null_sink.h>

#include <algorithm>

using namespace polysolve;
using namespace polysolve::nonlinear;

namespace
{
    class ScalarQuadratic : public Problem
    {
    public:
        explicit ScalarQuadratic(const double curvature) : curvature(curvature) {}

        double value(const TVector &x) override { return 0.5 * curvature * x.squaredNorm(); }
        void gradient(const TVector &x, TVector &grad) override { grad = curvature * x; }
        void hessian(const TVector &x, THessian &hess) override
        {
            hess.resize(x.size(), x.size());
            hess.setIdentity();
            hess *= curvature;
        }

    private:
        double curvature;
    };
} // namespace

TEST_CASE("bfgs-uses-current-secant", "[solver][bfgs]")
{
    spdlog::logger logger("bfgs-current-secant", std::make_shared<spdlog::sinks::null_sink_mt>());
    BFGS strategy(json::object(), {{"solver", "Eigen::LDLT"}}, 1, logger);

    // In one dimension a secant recovers the exact curvature of a quadratic.
    // The very next direction must therefore lead directly to its minimizer.
    // Reuse the strategy to check that reset also discards the previous scale.
    for (const double curvature : {4., 9., 0.25})
    {
        CAPTURE(curvature);
        ScalarQuadratic problem(curvature);
        strategy.reset(1);
        Eigen::VectorXd x = Eigen::VectorXd::Ones(1), grad, direction;
        problem.gradient(x, grad);
        REQUIRE(strategy.compute_update_direction(problem, x, grad, direction));
        CHECK(direction[0] == Approx(-curvature));

        for (const double position : {0.5, 0.25, 0.125})
        {
            x[0] = position;
            problem.gradient(x, grad);
            REQUIRE(strategy.compute_update_direction(problem, x, grad, direction));
            CHECK(direction[0] == Approx(-position));
        }
    }
}

TEST_CASE("bfgs-quadratic-converges-after-first-secant", "[solver][bfgs]")
{
    spdlog::logger logger("bfgs-quadratic", std::make_shared<spdlog::sinks::null_sink_mt>());
    ScalarQuadratic problem(3.);
    json params = {
        {"solver", json::array({json{{"type", "BFGS"}}})},
        {"grad_norm_tol", 1e-12},
        {"rel_grad_norm_tol", 0},
        {"line_search", {{"method", "Armijo"}}},
        {"advanced", {{"derivative_along_delta_x_tol", 0}}}};
    auto solver = Solver::create(params, {{"solver", "Eigen::LDLT"}}, 1, logger);
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    REQUIRE_NOTHROW(solver->minimize(problem, x));
    CHECK(solver->status() == Status::GradNormTolerance);
    CHECK(x.norm() < 1e-12);
    CHECK(solver->current_criteria().iterations == 2);
}

// ===========================================================================
// Curvature safeguard
//
// A BFGS approximation stays positive definite only while every stored pair
// satisfies s.y > 0 with a representable scale. None of the available line
// searches tests the Wolfe curvature condition, and the audit of 2026-09-22
// reproduced ordinary Armijo steps whose pairs have zero, negative or
// unrepresentably scaled curvature. These regressions state what the
// strategies do with such pairs, and that valid ones are untouched.
// ===========================================================================

namespace
{
    /// One-variable polynomial given by its coefficients, lowest order first.
    class Polynomial : public Problem
    {
    public:
        explicit Polynomial(std::vector<double> coefficients) : c(std::move(coefficients)) {}

        double value(const TVector &x) override
        {
            double v = 0;
            for (int i = int(c.size()) - 1; i >= 0; --i)
                v = v * x[0] + c[i];
            return v;
        }

        void gradient(const TVector &x, TVector &grad) override
        {
            grad = TVector::Zero(1);
            for (int i = int(c.size()) - 1; i >= 1; --i)
                grad[0] = grad[0] * x[0] + i * c[i];
        }

        void hessian(const TVector &x, THessian &hess) override
        {
            double v = 0;
            for (int i = int(c.size()) - 1; i >= 2; --i)
                v = v * x[0] + i * (i - 1) * c[i];
            hess.resize(1, 1);
            hess.setZero();
            hess.coeffRef(0, 0) = v;
        }

    private:
        std::vector<double> c;
    };

    /// The BFGS strategies read only the (x, gradient) pairs they are handed,
    /// so a test can state a secant pair exactly without an objective that
    /// produces it. Nothing here is ever evaluated.
    class UnusedProblem : public Problem
    {
    public:
        double value(const TVector &) override { return 0; }
        void gradient(const TVector &x, TVector &grad) override { grad = TVector::Zero(x.size()); }
        void hessian(const TVector &, THessian &) override { throw std::runtime_error("not used"); }
    };

    Eigen::VectorXd scalar(const double v)
    {
        Eigen::VectorXd x(1);
        x << v;
        return x;
    }

    Eigen::VectorXd vector(const std::vector<double> &entries)
    {
        return Eigen::Map<const Eigen::VectorXd>(entries.data(), entries.size());
    }

    /// Exact equality: the safeguarded fallbacks are the steepest descent
    /// direction itself, not an approximation of it.
    bool identical(const Eigen::VectorXd &a, const Eigen::VectorXd &b)
    {
        return a.size() == b.size() && (a.array() == b.array()).all();
    }

    json guard_report(DescentStrategy &strategy, const std::string &name)
    {
        json info;
        strategy.update_solver_info(info, 1);
        return info["curvature_guard"][name];
    }

    /// Hand one secant pair to a freshly reset strategy. The first direction
    /// has no history and is -g0 whatever the pair is; the second is the one
    /// the pair produced, and is returned with the guard's own report.
    json feed_pair(DescentStrategy &strategy, const std::string &name,
                   const Eigen::VectorXd &x0, const Eigen::VectorXd &g0,
                   const Eigen::VectorXd &s, const Eigen::VectorXd &y,
                   Eigen::VectorXd &direction)
    {
        UnusedProblem problem;
        strategy.reset(int(x0.size()));
        strategy.reset_times();

        Eigen::VectorXd first;
        REQUIRE(strategy.compute_update_direction(problem, x0, g0, first));
        REQUIRE(identical(first, -g0));

        // The dense strategy solves into the direction through an Eigen::Ref,
        // which the caller sizes; the solver passes it a zeroed vector.
        direction = Eigen::VectorXd::Zero(x0.size());
        strategy.compute_update_direction(problem, x0 + s, g0 + y, direction);
        return guard_report(strategy, name);
    }

    /// Both strategies under one set of safeguard options. The dense form
    /// needs a dense linear solver; neither reads the other's parameters.
    std::vector<std::pair<std::string, std::shared_ptr<DescentStrategy>>>
    both_strategies(const json &options, spdlog::logger &logger)
    {
        json limited_memory = options;
        limited_memory["history_size"] = 6;
        return {{"L-BFGS", std::make_shared<LBFGS>(limited_memory, 1, logger)},
                {"BFGS", std::make_shared<BFGS>(options, json{{"solver", "Eigen::LDLT"}}, 1, logger)}};
    }

    /// A solver made of one strategy alone: a converged fallback chain does
    /// not establish that the strategy itself stayed valid.
    json pure_strategy(const std::string &type, const json &options)
    {
        json strategy = options;
        strategy["type"] = type;
        return json{
            {"solver", json::array({strategy})},
            {"grad_norm_tol", 1e-10},
            {"rel_grad_norm_tol", 0},
            {"max_iterations", 500},
            {"line_search", {{"method", "Armijo"}}},
            {"advanced", {{"derivative_along_delta_x_tol", 0}}}};
    }
} // namespace

TEST_CASE("bfgs-skips-invalid-curvature-pairs", "[solver][bfgs][curvature]")
{
    spdlog::logger logger("bfgs-invalid-pairs", std::make_shared<spdlog::sinks::null_sink_mt>());

    // The pair, the reason it cannot enter the approximation, and where it
    // comes from. Every one of these is an ordinary accepted step.
    struct Invalid
    {
        std::vector<double> s, y;
        std::string reason;
    };
    const std::vector<Invalid> invalid{
        // The accepted first Armijo step of f(x)=x+x^2(x+1)^2 from x=0: the
        // energy falls from 0 to -1 and the gradient does not move at all.
        {{-1.}, {0.}, "zero_gradient_change"},
        // The accepted first Armijo step of f(x)=x^4/4-x^2/2+0.1x from x=0,
        // whose curvature is negative (s*y = -0.0099).
        {{-0.1}, {0.099}, "insufficient_curvature"},
        // Gradients of two different objectives, the mechanism of a mid-solve
        // contact retune: f=x^2/8 at x=1, then f=2x^2 at x=0.75, s*y=-0.6875.
        {{-0.25}, {2.75}, "insufficient_curvature"},
        // A displacement the line search shrank to nothing.
        {{0.}, {1.}, "zero_displacement"},
        // Exactly zero curvature with both vectors nonzero.
        {{1., -1.}, {1., 1.}, "insufficient_curvature"},
        // Nearly orthogonal: positive, but far below the relative threshold.
        {{1., 0.}, {1e-12, 1.}, "insufficient_curvature"},
        // Positive curvature, but |y|^2 overflows before it is ever used.
        {{1e-8}, {1e200}, "non_finite_pair"},
        // Positive curvature and finite norms, but the initial scale
        // |y|^2 / s.y is not representable.
        {{1e-160}, {1e150}, "invalid_scale"},
    };

    // curvature_restart is off here so that the refusal is the only thing
    // under test; what the default does with the approximation afterwards is
    // in bfgs-restarts-when-the-approximation-goes-stale.
    for (const auto &[name, strategy] : both_strategies(json{{"curvature_restart", 0}}, logger))
    {
        for (const auto &pair : invalid)
        {
            CAPTURE(name, pair.reason);
            const Eigen::VectorXd s = vector(pair.s), y = vector(pair.y);
            const Eigen::VectorXd g0 = Eigen::VectorXd::Ones(s.size());
            Eigen::VectorXd direction;
            const json report = feed_pair(*strategy, name, Eigen::VectorXd::Zero(s.size()), g0, s, y, direction);

            // Nothing was stored, so the approximation is still the identity
            // and the direction is the steepest descent one at the new point.
            CHECK(identical(direction, -(g0 + y)));
            CHECK(report["accepted_pairs"] == 0);
            CHECK(report["damped_pairs"] == 0);
            CHECK(report["skipped_pairs"] == 1);
            CHECK(report["skipped"][pair.reason] == 1);
            CHECK(report["history_resets"] == 0);
        }

        // A gradient that is not finite at all leaves no usable direction:
        // the pair is still refused, and the strategy reports the failure
        // instead of handing back a poisoned step.
        UnusedProblem problem;
        strategy->reset(1);
        strategy->reset_times();
        Eigen::VectorXd direction;
        REQUIRE(strategy->compute_update_direction(problem, scalar(0.), scalar(1.), direction));
        const double nan = std::numeric_limits<double>::quiet_NaN();
        CHECK_FALSE(strategy->compute_update_direction(problem, scalar(1.), scalar(nan), direction));
        const json report = guard_report(*strategy, name);
        CHECK(report["accepted_pairs"] == 0);
        CHECK(report["skipped"]["non_finite_pair"] == 1);
    }
}

TEST_CASE("bfgs-keeps-valid-curvature-pairs", "[solver][bfgs][curvature]")
{
    spdlog::logger logger("bfgs-valid-pairs", std::make_shared<spdlog::sinks::null_sink_mt>());

    // A one-dimensional secant recovers the exact curvature of a quadratic,
    // over the whole range of scales, so the safeguarded direction still
    // lands on the minimizer. These are the controls the guard must not move.
    for (const double curvature : {1e-8, 1e-3, 1., 1e3, 1e8})
    {
        for (const auto &[name, strategy] : both_strategies(json::object(), logger))
        {
            CAPTURE(name, curvature);
            ScalarQuadratic problem(curvature);
            const Eigen::VectorXd x0 = scalar(1.), x1 = scalar(0.5);
            Eigen::VectorXd g0, g1, direction;
            problem.gradient(x0, g0);
            problem.gradient(x1, g1);

            const json report = feed_pair(*strategy, name, x0, g0, x1 - x0, g1 - g0, direction);
            CHECK(direction[0] == Approx(-0.5)); // straight to the minimizer
            CHECK(report["accepted_pairs"] == 1);
            CHECK(report["skipped_pairs"] == 0);
            CHECK(report["history_resets"] == 0);
        }
    }
}

TEST_CASE("bfgs-discards-an-approximation-that-cannot-produce-descent", "[solver][bfgs][curvature]")
{
    spdlog::logger logger("bfgs-poisoned", std::make_shared<spdlog::sinks::null_sink_mt>());

    // Validating the pairs is not enough on its own: a pair can pass every
    // test and still leave an approximation whose recursion overflows at a
    // later gradient. s.y = 1 with |s| = 1e100 leaves an initial scale of
    // 1e-200, and the two-loop recursion then returns NaN at a gradient of
    // 1e300. The approximation is kept here (curvature_restart off) so that
    // the direction, not the refusal, is what discards it.
    json options;
    options["history_size"] = 6;
    options["curvature_restart"] = 0;
    LBFGS strategy(options, 1, logger);
    UnusedProblem problem;
    strategy.reset(1);
    strategy.reset_times();

    Eigen::VectorXd direction;
    REQUIRE(strategy.compute_update_direction(problem, scalar(0.), scalar(1e-100), direction));
    REQUIRE(strategy.compute_update_direction(problem, scalar(1e100), scalar(2e-100), direction));
    REQUIRE(guard_report(strategy, "L-BFGS")["accepted_pairs"] == 1);

    const Eigen::VectorXd huge = scalar(1e300);
    REQUIRE(strategy.compute_update_direction(problem, scalar(2e100), huge, direction));

    // The approximation is discarded and the reason recorded; the direction
    // is the steepest descent one, which is finite here.
    CHECK(identical(direction, -huge));
    const json report = guard_report(strategy, "L-BFGS");
    CHECK(report["skipped"]["non_finite_pair"] == 1);
    CHECK(report["history_resets"] == 1);
    CHECK(report["resets"]["non_finite_direction"] == 1);
}

TEST_CASE("bfgs-converges-on-the-audited-polynomials", "[solver][bfgs][curvature]")
{
    spdlog::logger logger("bfgs-polynomials", std::make_shared<spdlog::sinks::null_sink_mt>());

    // Both objectives are smooth and bounded below, and both made the
    // unsafeguarded strategies fail at their first iteration: one with a NaN
    // direction, the other with an ascent direction. There is no fallback
    // strategy here, so convergence is the strategy's own.
    const std::vector<std::pair<std::string, std::vector<double>>> objectives{
        {"x+x^2(x+1)^2", {0., 1., 1., 2., 1.}},
        {"x^4/4-x^2/2+0.1x", {0., 0.1, -0.5, 0., 0.25}},
    };

    for (const auto &[label, coefficients] : objectives)
    {
        for (const std::string &type : {"L-BFGS", "BFGS"})
        {
            for (const std::string &search : {"Armijo", "RobustArmijo", "Backtracking"})
            {
                CAPTURE(label, type, search);
                json params = pure_strategy(type, json::object());
                params["line_search"]["method"] = search;
                auto solver = Solver::create(params, {{"solver", "Eigen::LDLT"}}, 1, logger);

                Polynomial problem(coefficients);
                Eigen::VectorXd x = Eigen::VectorXd::Zero(1);
                REQUIRE_NOTHROW(solver->minimize(problem, x));
                CHECK(solver->status() == Status::GradNormTolerance);

                Eigen::VectorXd grad;
                problem.gradient(x, grad);
                CHECK(grad.norm() < 1e-10);

                // Every refused pair is reported, and the solve is the
                // strategy's own: there is no fallback in this configuration.
                const json report = solver->info()["curvature_guard"][type];
                CHECK(report["policy"] == "Skip");
                CHECK(report["skipped_pairs"] > 0);
            }
        }
    }
}

TEST_CASE("bfgs-leaves-valid-solves-alone", "[solver][bfgs][curvature]")
{
    spdlog::logger logger("bfgs-controls", std::make_shared<spdlog::sinks::null_sink_mt>());

    // Positive quadratics across scales, in one and in several variables: the
    // pairs are all valid, so no pair may be skipped and no approximation
    // discarded, at the unchanged tolerances.
    // A unit-curvature quadratic is not in the sweep: steepest descent lands
    // on its minimizer in one step, before a secant pair ever exists.
    for (const double curvature : {1e-8, 1e-3, 1e3, 1e8})
    {
        for (const std::string &type : {"L-BFGS", "BFGS"})
        {
            CAPTURE(curvature, type);
            json params = pure_strategy(type, json::object());
            params["grad_norm_tol"] = 1e-12 * curvature;
            auto solver = Solver::create(params, {{"solver", "Eigen::LDLT"}}, 1, logger);

            ScalarQuadratic problem(curvature);
            Eigen::VectorXd x = Eigen::VectorXd::Ones(3);
            REQUIRE_NOTHROW(solver->minimize(problem, x));
            CHECK(solver->status() == Status::GradNormTolerance);
            CHECK(x.norm() < 1e-12);

            const json report = solver->info()["curvature_guard"][type];
            CHECK(report["accepted_pairs"] > 0);
            CHECK(report["damped_pairs"] == 0);
            CHECK(report["skipped_pairs"] == 0);
            CHECK(report["history_resets"] == 0);
        }
    }
}

TEST_CASE("bfgs-restarts-when-the-approximation-goes-stale", "[solver][bfgs][curvature]")
{
    spdlog::logger logger("bfgs-restart", std::make_shared<spdlog::sinks::null_sink_mt>());

    // Refusing a pair leaves an approximation built from a stretch of the
    // solve that is over. Keeping it indefinitely stalls the solve: the stale
    // direction is still a descent direction, so the line search accepts it
    // and the solver's own fallback is never reached. Both policies stalled
    // that way on the fixed corpus, so the default discards the approximation
    // after one refusal. See docs/bfgs-curvature-safeguard-20260922.md.
    for (const int restart : {0, 1})
    {
        json options;
        options["curvature_restart"] = restart;
        for (const auto &[name, strategy] : both_strategies(options, logger))
        {
            CAPTURE(name, restart);
            UnusedProblem problem;
            strategy->reset(1);
            strategy->reset_times();

            // A valid pair first: the secant of f(x)=2x^2 between 1 and 0.5
            // recovers its curvature exactly.
            Eigen::VectorXd direction;
            REQUIRE(strategy->compute_update_direction(problem, scalar(1.), scalar(4.), direction));
            direction = Eigen::VectorXd::Zero(1);
            REQUIRE(strategy->compute_update_direction(problem, scalar(0.5), scalar(2.), direction));
            REQUIRE(direction[0] == Approx(-0.5));

            // Then a refused one: the step did not move the gradient at all.
            direction = Eigen::VectorXd::Zero(1);
            REQUIRE(strategy->compute_update_direction(problem, scalar(0.25), scalar(2.), direction));

            const json report = guard_report(*strategy, name);
            CHECK(report["skipped"]["zero_gradient_change"] == 1);
            if (restart == 0)
            {
                CHECK(direction[0] == Approx(-0.5)); // the approximation is kept
                CHECK(report["history_resets"] == 0);
            }
            else
            {
                CHECK(direction[0] == Approx(-2.)); // steepest descent again
                CHECK(report["history_resets"] == 1);
                CHECK(report["resets"]["unsupported_history"] == 1);

                // The strategy is still usable: the next valid pair builds a
                // new approximation instead of reviving the discarded one.
                direction = Eigen::VectorXd::Zero(1);
                REQUIRE(strategy->compute_update_direction(problem, scalar(0.125), scalar(1.5), direction));
                CHECK(direction[0] == Approx(-0.375)); // the new secant, curvature 4
                CHECK(guard_report(*strategy, name)["accepted_pairs"] == 2);
            }
        }
    }
}

TEST_CASE("bfgs-damping-stores-a-positive-curvature-pair", "[solver][bfgs][curvature]")
{
    spdlog::logger logger("bfgs-damping", std::make_shared<spdlog::sinks::null_sink_mt>());

    // The Damp policy is the alternative the corpus was used to compare. It
    // replaces a pair that carries no usable curvature by the nearest one
    // that does, measured in the current approximation, instead of refusing
    // it. It is not the default: it converged on fewer corpus rows and in
    // more iterations. The pair it stores must still be a valid one.
    const Eigen::VectorXd s = scalar(-0.1), y = scalar(0.099); // s.y = -0.0099
    for (const auto &[name, strategy] : both_strategies(json{{"curvature_policy", "Damp"}}, logger))
    {
        CAPTURE(name);
        Eigen::VectorXd direction;
        const json report = feed_pair(*strategy, name, scalar(0.), scalar(0.1), s, y, direction);
        CHECK(report["damped_pairs"] == 1);
        CHECK(report["skipped_pairs"] == 0);
        CHECK(report["history_resets"] == 0);

        // A damped pair is stored, so the direction is no longer the steepest
        // descent one, and it still has to point downhill.
        const Eigen::VectorXd grad = scalar(0.199);
        CHECK(direction[0] * grad[0] < 0);
        CHECK(direction[0] != Approx(-grad[0]));
    }
}

// ===========================================================================
// Objective generation
//
// A quasi-Newton pair is a secant of one function. A problem that retunes
// itself during a minimization -- a contact barrier stiffness or trim that
// moves in post_step -- makes the next pair subtract gradients of two
// different functions, which is a secant of neither and can carry any
// curvature at all (audit finding 2). The problem reports a new objective
// generation; the solver then discards the history before a pair is formed.
// ===========================================================================

namespace
{
    /// A quadratic whose curvature and minimizer change once, mid-solve, in
    /// post_step, the way a contact retune changes a barrier. `reports` off
    /// is the behavior before this signal existed: the change happens and
    /// nothing is told about it.
    class RetunedQuadratic : public Problem
    {
    public:
        RetunedQuadratic(const bool reports, const int retune_after)
            : reports(reports), retune_after(retune_after) {}

        double curvature() const { return retuned ? 4. : 0.25; }
        TVector minimizer(const int size) const
        {
            TVector m = TVector::Zero(size);
            if (retuned)
                m[0] = 0.5;
            return m;
        }

        double value(const TVector &x) override
        {
            return 0.5 * curvature() * (x - minimizer(int(x.size()))).squaredNorm();
        }
        void gradient(const TVector &x, TVector &grad) override
        {
            grad = curvature() * (x - minimizer(int(x.size())));
        }
        void hessian(const TVector &x, THessian &hess) override
        {
            hess.resize(x.size(), x.size());
            hess.setIdentity();
            hess *= curvature();
        }

        void post_step(const PostStepData &data) override
        {
            if (retuned || data.iter_num != retune_after)
                return;
            retuned = true;
            ++generation; // the objective is a different function from here
        }

        uint64_t objective_generation() const override { return reports ? generation : 0; }

        /// Rebuilding a cache for the same function is not a change.
        void solution_changed(const TVector &) override { ++cache_rebuilds; }

        const bool reports;
        const int retune_after;
        bool retuned = false;
        uint64_t generation = 0;
        int cache_rebuilds = 0;
    };

    /// Iterations recorded from the solver's own callback.
    struct Step
    {
        size_t iteration;
        double grad_norm;
        double x_delta_dot_grad;
    };

    /// -H*g with an empty history is exactly the steepest descent direction,
    /// whose slope is -|g|^2. That is the visible signature of a strategy
    /// that holds nothing.
    bool is_steepest_descent(const Step &step)
    {
        return std::abs(step.x_delta_dot_grad + step.grad_norm * step.grad_norm)
               <= 1e-12 * step.grad_norm * step.grad_norm;
    }

} // namespace

TEST_CASE("bfgs-discards-history-when-the-objective-changes", "[solver][bfgs][objective]")
{
    spdlog::logger logger("bfgs-objective", std::make_shared<spdlog::sinks::null_sink_mt>());

    for (const std::string &type : {"L-BFGS", "BFGS"})
    {
        for (const bool reports : {true, false})
        {
            CAPTURE(type, reports);
            json params = pure_strategy(type, json::object());
            params["grad_norm_tol"] = 1e-12;
            auto solver = Solver::create(params, {{"solver", "Eigen::LDLT"}}, 1, logger);

            std::vector<Step> steps;
            solver->set_iteration_callback([&steps](const Criteria &c) {
                steps.push_back({c.iterations, c.gradNorm, c.xDeltaDotGrad});
                return false;
            });

            RetunedQuadratic problem(reports, /*retune_after=*/1);
            Eigen::VectorXd x = Eigen::VectorXd::Ones(2);
            REQUIRE_NOTHROW(solver->minimize(problem, x));
            REQUIRE(problem.retuned);
            CHECK((x - problem.minimizer(2)).norm() < 1e-12);

            const json info = solver->info();
            const json report = info["curvature_guard"][type];
            const int steepest = int(std::count_if(steps.begin(), steps.end(), is_steepest_descent));
            if (reports)
            {
                // The solver was told, so the pair that would have spanned
                // the change was never formed: the first direction after it
                // is the steepest descent one, as at the start of the solve.
                CHECK(info["objective_changes"] == 1);
                CHECK(report["resets"]["objective_changed"] == 1);
                CHECK(steepest == 2);
            }
            else
            {
                // Without the signal the change is invisible and only the
                // stage 1 safeguard stands between the spanning pair and the
                // approximation -- and only when its curvature is invalid.
                CHECK(info["objective_changes"] == 0);
                CHECK(report["resets"].find("objective_changed") == report["resets"].end());
                CHECK(steepest == 1);
            }
        }
    }
}

TEST_CASE("bfgs-keeps-history-when-only-the-iterate-moves", "[solver][bfgs][objective]")
{
    spdlog::logger logger("bfgs-objective-control", std::make_shared<spdlog::sinks::null_sink_mt>());

    // The control of the test above: the same problem, never retuned. The
    // solver rebuilds the problem's caches at every trial point and must not
    // discard anything for it.
    for (const std::string &type : {"L-BFGS", "BFGS"})
    {
        CAPTURE(type);
        json params = pure_strategy(type, json::object());
        params["grad_norm_tol"] = 1e-12;
        auto solver = Solver::create(params, {{"solver", "Eigen::LDLT"}}, 1, logger);

        RetunedQuadratic problem(true, /*retune_after=*/-1);
        Eigen::VectorXd x = Eigen::VectorXd::Ones(2);
        REQUIRE_NOTHROW(solver->minimize(problem, x));
        REQUIRE_FALSE(problem.retuned);
        CHECK(problem.cache_rebuilds > 0);

        const json info = solver->info();
        CHECK(info["objective_changes"] == 0);
        CHECK(info["curvature_guard"][type]["history_resets"] == 0);
        CHECK(info["curvature_guard"][type]["skipped_pairs"] == 0);
    }
}

TEST_CASE("bfgs-objective-change-discards-the-stored-iterate", "[solver][bfgs][objective]")
{
    spdlog::logger logger("bfgs-objective-direct", std::make_shared<spdlog::sinks::null_sink_mt>());

    // The audit's reproduction, stated on the strategies themselves: an
    // accepted step of f=x^2/8 from x=1 to x=0.75, then f=2x^2. The pair
    // (s=-0.25, y=2.75) has s.y=-0.6875 and is a secant of neither function.
    for (const auto &[name, strategy] : both_strategies(json::object(), logger))
    {
        CAPTURE(name);
        UnusedProblem problem;
        strategy->reset(1);
        strategy->reset_times();

        Eigen::VectorXd direction;
        REQUIRE(strategy->compute_update_direction(problem, scalar(1.), scalar(0.25), direction));

        strategy->objective_changed(1);

        direction = Eigen::VectorXd::Zero(1);
        const Eigen::VectorXd grad = scalar(3.);
        REQUIRE(strategy->compute_update_direction(problem, scalar(0.75), grad, direction));

        CHECK(identical(direction, -grad));
        const json report = guard_report(*strategy, name);
        CHECK(report["accepted_pairs"] == 0);
        CHECK(report["skipped_pairs"] == 0); // no pair was even formed
        CHECK(report["history_resets"] == 1);
        CHECK(report["resets"]["objective_changed"] == 1);
    }
}

TEST_CASE("bfgs-curvature-alone-cannot-see-an-objective-change", "[solver][bfgs][objective]")
{
    spdlog::logger logger("bfgs-objective-positive", std::make_shared<spdlog::sinks::null_sink_mt>());

    // What stage 2 adds to stage 1. A pair that spans a change can carry
    // perfectly good curvature: the gradient at x=1 under f=x^2/2 is 1, the
    // gradient at x=0.75 under f=1.2*x^2/2 is 0.9, and the pair
    // (s=-0.25, y=-0.1) has s.y=0.025>0 and apparent curvature 0.4, which is
    // the curvature of neither function. The tests of stage 1 accept it; only
    // the reported change prevents it from being formed at all.
    for (const bool told : {false, true})
    {
        for (const auto &[name, strategy] : both_strategies(json::object(), logger))
        {
            CAPTURE(name, told);
            UnusedProblem problem;
            strategy->reset(1);
            strategy->reset_times();

            Eigen::VectorXd direction;
            REQUIRE(strategy->compute_update_direction(problem, scalar(1.), scalar(1.), direction));

            if (told)
                strategy->objective_changed(1);

            direction = Eigen::VectorXd::Zero(1);
            const Eigen::VectorXd grad = scalar(0.9); // 1.2 * 0.75
            REQUIRE(strategy->compute_update_direction(problem, scalar(0.75), grad, direction));

            const json report = guard_report(*strategy, name);
            CHECK(report["accepted_pairs"] == (told ? 0 : 1));
            CHECK(report["skipped_pairs"] == 0);
            CHECK(report["resets"].count("objective_changed") == (told ? 1u : 0u));
            if (told)
                CHECK(identical(direction, -grad));
            else
                CHECK_FALSE(identical(direction, -grad));
        }
    }
}

TEST_CASE("bfgs-iteration-diagnostics-distinguish-gradient-descent-escalation", "[solver][bfgs][diagnostics]")
{
    spdlog::logger logger("bfgs-iteration-diagnostics", std::make_shared<spdlog::sinks::null_sink_mt>());

    // A highly anisotropic fixed quadratic makes the scale discontinuity easy
    // to see. After the first accepted step, reject directions whose norm has
    // collapsed relative to the gradient. The L-BFGS history direction is
    // therefore rejected, while the separately configured GradientDescent
    // fallback is feasible. This is a diagnostic fixture, not an argument for
    // that validity rule or a change to fallback policy.
    class DirectionGateQuadratic : public Problem
    {
    public:
        double value(const TVector &x) override { return .5 * (1e6 * x[0] * x[0] + x[1] * x[1]); }
        void gradient(const TVector &x, TVector &grad) override
        {
            grad.resize(2);
            grad << 1e6 * x[0], x[1];
        }
        void hessian(const TVector &, THessian &hess) override
        {
            hess.resize(2, 2);
            hess.setZero();
            hess.coeffRef(0, 0) = 1e6;
            hess.coeffRef(1, 1) = 1;
        }
        bool is_step_valid(const TVector &x0, const TVector &x1) override
        {
            if (!gate)
                return true;
            TVector grad;
            gradient(x0, grad);
            const TVector step = x1 - x0;
            if (grad.norm() > 0 && step.norm() / grad.norm() > 0.5)
                admit_trial_sequence = true;
            return admit_trial_sequence;
        }
        void post_step(const PostStepData &data) override
        {
            if (data.solver_info.contains("iteration_diagnostics")
                && data.solver_info["iteration_diagnostics"].is_object())
            {
                iterations.push_back(data.solver_info["iteration_diagnostics"]);
                gate = true;
                admit_trial_sequence = false;
            }
        }

        bool gate = false;
        bool admit_trial_sequence = false;
        std::vector<json> iterations;
    } problem;

    json params = {
        {"solver", "L-BFGS"}, // public form: L-BFGS, then GradientDescent
        {"max_iterations", 2},
        {"allow_out_of_iterations", true},
        {"grad_norm_tol", 0},
        {"rel_grad_norm_tol", 0},
        {"first_grad_norm_tol", 0},
        {"line_search", {{"method", "Armijo"}}},
        {"advanced", {{"derivative_along_delta_x_tol", 0}, {"iteration_diagnostics", true}}}};
    auto solver = Solver::create(params, {{"solver", "Eigen::LDLT"}}, 1, logger);
    Eigen::VectorXd x = Eigen::VectorXd::Ones(2);
    REQUIRE_NOTHROW(solver->minimize(problem, x));

    REQUIRE(problem.iterations.size() == 2);
    const json &limited = problem.iterations[0];
    const json &fallback = problem.iterations[1];
    CHECK(limited["strategy"] == "L-BFGS");
    CHECK(limited["strategy_state"]["direction_source"] == "steepest_descent_initial_or_reset");
    CHECK(limited["accepted"]["endpoint"]["same_objective_as_initial_slope"] == true);
    CHECK(limited["accepted"]["endpoint"]["slope"].is_number());

    CHECK(fallback["strategy"] == "GradientDescent");
    CHECK(fallback["strategy_state"]["direction_source"] == "gradient_descent");
    CHECK(fallback["direction"]["norm_over_gradient_norm"] == Approx(1));
    REQUIRE(fallback["strategy_transitions_since_previous_accept"].size() == 1);
    const json &transition = fallback["strategy_transitions_since_previous_accept"][0];
    CHECK(transition["from"] == "L-BFGS");
    CHECK(transition["to"] == "GradientDescent");
    CHECK(transition["reason"] == "line_search_failed");
    REQUIRE(transition["failed_attempt"]["direction_norm"].is_number());
    const double rejected_norm = transition["failed_attempt"]["direction_norm"];
    const double gradient_direction_norm = fallback["direction"]["euclidean_norm"];
    CHECK(gradient_direction_norm / rejected_norm > 1e3);

    const json info = solver->info();
    CHECK(info["strategy_transition_counts"]["L-BFGS->GradientDescent:line_search_failed"] == 1);
    CHECK(info["direction_sources"]["L-BFGS"]["limited_memory"] == 1);
}

TEST_CASE("slope-tolerance-stops-only-hessian-directions", "[solver][bfgs][stopping]")
{
    spdlog::logger logger("slope-tolerance", std::make_shared<spdlog::sinks::null_sink_mt>());

    // An anisotropic fixed quadratic. After its first secant, L-BFGS scales
    // its direction by the stiff curvature (~1e-6), so |g.d| ~ 1e-6 while the
    // soft coordinate is still ~1 from the minimizer: the slope tolerance
    // below would stop it there. That tolerance is a Newton-decrement test
    // and must only end a solve whose direction solves with the Hessian.
    class Anisotropic : public Problem
    {
    public:
        double value(const TVector &x) override { return .5 * (1e6 * x[0] * x[0] + x[1] * x[1]); }
        void gradient(const TVector &x, TVector &grad) override
        {
            grad.resize(2);
            grad << 1e6 * x[0], x[1];
        }
        void hessian(const TVector &, THessian &hess) override
        {
            hess.resize(2, 2);
            hess.setZero();
            hess.coeffRef(0, 0) = 1e6;
            hess.coeffRef(1, 1) = 1;
        }
    };

    SECTION("a limited-memory direction runs on to the gradient criterion")
    {
        for (const bool pure : {true, false})
        {
            CAPTURE(pure);
            json params = pure_strategy("L-BFGS", json::object());
            if (!pure)
                params["solver"] = "L-BFGS"; // public form: L-BFGS, then GradientDescent
            params["advanced"]["derivative_along_delta_x_tol"] = 1e-4;
            params["x_delta_tol"] = 1e-3;
            params["allow_non_grad_convergence"] = true;
            auto solver = Solver::create(params, {{"solver", "Eigen::LDLT"}}, 1, logger);

            Anisotropic problem;
            Eigen::VectorXd x(2);
            x << 1, 1;
            REQUIRE_NOTHROW(solver->minimize(problem, x));
            CHECK(solver->status() == Status::GradNormTolerance);
            CHECK(x.norm() < 1e-9);
            CHECK_FALSE(solver->direction_solves_with_hessian());
            CHECK(solver->info()["active_strategy_solves_with_hessian"] == false);
        }
    }

    SECTION("Newton keeps its decrement test")
    {
        json params = {
            {"solver", "Newton"},
            {"grad_norm_tol", 1e-10},
            {"rel_grad_norm_tol", 0},
            {"max_iterations", 500},
            {"line_search", {{"method", "Armijo"}}},
            {"advanced", {{"derivative_along_delta_x_tol", 1e7}}}};
        auto solver = Solver::create(params, {{"solver", "Eigen::SimplicialLDLT"}}, 1, logger);

        Anisotropic problem;
        Eigen::VectorXd x(2);
        x << 1, 1;
        REQUIRE_NOTHROW(solver->minimize(problem, x));
        // |g.d| = g'H^-1 g = 1e6 + 1 < 1e7: stopped before the first step
        CHECK(solver->status() == Status::NotDescentDirection);
        CHECK(solver->current_criteria().iterations == 0);
        CHECK(solver->direction_solves_with_hessian());
    }
}
