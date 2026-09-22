#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/descent_strategies/BFGS.hpp>

#include <catch2/catch.hpp>
#include <spdlog/sinks/null_sink.h>

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
