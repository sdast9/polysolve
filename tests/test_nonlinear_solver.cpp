//////////////////////////////////////////////////////////////////////////
#include "autodiff.h"
#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/BoxConstraintSolver.hpp>
#include <polysolve/nonlinear/Problem.hpp>
#include <polysolve/nonlinear/line_search/LineSearch.hpp>
#include <limits>
#include <map>
#include <polysolve/Utils.hpp>
#include <polysolve/Types.hpp>
#include <polysolve/linear/Solver.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <polysolve/JSONUtils.hpp>
#include <catch2/catch.hpp>

//////////////////////////////////////////////////////////////////////////

using namespace polysolve;
using namespace polysolve::nonlinear;

DECLARE_DIFFSCALAR_BASE();

static const int N_RANDOM = 5;

typedef DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd> AutodiffScalarHessian;
typedef Eigen::Matrix<AutodiffScalarHessian, Eigen::Dynamic, 1> AutodiffHessian;

class TestProblem : public Problem
{
public:
    virtual std::vector<TVector> solutions() = 0;
    virtual int size() = 0;

    virtual TVector min() = 0;
    virtual TVector max() = 0;

    virtual std::string name() = 0;
};

class AnalyticTestProblem : public TestProblem
{
protected:
    virtual AutodiffScalarHessian eval_fun(const AutodiffHessian &x) const = 0;

private:
    AutodiffScalarHessian wrap(const TVector &x)
    {
        DiffScalarBase::setVariableCount(x.size());

        AutodiffHessian dx(x.size());

        for (long i = 0; i < x.size(); ++i)
            dx(i) = AutodiffScalarHessian(i, x(i));

        return eval_fun(dx);
    }

public:
    double value(const TVector &x) override
    {
        return wrap(x).getValue();
    }
    void gradient(const TVector &x, TVector &gradv) override
    {
        gradv = wrap(x).getGradient();
    }
    void hessian(const TVector &x, THessian &hessian) override
    {
        hessian = wrap(x).getHessian().sparseView();
    }
    void hessian(const TVector &x, Eigen::MatrixXd &hessian) override
    {
        hessian = wrap(x).getHessian();
    }
};

// f=(x(0)+2)^2 + (x(1)-3)^2 +(x(2)-1)^2
// f'=2(x(0)+2), 2(x(1)-3), 2(x(2)-1)
// f'' identy
class QuadraticProblem : public TestProblem
{
public:
    double value(const TVector &x) override
    {
        return (x(0) + 2) * (x(0) + 2) + //
               (x(1) - 3) * (x(1) - 3) + //
               (x(2) - 1) * (x(2) - 1);
    }
    void gradient(const TVector &x, TVector &gradv) override
    {
        gradv.resize(3);
        gradv(0) = 2 * (x(0) + 2);
        gradv(1) = 2 * (x(1) - 3);
        gradv(2) = 2 * (x(2) - 1);
    }
    void hessian(const TVector &x, THessian &hessian) override
    {
        hessian.resize(3, 3);
        hessian = sparse_identity(hessian.rows(), hessian.cols());
        hessian *= 2;
    }
    void hessian(const TVector &x, Eigen::MatrixXd &hessian) override
    {
        hessian.resize(3, 3);
        hessian.setIdentity();
        hessian *= 2;
    }

    std::vector<TVector> solutions() override
    {
        TVector res(3, 1);
        res << -2, 3, 1;
        return {res};
    }

    TVector min() override
    {
        TVector res(3, 1);
        res << 0, 0, 0;
        return res;
    }
    TVector max() override
    {
        TVector res(3, 1);
        res << 1, 1, 1;
        return res;
    }

    int size() override { return 3; }
    std::string name() override { return "Quadratic"; }
};

class Rosenbrock : public AnalyticTestProblem
{
    AutodiffScalarHessian eval_fun(const AutodiffHessian &x) const override
    {
        AutodiffScalarHessian res = AutodiffScalarHessian(0.0);
        for (int i = 0; i < x.size() - 1; ++i)
            res += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);

        return res;
    }

public:
    std::string name() override { return "Rosenbrock"; }

    int size() override { return 10; }

    std::vector<TVector> solutions() override
    {
        TVector res(size(), 1);
        res.setOnes();
        return {res};
    }

    TVector min() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= -5;
        return res;
    }
    TVector max() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= 5;
        return res;
    }
};

class Sphere : public AnalyticTestProblem
{
    AutodiffScalarHessian eval_fun(const AutodiffHessian &x) const override
    {
        AutodiffScalarHessian res = AutodiffScalarHessian(0.0);
        for (int i = 0; i < x.size(); ++i)
            res += x[i] * x[i];
        return res;
    }

public:
    std::string name() override { return "Sphere"; }
    int size() override { return 10; }

    std::vector<TVector> solutions() override
    {
        TVector res(size(), 1);
        res.setZero();
        return {res};
    }

    TVector min() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= -5;
        return res;
    }
    TVector max() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= 5;
        return res;
    }
};

class Beale : public AnalyticTestProblem
{
    AutodiffScalarHessian eval_fun(const AutodiffHessian &x) const override
    {
        return (1.5 - x[0] + x[0] * x[1]) * (1.5 - x[0] + x[0] * x[1]) +                 //
               (2.25 - x[0] + x[0] * x[1] * x[1]) * (2.25 - x[0] + x[0] * x[1] * x[1]) + //
               (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]) * (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]);
    }

public:
    std::string name() override { return "Beale"; }
    int size() override { return 2; }

    std::vector<TVector> solutions() override
    {
        TVector res(size(), 1);
        res << 3, 0.5;
        return {res};
    }

    TVector min() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= -2;
        return res;
    }
    TVector max() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= 6;
        return res;
    }
};

class InequalityConstraint : public Problem
{
public:
    InequalityConstraint(const double upper_bound) : upper_bound_(upper_bound) {}
    double value(const TVector &x) override { return x(0) - upper_bound_; }
    void gradient(const TVector &x, TVector &gradv) override
    {
        gradv.setZero(x.size());
        gradv(0) = 1;
    }
    void hessian(const TVector &x, THessian &hessian) override {}

private:
    const double upper_bound_;
};

void test_solvers(const std::vector<std::string> &solvers, const int iters, const bool exceptions_are_errors)
{
    std::vector<std::unique_ptr<TestProblem>> problems;
    problems.push_back(std::make_unique<QuadraticProblem>());
    if (!exceptions_are_errors)
        problems.push_back(std::make_unique<Rosenbrock>());
    problems.push_back(std::make_unique<Sphere>());
    problems.push_back(std::make_unique<Beale>());

    json solver_params, linear_solver_params;
    solver_params["line_search"] = {};
    solver_params["max_iterations"] = iters;
    solver_params["rel_grad_norm_tol"] = 0;

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test-logger");
    logger->set_level(spdlog::level::info);
    TestProblem::TVector g;
    // Which solves a fallback strategy finished, reported rather than hidden:
    // a converged solve here does not say the configured strategy converged.
    std::map<std::string, int> escalated; // solver/line search/problem -> solves
    int converged = 0, abandoned = 0, escalations = 0;
    for (auto &prob : problems)
    {
        for (auto solver_name : solvers)
        {
            if (solver_name == "BFGS" || solver_name == "DenseNewton")
                linear_solver_params["solver"] = "Eigen::LDLT";
            else
                linear_solver_params["solver"] = "Eigen::SimplicialLDLT";

            solver_params["solver"] = solver_name;

            for (const auto &ls : line_search::LineSearch::available_methods())
            {
                if (exceptions_are_errors && ls == "None")
                    continue;
                solver_params["line_search"]["method"] = ls;

                TestProblem::TVector x(prob->size());
                x.setZero();

                if (exceptions_are_errors)
                {
                    x.setRandom();
                    x /= 10;
                    x += prob->solutions()[0];
                }

                for (int i = 0; i < N_RANDOM; ++i)
                {
                    auto solver = Solver::create(solver_params,
                                                 linear_solver_params,
                                                 characteristic_length,
                                                 *logger);

                    try
                    {
                        solver->minimize(*prob, x);

                        double err = std::numeric_limits<double>::max();
                        for (auto sol : prob->solutions())
                            err = std::min(err, (x - sol).norm());
                        if (err >= 1e-7)
                        {
                            prob->gradient(x, g);
                            err = g.norm();
                        }
                        INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                        CHECK(err < 1e-7);
                        if (err >= 1e-7)
                            break;
                        ++converged;
                        if (!solver->info()["strategy_transition_counts"].empty())
                        {
                            ++escalated[solver_name + "/" + ls + "/" + prob->name()];
                            ++escalations;
                        }
                    }
                    catch (const std::exception &)
                    {
                        if (exceptions_are_errors)
                        {
                            INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                            CHECK(false);
                        }
                        else
                        {
                            ++abandoned;
                            break;
                        }
                    }

                    x.setRandom();
                    if (exceptions_are_errors)
                    {
                        x.setRandom();
                        x /= 10;
                        x += prob->solutions()[0];
                    }
                    else
                    {
                        x += prob->min();
                        x.array() *= (prob->max() - prob->min()).array();
                    }
                }
            }
        }
    }
    std::string report = fmt::format("{} converged solves, {} abandoned on an exception{}; {} escalated to a fallback strategy on the way",
                                     converged, abandoned, exceptions_are_errors ? " (none allowed)" : " (permitted here)", escalations);
    for (const auto &[combination, count] : escalated)
        report += fmt::format("\n  {} x{}", combination, count);
    WARN(report);
}

void test_solvers_gradient_fd(const bool full_fd)
{
    std::unique_ptr<TestProblem> problem = std::make_unique<QuadraticProblem>();
    std::string solver_name = "L-BFGS";

    json solver_params, linear_solver_params;
    solver_params["line_search"] = {};
    solver_params["max_iterations"] = 100;
    if (full_fd)
        solver_params["advanced"]["apply_gradient_fd"] = "FullFiniteDiff";
    else
        solver_params["advanced"]["apply_gradient_fd"] = "DirectionalDerivative";
    solver_params["solver"] = solver_name;

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("gradient-fd-test-logger");
    logger->set_level(spdlog::level::info);
    TestProblem::TVector g;
    linear_solver_params["solver"] = "Eigen::LDLT";

    for (const auto &ls : line_search::LineSearch::available_methods())
    {
        solver_params["line_search"]["method"] = ls;

        TestProblem::TVector x(problem->size());
        x.setZero();

        for (int i = 0; i < N_RANDOM; ++i)
        {
            auto solver = Solver::create(solver_params,
                                         linear_solver_params,
                                         characteristic_length,
                                         *logger);

            try
            {
                solver->minimize(*problem, x);

                double err = std::numeric_limits<double>::max();
                for (auto sol : problem->solutions())
                    err = std::min(err, (x - sol).norm());
                if (err >= 1e-7)
                {
                    problem->gradient(x, g);
                    err = g.norm();
                }
                INFO("solver: " + solver_name + " LS: " + ls + " problem " + problem->name());
                CHECK(err < 1e-7);
                if (err >= 1e-7)
                    break;
            }
            catch (const std::exception &)
            {
                break;
            }

            x.setRandom();
            x += problem->min();
            x.array() *= (problem->max() - problem->min()).array();
        }
    }
}

// A permissive stress test, deliberately: random starts over wide boxes, and an
// exception ends a combination without failing it. Its WARN line reports how
// many solves were abandoned and which finished on a fallback strategy. The
// deterministic BFGS regressions are "bfgs-deterministic-*".
TEST_CASE("nonlinear-stress-permissive", "[solver][stress]")
{
    test_solvers(Solver::available_solvers(), 1000, false);
    // test_solvers({"L-BFGS"}, 1000, false);
}

TEST_CASE("nonlinear-fallbacks", "[solver]")
{
    json solver_params = R"({"solver": [
        {
            "type": "Newton"
        },
        {
            "type": "RegularizedNewton"
        },
        {
            "type": "L-BFGS",
            "history_size": 10
        },
        {
            "type": "StochasticGradientDescent",
            "erase_component_probability": 0.5
        }
    ]
    })"_json;
    json linear_solver_params;
    linear_solver_params["solver"] = "Eigen::SimplicialLDLT";

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("nonlinear-fallbacks-test-logger");
    logger->set_level(spdlog::level::info);
    TestProblem::TVector g;
    auto prob = std::make_unique<QuadraticProblem>();

    TestProblem::TVector x(prob->size());
    x.setZero();

    for (int i = 0; i < N_RANDOM; ++i)
    {
        auto solver = Solver::create(solver_params,
                                     linear_solver_params,
                                     characteristic_length,
                                     *logger);

        solver->minimize(*prob, x);

        double err = std::numeric_limits<double>::max();
        for (auto sol : prob->solutions())
            err = std::min(err, (x - sol).norm());
        if (err >= 1e-7)
        {
            prob->gradient(x, g);
            err = g.norm();
        }
        CHECK(err < 1e-7);
        if (err >= 1e-7)
            break;

        x.setRandom();
    }
}

TEST_CASE("nonlinear-gradient-fd", "[solver]")
{
    test_solvers_gradient_fd(false);
    test_solvers_gradient_fd(true);
}

// Exceptions fail here, but the string-configured fallback may finish a solve;
// the WARN line lists every solve that needed it.
TEST_CASE("nonlinear-easier", "[solver]")
{
    test_solvers(Solver::available_solvers(), 5000, true);
}

TEST_CASE("nonlinear-box-constraint", "[solver]")
{
    std::vector<std::unique_ptr<TestProblem>> problems;
    problems.push_back(std::make_unique<QuadraticProblem>());
    problems.push_back(std::make_unique<Rosenbrock>());
    problems.push_back(std::make_unique<Sphere>());
    problems.push_back(std::make_unique<Beale>());

    json solver_params, linear_solver_params;
    solver_params["box_constraints"] = {};
    solver_params["iterations_per_strategy"] = {5, 5};
    solver_params["box_constraints"]["bounds"] = std::vector<double>({{0, 4}});
    solver_params["box_constraints"]["max_change"] = 4;

    solver_params["max_iterations"] = 1000;
    solver_params["line_search"] = {};

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("nonlinear-box-constraint-test-logger");
    logger->set_level(spdlog::level::info);

    for (auto &prob : problems)
    {
        for (auto solver_name : BoxConstraintSolver::available_solvers())
        {
            solver_params["solver"] = solver_name;

            for (const auto &ls : line_search::LineSearch::available_methods())
            {
                if (ls == "None" && solver_name != "MMA")
                    continue;
                if (solver_name == "MMA" && ls != "None")
                    continue;
                solver_params["line_search"]["method"] = ls;

                auto solver = BoxConstraintSolver::create(solver_params,
                                                          linear_solver_params,
                                                          characteristic_length,
                                                          *logger);

                QuadraticProblem::TVector x(prob->size());
                x.setConstant(3);

                for (int i = 0; i < N_RANDOM; ++i)
                {
                    try
                    {
                        solver->minimize(*prob, x);

                        INFO("solver: " + solver_params["solver"].get<std::string>() + " LS: " + ls);

                        Eigen::VectorXd gradv;
                        prob->gradient(x, gradv);
                        CHECK(solver->compute_grad_norm(*prob, x, gradv) < 1e-7);
                    }
                    catch (const std::exception &)
                    {
                        // INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                        // CHECK(false);
                        break;
                    }

                    x.setRandom();
                    x.array() += 3;
                }
            }
        }
    }
}

TEST_CASE("nonlinear-box-constraint-input", "[solver]")
{
    std::vector<std::unique_ptr<TestProblem>> problems;
    problems.push_back(std::make_unique<QuadraticProblem>());

    json solver_params, linear_solver_params;
    solver_params["box_constraints"] = {};

    solver_params["max_iterations"] = 1000;
    solver_params["line_search"] = {};

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("nonlinear-box-constraint-input-test-logger");
    logger->set_level(spdlog::level::err);

    for (auto &prob : problems)
    {
        for (auto solver_name : BoxConstraintSolver::available_solvers())
        {
            solver_params["solver"] = solver_name;

            for (const auto &ls : line_search::LineSearch::available_methods())
            {
                if (ls == "None" && solver_name != "MMA")
                    continue;
                if (solver_name == "MMA" && ls != "None")
                    continue;
                solver_params["line_search"]["method"] = ls;

                QuadraticProblem::TVector x(prob->size());
                x.setConstant(3);

                Eigen::MatrixXd bounds(2, x.size());
                bounds.row(0).array() = 0;
                bounds.row(1).array() = 4;
                solver_params["box_constraints"]["bounds"] = bounds;
                Eigen::MatrixXd max_change(1, x.size());
                max_change.array() = 4;
                solver_params["box_constraints"]["max_change"] = 4;

                auto solver = BoxConstraintSolver::create(solver_params,
                                                          linear_solver_params,
                                                          characteristic_length,
                                                          *logger);

                try
                {
                    solver->minimize(*prob, x);

                    INFO("solver: " + solver_params["solver"].get<std::string>() + " LS: " + ls);

                    Eigen::VectorXd gradv;
                    prob->gradient(x, gradv);
                    CHECK(solver->compute_grad_norm(*prob, x, gradv) < 1e-7);
                }
                catch (const std::exception &)
                {
                    // INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                    // CHECK(false);
                    break;
                }
            }
        }
    }
}

TEST_CASE("MMA", "[solver]")
{
    std::vector<std::unique_ptr<TestProblem>> problems;
    problems.push_back(std::make_unique<QuadraticProblem>());
    problems.push_back(std::make_unique<Rosenbrock>());
    problems.push_back(std::make_unique<Sphere>());
    problems.push_back(std::make_unique<Beale>());

    json solver_params, linear_solver_params;
    solver_params["box_constraints"] = {};
    solver_params["box_constraints"]["bounds"] = std::vector<double>({{0, 4}});
    solver_params["box_constraints"]["max_change"] = 4;

    solver_params["max_iterations"] = 1000;
    solver_params["line_search"] = {};

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("MMA-test-logger");
    logger->set_level(spdlog::level::info);

    for (auto &prob : problems)
    {
        solver_params["solver"] = "MMA";
        solver_params["line_search"]["method"] = "None";

        auto solver = BoxConstraintSolver::create(
            solver_params, linear_solver_params, characteristic_length, *logger);

        auto c = std::make_shared<InequalityConstraint>(solver_params["box_constraints"]["bounds"][1]);
        dynamic_cast<BoxConstraintSolver &>(*solver).add_constraint(c);

        QuadraticProblem::TVector x(prob->size());
        x.setConstant(3);

        for (int i = 0; i < N_RANDOM; ++i)
        {
            try
            {
                solver->minimize(*prob, x);

                INFO("solver: " + solver_params["solver"].get<std::string>());

                Eigen::VectorXd gradv;
                prob->gradient(x, gradv);
                CHECK(solver->compute_grad_norm(*prob, x, gradv) < 1e-7);
            }
            catch (const std::exception &)
            {
                // INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                // CHECK(false);
                break;
            }

            x.setRandom();
            x.array() += 3;
        }
    }
}

TEST_CASE("sample", "[solver]")
{
    Rosenbrock rb;

    Eigen::VectorXd alphas;
    Eigen::VectorXd fs;
    Eigen::VectorXi valid;

    Eigen::VectorXd dir(rb.size());
    dir.setOnes();
    for (int i = 0; i < N_RANDOM; ++i)
    {
        rb.sample_along_direction(rb.solutions()[0], dir, 0, 1, 10, alphas, fs, valid);
        dir.setRandom();

        for (int i = 1; i < fs.size(); ++i)
            CHECK(fs[0] <= fs[i]);
    }
}

TEST_CASE("iteration-callback", "[solver][callback]")
{
    Rosenbrock problem;

    json solver_params, linear_solver_params;
    solver_params["solver"] = "Newton";
    solver_params["line_search"] = {};
    solver_params["line_search"]["method"] = "Backtracking";
    solver_params["max_iterations"] = 100;
    solver_params["rel_grad_norm_tol"] = 0;
    linear_solver_params["solver"] = "Eigen::SimplicialLDLT";

    static std::shared_ptr<spdlog::logger> logger =
        spdlog::stdout_color_mt("callback-test");
    logger->set_level(spdlog::level::warn);

    auto solver = Solver::create(solver_params, linear_solver_params, 1, *logger);

    int n_calls = 0;
    bool alpha_ok = true;
    solver->set_iteration_callback([&](const Criteria &crit) -> bool {
        ++n_calls;
        alpha_ok = alpha_ok && std::isfinite(crit.alpha) && crit.alpha > 0;
        return crit.iterations >= 2; // request an early stop
    });

    TestProblem::TVector x(problem.size());
    x.setZero();
    REQUIRE_NOTHROW(solver->minimize(problem, x));

    // The callback fired each iteration with a finite line-search alpha and
    // stopped the solver early (Status::ObjectiveCustomStop does not throw).
    CHECK(n_calls >= 1);
    CHECK(n_calls <= 4);
    CHECK(alpha_ok);

    // The solver must have stopped before converging.
    TestProblem::TVector g;
    problem.gradient(x, g);
    CHECK(g.norm() > 1e-7);
}

// f = ½ (a x₀² + x₁²) whose *measured* energy cannot fall below `floor`,
// as when summation roundoff hides the last decrements of a real energy.
// Gradient and Hessian are exact. Models the step-1 stall reproduced in
// PolyFEM (energy pinned near 1e-14 while ‖∇f‖ was still 40× the tolerance).
class FlooredQuadratic : public Problem
{
public:
    FlooredQuadratic(const double a, const double floor) : a(a), floor(floor) {}

    double value(const TVector &x) override
    {
        return std::max(0.5 * (a * x(0) * x(0) + x(1) * x(1)), floor);
    }
    void gradient(const TVector &x, TVector &gradv) override
    {
        gradv.resize(2);
        gradv(0) = a * x(0);
        gradv(1) = x(1);
    }
    void hessian(const TVector &x, THessian &hessian) override
    {
        hessian.resize(2, 2);
        hessian.setIdentity();
        hessian.coeffRef(0, 0) = a;
    }
    void hessian(const TVector &x, Eigen::MatrixXd &hessian) override
    {
        hessian.setIdentity(2, 2);
        hessian(0, 0) = a;
    }

    double a, floor;
};

TEST_CASE("line-search-energy-roundoff", "[solver][line_search]")
{
    static std::shared_ptr<spdlog::logger> logger =
        spdlog::stdout_color_mt("roundoff-test");
    logger->set_level(spdlog::level::warn);

    json linear_solver_params;
    linear_solver_params["solver"] = "Eigen::SimplicialLDLT";

    const std::string method = GENERATE("Armijo", "RobustArmijo");

    auto make_params = [&](const double use_grad_norm_tol, const double roundoff_tolerance) {
        json p;
        p["solver"] = "Newton";
        p["line_search"]["method"] = method;
        p["line_search"]["use_grad_norm_tol"] = use_grad_norm_tol;
        p["line_search"]["Armijo"]["roundoff_tolerance"] = roundoff_tolerance;
        p["max_iterations"] = 50;
        p["grad_norm_tol"] = 1e-10;
        p["rel_grad_norm_tol"] = 0;
        return p;
    };

    // Start where the exact decrease (1e-14) is below the measured floor.
    FlooredQuadratic problem(1, 2e-14);
    TestProblem::TVector x0(2);
    x0 << 1e-7, 1e-7;

    SECTION("use_grad_norm switch rescues the stall")
    {
        auto solver = Solver::create(make_params(1e-6, 0), linear_solver_params, 1, *logger);
        int iterations = 0;
        bool full_steps = true;
        solver->set_iteration_callback([&](const Criteria &crit) -> bool {
            ++iterations;
            full_steps = full_steps && crit.alpha == 1;
            return false;
        });
        TestProblem::TVector x = x0;
        REQUIRE_NOTHROW(solver->minimize(problem, x));
        CHECK(solver->status() == Status::GradNormTolerance);
        CHECK(x.norm() < 1e-10);
        // The exact Newton step is accepted at α=1 on the gradient criterion.
        CHECK(full_steps);
        CHECK(iterations <= 2);
    }

    SECTION("roundoff floor alone rescues the stall")
    {
        auto solver = Solver::create(make_params(0, 2.220446049250313e-16), linear_solver_params, 1, *logger);
        TestProblem::TVector x = x0;
        REQUIRE_NOTHROW(solver->minimize(problem, x));
        CHECK(solver->status() == Status::GradNormTolerance);
        CHECK(x.norm() < 1e-10);
    }

    SECTION("both disabled reproduces the pre-fix behavior")
    {
        auto solver = Solver::create(make_params(0, 0), linear_solver_params, 1, *logger);
        TestProblem::TVector x = x0;
        if (method == "Armijo")
        {
            // Every step looks like a non-decrease: the search collapses.
            REQUIRE_THROWS(solver->minimize(problem, x));
        }
        else
        {
            // RobustArmijo's gradient-integral estimate cancels exactly on
            // a full Newton step, so it only ever accepts α=1/2 here.
            bool halved = true;
            solver->set_iteration_callback([&](const Criteria &crit) -> bool {
                halved = halved && crit.alpha == 0.5;
                return false;
            });
            REQUIRE_NOTHROW(solver->minimize(problem, x));
            CHECK(halved);
        }
    }

    SECTION("the fallback rejects a growing gradient at equal measured energy")
    {
        // Gradient descent on an ill-conditioned floored quadratic: α ≥ 1/32
        // overshoots and *increases* ‖∇f‖ while the energy still reads
        // "unchanged". The fallback must reject those and accept 1/64.
        FlooredQuadratic stiff(100, 1e-8);
        json p = make_params(1e100, 2.220446049250313e-16); // always in the roundoff regime
        p["solver"] = "GradientDescent";
        p["max_iterations"] = 5;
        p["allow_out_of_iterations"] = true;
        auto solver = Solver::create(p, linear_solver_params, 1, *logger);
        double max_alpha = 0;
        solver->set_iteration_callback([&](const Criteria &crit) -> bool {
            max_alpha = std::max(max_alpha, crit.alpha);
            return false;
        });
        TestProblem::TVector x(2);
        x << 1e-5, 1e-5;
        REQUIRE_NOTHROW(solver->minimize(stiff, x));
        CHECK(max_alpha == Approx(1. / 64));
    }
}

// Smooth and bounded below. A small gradient can fall to zero at a strict
// local maximum, so gradient-norm reduction alone cannot justify a step.
class SmallNonconvex : public Problem
{
public:
    double value(const TVector &x) override
    {
        const double t = x[0];
        return 1e-7 * (.5 * t * t - t * t * t / 3 + .01 * t * t * (t - 1) * (t - 1));
    }
    void gradient(const TVector &x, TVector &g) override
    {
        const double t = x[0];
        g = TVector::Constant(1, 1e-7 * (t - t * t + .01 * (4 * t * t * t - 6 * t * t + 2 * t)));
    }
    void hessian(const TVector &x, THessian &h) override
    {
        const double t = x[0];
        h.resize(1, 1);
        h.coeffRef(0, 0) = 1e-7 * (1 - 2 * t + .01 * (12 * t * t - 12 * t + 2));
    }
};

TEST_CASE("line-search-small-gradient-keeps-energy-bound", "[solver][line_search]")
{
    using TVector = Problem::TVector;
    static auto logger = spdlog::stdout_color_mt("bounded-roundoff-test");
    logger->set_level(spdlog::level::warn);
    const std::string method = GENERATE("Armijo", "RobustArmijo");
    const double eps = std::numeric_limits<double>::epsilon();
    json p = {{"line_search", {{"method", method}, {"min_step_size", 1e-12}, {"max_step_size_iter", 100}, {"min_step_size_final", 1e-12}, {"max_step_size_iter_final", 100}, {"default_init_step_size", 1.}, {"step_ratio", .5}, {"Armijo", {{"c", 1e-4}, {"roundoff_tolerance", eps}}}, {"RobustArmijo", {{"delta_relative_tolerance", 1e-10}}}}}};
    auto ls = line_search::LineSearch::create(p, *logger);
    ls->set_is_final_strategy(true);
    ls->reset_times();
    ls->use_grad_norm_tol = 1e-6;

    SECTION("rejects a full step to a nonconvex local maximum")
    {
        SmallNonconvex f;
        TVector x = TVector::Constant(1, -.1), step = TVector::Constant(1, 1.1), g, g_max;
        const TVector maximum = TVector::Ones(1);
        f.gradient(x, g);
        f.gradient(maximum, g_max);
        CHECK(g.dot(step) < 0);
        CHECK(g.norm() < ls->use_grad_norm_tol);
        CHECK(g_max.norm() == 0);
        CHECK(f.value(maximum) > f.value(x) + 1e6 * eps);
        const double alpha = ls->line_search(x, step, f);
        CHECK(alpha == Approx(.125));
        CHECK(f.value(x + alpha * step) < f.value(x));
    }

    SECTION("characteristic energy scale accommodates bounded cancellation noise")
    {
        class CancellationQuadratic : public FlooredQuadratic
        {
        public:
            CancellationQuadratic() : FlooredQuadratic(1, 2e-14) {}
            double scale = 90000.; // the unit-cube PolyFEM characteristic scale
            double energy_norm_rescaling(const NormType) const override { return scale; }
            double value(const TVector &x) override
            {
                // Explicit synthetic evaluation error (up to 1e-13), not
                // the exact objective. Gradient/Hessian remain exact.
                return FlooredQuadratic::value(x) + 1e-13 * (1 - std::min(1., std::abs(x[0]) / 1e-7));
            }
        } f;
        const TVector x = TVector::Constant(2, 1e-7);
        TVector step = -x;
        CHECK(f.value(x + step) - f.value(x) > eps);
        CHECK(f.value(x + step) - f.value(x) < eps * f.scale);
        CHECK(ls->line_search(x, step, f) == 1.);
        for (const double scale : {1., 0., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        {
            CAPTURE(scale);
            f.scale = scale;
            step = -x;
            // Invalid scales fail closed; a smaller valid scale tightens
            // the bound. The gradient-integral estimate may accept a
            // partial step in RobustArmijo independently of this fallback.
            const double alpha = ls->line_search(x, step, f);
            CHECK((std::isnan(alpha) || (alpha > 0 && alpha < 1.)));
        }
    }
}

// ===========================================================================
// Deterministic quasi-Newton regressions (BFGS audit stage 5)
//
// The "nonlinear" matrix above is a permissive stress test: random starts, and
// an exception ends a combination without failing it. "nonlinear-easier"
// fails on exceptions but lets the string-configured fallback (GradientDescent)
// finish the solve. Neither says whether a BFGS strategy converged by itself.
// Here every start is fixed, the solver is the strategy alone, an exception is
// a failure, and the strategy that finished is checked.
// ===========================================================================

namespace
{
    struct DeterministicCase
    {
        std::shared_ptr<TestProblem> problem;
        std::vector<TestProblem::TVector> starts;
    };

    std::vector<DeterministicCase> deterministic_cases()
    {
        const auto vec = [](std::initializer_list<double> v) {
            TestProblem::TVector x(v.size());
            int i = 0;
            for (const double e : v)
                x[i++] = e;
            return x;
        };
        TestProblem::TVector rosenbrock_classic(10), alternating(10);
        for (int i = 0; i < 10; ++i)
        {
            rosenbrock_classic[i] = i % 2 == 0 ? -1.2 : 1.;
            alternating[i] = i % 2 == 0 ? 4. : -3.;
        }
        return {
            {std::make_shared<QuadraticProblem>(), {vec({0, 0, 0}), vec({5, -4, 2})}},
            {std::make_shared<Sphere>(), {TestProblem::TVector::Constant(10, 3.), alternating}},
            {std::make_shared<Beale>(), {vec({1, 1}), vec({0, 0})}},
            {std::make_shared<Rosenbrock>(), {rosenbrock_classic, TestProblem::TVector::Zero(10)}}};
    }
} // namespace

TEST_CASE("bfgs-deterministic-matrix", "[solver][bfgs][deterministic]")
{
    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("bfgs-deterministic");
    logger->set_level(spdlog::level::off);

    for (const std::string strategy : {"BFGS", "L-BFGS"})
    {
        for (const std::string ls : {"Armijo", "RobustArmijo", "Backtracking", "Wolfe"})
        {
            for (const auto &c : deterministic_cases())
            {
                for (size_t s = 0; s < c.starts.size(); ++s)
                {
                    INFO("strategy " << strategy << ", line search " << ls << ", problem " << c.problem->name() << ", start " << s);
                    json params;
                    params["solver"] = json::array({json{{"type", strategy}}});
                    params["line_search"]["method"] = ls;
                    params["grad_norm_tol"] = 1e-8;
                    params["rel_grad_norm_tol"] = 0;
                    params["max_iterations"] = 5000;
                    json linear = {{"solver", "Eigen::LDLT"}};
                    auto solver = Solver::create(params, linear, 1, *logger);

                    TestProblem::TVector x = c.starts[s];
                    bool threw = false;
                    try
                    {
                        solver->minimize(*c.problem, x);
                    }
                    catch (const std::exception &e)
                    {
                        threw = true;
                        UNSCOPED_INFO("exception: " << e.what());
                    }
                    CHECK_FALSE(threw);
                    if (threw)
                        continue;
                    CHECK(solver->status() == Status::GradNormTolerance);
                    CHECK(solver->info()["active_strategy"] == strategy);
                    CHECK(solver->info()["strategy_transition_counts"].empty());
                    double err = std::numeric_limits<double>::max();
                    for (const auto &sol : c.problem->solutions())
                        err = std::min(err, (x - sol).norm());
                    CHECK(err < 1e-6);
                }
            }
        }
    }
}

TEST_CASE("bfgs-deterministic-chain-records-the-finishing-strategy", "[solver][bfgs][deterministic]")
{
    // The usual string form appends GradientDescent as a fallback, so a
    // converged solve does not by itself say the quasi-Newton strategy stayed
    // valid (the audit's Rosenbrock observation). From the fixed starts it must
    // converge without an exception, and the strategy that finished and every
    // escalation are read back from the solver info.
    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("bfgs-deterministic-chain");
    logger->set_level(spdlog::level::off);

    for (const std::string strategy : {"BFGS", "L-BFGS"})
    {
        for (const std::string ls : {"Armijo", "RobustArmijo", "Backtracking", "Wolfe"})
        {
            for (const auto &c : deterministic_cases())
            {
                for (size_t s = 0; s < c.starts.size(); ++s)
                {
                    INFO("strategy " << strategy << ", line search " << ls << ", problem " << c.problem->name() << ", start " << s);
                    json params;
                    params["solver"] = strategy;
                    params["line_search"]["method"] = ls;
                    params["grad_norm_tol"] = 1e-8;
                    params["rel_grad_norm_tol"] = 0;
                    params["max_iterations"] = 5000;
                    auto solver = Solver::create(params, {{"solver", "Eigen::LDLT"}}, 1, *logger);

                    TestProblem::TVector x = c.starts[s];
                    REQUIRE_NOTHROW(solver->minimize(*c.problem, x));
                    CHECK(solver->status() == Status::GradNormTolerance);
                    const json &info = solver->info();
                    INFO("finished on " << info["active_strategy"] << ", transitions " << info["strategy_transition_counts"].dump());
                    CHECK(info["active_strategy"] == strategy);
                    CHECK(info["strategy_transition_counts"].empty());
                }
            }
        }
    }
}

TEST_CASE("box-constrained-methods-are-named-by-the-unconstrained-solver", "[solver]")
{
    // The spec shared by both solvers offers L-BFGS-B and MMA; the
    // unconstrained solver cannot run them and says why, rather than calling
    // them unrecognized (BFGS audit finding 5).
    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("box-constrained-names");
    logger->set_level(spdlog::level::off);
    for (const std::string name : BoxConstraintSolver::available_solvers())
    {
        INFO(name);
        CHECK_THROWS_WITH(Solver::create(json{{"solver", name}}, json::object(), 1, *logger),
                          Catch::Contains(name + " is a box-constrained method") && Catch::Contains("L-BFGS"));
        // BoxConstraintSolver still constructs it
        json boxed = {{"solver", name}, {"box_constraints", {{"bounds", std::vector<double>({0, 1})}, {"max_change", 1}}}};
        if (name == "MMA")
            boxed["line_search"] = {{"method", "None"}};
        CHECK_NOTHROW(BoxConstraintSolver::create(boxed, json::object(), 1, *logger));
    }
    CHECK_THROWS_WITH(Solver::create(json{{"solver", "NoSuchMethod"}}, json::object(), 1, *logger),
                      Catch::Contains("invalid input json") || Catch::Contains("Unrecognized solver type"));
}

TEST_CASE("dense-method-requirements-are-named", "[solver]")
{
    // BFGS audit stage 5: the requirements a forward caller can miss are
    // stated where they are enforced -- dense BFGS needs a dense linear
    // solver, the dense Newton strategies a problem with a dense Hessian.
    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("dense-requirements");
    logger->set_level(spdlog::level::off);
    CHECK_THROWS_WITH(Solver::create(json{{"solver", "BFGS"}}, json{{"solver", "Eigen::SimplicialLDLT"}}, 1, *logger),
                      Catch::Contains("BFGS linear solver must be dense") && Catch::Contains("Eigen::LDLT"));
    CHECK_NOTHROW(Solver::create(json{{"solver", "BFGS"}}, json{{"solver", "Eigen::LDLT"}}, 1, *logger));

    // A problem that assembles only a sparse Hessian, as PolyFEM's do.
    class SparseOnly : public Problem
    {
    public:
        double value(const TVector &x) override { return 0.5 * x.squaredNorm(); }
        void gradient(const TVector &x, TVector &g) override { g = x; }
        void hessian(const TVector &x, THessian &h) override { h = sparse_identity(x.size(), x.size()); }
    } problem;
    json params = {{"solver", json::array({json{{"type", "DenseNewton"}}})}, {"max_iterations", 10}};
    auto solver = Solver::create(params, json{{"solver", "Eigen::LDLT"}}, 1, *logger);
    Eigen::VectorXd x = Eigen::VectorXd::Ones(3);
    CHECK_THROWS_WITH(solver->minimize(problem, x),
                      Catch::Contains("Dense Hessian not implemented by this problem") && Catch::Contains("use a sparse Newton strategy"));
}

TEST_CASE("adam-takes-its-own-steps", "[solver][adam]")
{
    // ADAM counted its steps from 0, so its first bias correction divided by
    // 1 - beta^0 = 0 and every first direction was NaN: the solver escalated
    // to GradientDescent before ADAM took a step, and again after each reset
    // (the escalation report of stage 5 showed it on every ADAM solve). It
    // also never kept its moment estimates. Deterministic, strategy alone.
    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("adam-own-steps");
    logger->set_level(spdlog::level::off);
    for (const std::string strategy : {"ADAM", "StochasticADAM"})
    {
        INFO(strategy);
        QuadraticProblem problem;
        json params;
        params["solver"] = json::array({json{{"type", strategy}}});
        params["line_search"]["method"] = "None";
        params["grad_norm_tol"] = 1e-6;
        params["rel_grad_norm_tol"] = 0;
        params["max_iterations"] = 20000;
        auto solver = Solver::create(params, json::object(), 1, *logger);
        TestProblem::TVector x = TestProblem::TVector::Zero(3);
        REQUIRE_NOTHROW(solver->minimize(problem, x));
        CHECK(solver->status() == Status::GradNormTolerance);
        CHECK(solver->info()["strategy_transition_counts"].empty());
        CHECK((x - problem.solutions()[0]).norm() < 1e-5);
    }
}
