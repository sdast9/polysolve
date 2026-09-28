# Armijo refuses an uphill direction — 2026-09-27

## Defect

All seven Debug jobs of PolySolve CI run 36295825495 (on `6099b9cd`) failed in
`nonlinear-easier`: six logs show `Armijo.cpp:24`, assertion
`armijo_criteria <= 0`; the Hybrid Debug log shows the same test aborting
without the text. Reproduced locally in a Debug build of `6099b9cd`; the
debugger at the assertion shows solver `ADAM`, line search `Armijo`, problem
`QuadraticProblem`, first random start, ADAM the active strategy (index 0).

Two contracts disagreed. `ADAM::is_direction_descent()` is `false`, so
`Solver::minimize` does not screen ADAM's direction for ascent (its momentum
can point uphill), and the direction goes to the line search as is.
`Armijo::init_compute_descent_step_size` asserted that it is a descent
direction. Before `c874cd5` (2026-09-22) ADAM's first direction was 0/0 and the
solver always escalated to its fallback before the line search, so the
assertion was unreachable; the ADAM repair exposed it. The nonlinear sources
are identical across the upstream merge (`448f1b8`..`6099b9c`), so the merge
did not cause it. Wolfe reached the same assertion through its RobustArmijo
fallback, which it takes for a non-descent direction.

In Release the assertion is compiled out: every trial step failed the
sufficient-decrease test, the search ran its whole backtracking budget (energy
and `solution_changed` at each step size) and failed, and the solver fell back
to gradient descent. The roundoff fallback could in principle accept one of
those uphill trials.

## Change (user decision 2026-09-27: the search fails)

`Backtracking` gains `admits_direction()`, checked right after
`init_compute_descent_step_size`. `Armijo` (and so `RobustArmijo` and Wolfe's
fallback) admits only a finite non-positive slope `delta_x · grad` — the slope
itself, since `c` may be 0. A refused direction returns NaN without trying a
step: the line search restores the problem, ends the interval, reports
`failure_stage = "descent_search"` with `rejected_direction = {reason: "ascent" |
"nonfinite_slope", slope}` in its diagnostics, and logs the slope (warning on
the final strategy). The solver's existing line-search-failure handling then
escalates as before. The assertion is removed, not weakened: the case it
guarded is now a stated outcome. Descent directions are unaffected.

## Verification

Native macOS arm64, AppleClang 21, Debug (tests on; optional linear solvers,
Hypre, AMGCL, Spectra, MPI off):

| Check | Result |
| --- | --- |
| `nonlinear-easier` before the change (`6099b9cd`) | aborts at `Armijo.cpp:24` (reproduces CI) |
| `nonlinear-easier` after | 600 assertions pass |
| New `line-search-refuses-uphill-direction` (Armijo, RobustArmijo, Wolfe × c ∈ {1e-4, 0}) | 48 assertions pass: NaN, `rejected_direction`, at most 3 energy evaluations; a descent direction still takes the full step |
| Full Debug suite | 62 cases / 2,879 assertions pass |

The committed `Backtracking.cpp` differs from the tested build only by
clang-format whitespace. PolyFEM's Newton-family directions are screened by the
solver before the line search, so its scenes do not reach this path (the five
public smokes log no refused direction). CI run
[36359256152](https://github.com/sdast9/polysolve/actions/runs/36359256152) on
`6a8c2cc9`: all 14 jobs pass — Linux, macOS and Windows, Debug and Release,
both index widths, and Hybrid Debug/Release (the seven Debug jobs failed on
`6099b9cd`).

Evidence: parent workspace `outputs/band-statistic/20260927/polysolve/`.
