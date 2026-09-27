# Upstream integration — 2026-09-26

## Scope and compatibility

Merge upstream `polyfem/polysolve@da4e7feb27af94b722015c774c501c2ace4e33d2`
into `iteration-callback` at `448f1b8e0da72101d17dcbf1f49da5f647fe0315`.
The merge adds CPUHybrid/GPUHybrid/cuDSS linear solvers, nano-MPI integration,
large-index build support and CI coverage. It has no textual conflicts.

The complete `src/polysolve/nonlinear` tree remains identical to the fork's
pre-merge version. This preserves iteration callbacks, objective-generation
tracking, BFGS safeguards, feasibility-respecting Wolfe searches and the
Newton-only slope/step-length stopping rules.

Upstream's build default is retained: `POLYSOLVE_WITH_MPI=ON` on non-Windows
hosts, OFF on Windows. Its nano-MPI implementation uses in-process rank threads;
it does not require an external MPI launcher. CUDA remains optional and OFF by
default. Adding these solver choices does not select a new production solver.

Integration adjustments:

- The direct BFGS tests now explicitly link the private LBFGSpp header target.
  A fresh build reproduced the missing `LBFGSpp/BFGSMat.h` include; an older
  local build had supplied its path manually through `CMAKE_CXX_FLAGS`.
- Hypre is pinned to `ba9c640318c5f96aeb9d2afc95ddc9243b2cd40e`, the validated
  snapshot of upstream's moving `thread-mpi-backend` branch. nano-MPI v0.1.1
  resolves to `456871cfa71744702889f476d372c0b53fd6dca2`.
- Corrected the hybrid CI comment about the MPI default and removed six
  upstream trailing-whitespace lines. No numerical algorithm was edited.

## Validation

Native host: macOS arm64, AppleClang 21, Release. Fresh builds have no manual
LBFGSpp compiler include flag. Both use the default MPI-enabled configuration;
the second enables `POLYSOLVE_LARGE_INDEX`. The large-index build disables the
32-bit-only Accelerate sparse solver interface as intended (Accelerate may
still supply BLAS/LAPACK to other libraries).

Default and large-index builds: pass. Local validation completed 2026-09-27:

- Default enabled suite: **61 cases / 2,962 assertions pass**.
- Large-index enabled suite: **61 cases / 2,848 assertions pass**.
- CPUHybrid, 32-cubed grid (32,768 unknowns): 1/2/4 nano-MPI ranks all exit zero
  with relative residuals `3.668e-11`, `2.281e-11`, `4.927e-11`, respectively,
  below the unchanged `1e-8` check. These are correctness checks, not controlled
  performance comparisons.
- The hybrid driver's linked libraries contain no external MPI runtime.
- The nonlinear source tree is unchanged from `448f1b8`.

The first unit run had five matrix-file loading failures because two builds
shared an ExternalProject test-data checkout and the second build recreated it
while the first suite was reading. Both fixture roots are now isolated at the
same pinned data revision `8a2eff19a33ccfd395c9294342e2c5c2babf2626`, and the
validation rerun is sequential. Those initial logs are retained.

CUDA compilation/execution and native Linux/Windows results are outside the
local validation. Newly available solvers are not a claim of improved speed
or convergence on the user's scenes.

## Evidence

Parent workspace: `outputs/polysolve-upstream/20260926/`. See `PROGRESS.md`,
`configure-default.log`, `build-default.log` (initial missing-header failure),
`build-default-repaired.log`, `configure-large-index-isolated-data.log`,
`build-large-index-isolated-data.log`, `suite-results.json`, `hybrid-results.json`
and the corresponding raw test logs. The preserved PolyFEM baseline is
`PolyFEM_bin-baseline-ad7f41622`, with `baseline-build-info.json`.
