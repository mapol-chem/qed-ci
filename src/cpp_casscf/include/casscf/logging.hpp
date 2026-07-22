#pragma once

#include <ostream>

namespace casscf {

// Print levels for the C++ port.
//
// Design agreed with the developer (see README.md's "Output and print levels"):
// the Python prints everything, deliberately, so problems are visible; the C++
// wants graded output instead.
//
//   Silent  library default. NOT merely "quiet by default" politeness -- the
//           27 ctest binaries and tools/validate_against_python.cpp produce or
//           parse output, so a library that prints unbidden would pollute the
//           test log and the validation harness. Drivers opt in.
//   Normal  energy change per orbital optimization / per CI optimization /
//           per macroiteration, plus the three consistency signals below.
//           Spin is reported BY EXCEPTION only -- a root is mentioned when it
//           deviates from the target, not otherwise.
//   Debug   per-orbital-iteration internals: trust radius evolution, hard
//           case, solver dispatch, subspace sizes, QN activation/reset,
//           Powell damping, step norms, predicted vs actual energy. Plus
//           residual and spin of EVERY root.
//   Trace   everything, shaped like the Python's current output -- including
//           the per-Davidson-iteration root table that ci_solver.c:1708
//           currently prints unconditionally. Its purpose is specific to this
//           port: make C++ vs Python divergence a diff. It can retire after
//           the TAMM handoff.
//
// Record format is deliberately machine-parseable: one record per line, a
// bracketed tag, then key=value fields in a stable order. This is a
// requirement rather than a nicety -- the entire runtime profile in
// validation/ci_cost_experiment/README.md was reconstructed by regex over
// Python logs, and that only worked because those lines happened to be
// consistent. Designing for it makes every production run self-profiling:
//
//   [macro] iter=3 E_ci=-199.543974661 E_rdm=-199.543974659 dE=+2.1e-09 ...
//   [ci]    macro=3 phase=micro micro=2 iters=26 E=-199.5439747 res=8.3e-10
//   [spin]  macro=3 phase=micro micro=2 root=2 S2=0.0142 target=0.0 dev=1.4e-02
//   [orb]   macro=3 phase=micro micro=2 step=4 rho=0.98 accept=1
//
// `phase=internal|micro` rather than a nested counter: internal optimization
// is a sibling of the microiteration loop, not a level inside it, so a dotted
// hierarchy could not express it -- and key=value greps cleanly.
enum class PrintLevel : int {
    Silent = 0,
    Normal = 1,
    Debug = 2,
    Trace = 3,
};

struct Logger {
    PrintLevel level = PrintLevel::Silent;
    std::ostream* os = nullptr; // null == discard, regardless of level

    bool enabled(PrintLevel required) const {
        return os != nullptr && static_cast<int>(level) >= static_cast<int>(required);
    }
};

// Emit one record if the level allows. A macro rather than a stream-like
// object returning a null sink, deliberately: the arguments must not be
// evaluated when disabled. Some Debug/Trace records format per-root or
// per-subspace-vector data inside loops that run thousands of times, and
// "format then throw away" would cost more than the computation being
// described (orbital_sigma is called 836 times per run at 0.086 ms each).
#define CASSCF_LOG(logger, required_level, stream_expr)                                            \
    do {                                                                                           \
        if ((logger).enabled(required_level)) {                                                    \
            *(logger).os << stream_expr << '\n';                                                   \
        }                                                                                          \
    } while (0)

// --- Spin monitoring -------------------------------------------------------
//
// Per the developer: of ci_solver.c's three spin mechanisms -- the S^4 penalty
// folded into H_diag (build_H_diag_cas_spin's F factor, ci_solver.c:691),
// S^2-based root selection inside davidson_spin (ci_solver.c:1507), and
// first_order_spin_projection (ci_solver.c:1355) -- **S^2 is the default**,
// because the other two do not balance accuracy against speed well. Runs
// typically carry up to ten roots.
//
// So the thing worth monitoring is simply whether each root still carries the
// expected S^2, since maintaining spin across orbital updates is the hard
// part. At Normal this is reported by exception; at Debug/Trace, every root.
//
// Note first_order_spin_projection, when it is used, has a silent failure
// mode worth surfacing: it is a `while (iteration < 20)` loop that breaks on
// |S^2 - S(S+1)| < 1e-10, so exhausting its 20 iterations returns a still
// contaminated vector with no error signalled. Its iteration count is also a
// leading indicator -- rising counts mean spin is becoming hard to hold
// several iterations before it actually fails -- and a real cost, since each
// iteration is a build_sigma_s_square over the full CI vector that is not
// labelled "build sigma" and so does not appear in the timing breakdown.
struct RootSpin {
    int root = 0;
    double s2 = 0.0;        // <S^2> of this root
    double target_s2 = 0.0; // S(S+1) for the requested spin
    double deviation = 0.0; // |s2 - target_s2|
    int projection_iters = -1; // first_order_spin_projection count, -1 if unused
    bool projection_failed = false; // hit its 20-iteration cap
};

// Default threshold for calling a root "spin-deviating" at Normal level.
//
// CALIBRATE THIS against a real run rather than trusting the default. Converged
// roots in the reference MgH+ logs show <S^2> ~ 1e-17, so anything above noise
// would do at convergence -- but mid-optimization, right after an orbital
// update, spin is legitimately less clean, and too tight a threshold would
// make Normal level cry wolf on every microiteration. 1e-4 is a deliberate
// middle ground: ~5e-5 of the O(1) gap between adjacent spin states (S^2 = 0,
// 2, 6), far above the observed 1e-17 noise floor, and far below any
// contamination that would matter physically.
inline constexpr double kSpinDeviationThreshold = 1e-4;

inline bool spin_deviates(const RootSpin& r, double threshold = kSpinDeviationThreshold) {
    return r.projection_failed || r.deviation > threshold;
}

} // namespace casscf
