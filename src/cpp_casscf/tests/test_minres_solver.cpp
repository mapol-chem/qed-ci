// Tests for minres_solve (port of scipy.sparse.linalg.minres). Case 2 is
// not synthetic -- it's the real, ill-conditioned (condition number ~1410)
// hard_case==2 Hessian captured from a live H2O/6-31G SA-QED-CASSCF run via
// cpp_casscf/validation/dump_lih_case.py (internal_optimization3's inline
// LSTRS bisection, helper_PFCI.py:7545-7551), hardcoded here so this test
// doesn't depend on regenerating the (gitignored) sweep dumps. Python's
// actual captured step on this exact (H, g) was independently confirmed
// (offline, via scipy.sparse.linalg.minres called directly on these arrays)
// to match scipy's minres to 0.0 difference and an exact dense solve
// (H.ldlt().solve(-g)) to differ by ~3.05e-3 -- i.e. this is precisely the
// case that motivated porting scipy's actual MINRES instead of substituting
// a direct solve. See minres_solver.hpp's doc comment and README "Sweep
// findings".
#include "casscf/minres_solver.hpp"

#include <cmath>
#include <cstdio>

using namespace casscf;

namespace {

int failures = 0;

void expect_near(double actual, double expected, double tol, const char* label) {
    if (std::abs(actual - expected) > tol) {
        std::printf("FAIL: %s -- expected %.10f, got %.10f\n", label, expected, actual);
        ++failures;
    } else {
        std::printf("PASS: %s (%.10f ~= %.10f)\n", label, actual, expected);
    }
}

} // namespace

int main() {
    // --- Case 1: well-conditioned SPD system -- MINRES should converge to
    //     (essentially) the exact solution well within rtol=1e-5. ---
    {
        Matrix A(3, 3);
        A << 4.0, 1.0, 0.0,
             1.0, 3.0, 1.0,
             0.0, 1.0, 2.0;
        Vector b(3);
        b << 1.0, 2.0, 3.0;

        MinresResult result = minres_solve(A, b, 1e-5);
        Vector exact = A.ldlt().solve(b);
        expect_near((result.x - exact).norm(), 0.0, 1e-4, "well-conditioned: matches exact solve");
    }

    // --- Case 2: real, ill-conditioned hard_case==2 Hessian from a live
    //     H2O/6-31G run (condition number ~1410). MINRES at rtol=1e-5 must
    //     reproduce Python's actual (deliberately early-stopped) step --
    //     NOT the exact solve, which differs by ~3e-3. ---
    {
        Matrix H(8, 8);
        H << 4.395388456983866377e-01, -3.161729628450488105e-03, -2.435383346022362838e-04, -9.812268923149736199e-04, -5.717694136231932234e-01, 2.050778424409058307e-03, 1.603949458462620645e-03, 3.722816656088953017e-03,
             -3.161729628447879081e-03, 3.623790241057123218e-02, -3.176889077172950770e-03, -5.506280493044357627e-03, 4.890900197501730357e-04, -6.440006029026333276e-03, 2.424866419664240338e-03, 5.966780984231571765e-03,
             -2.435383346037212071e-04, -3.176889077174449572e-03, 3.401667319577672544e-02, 1.028871648584256393e-03, 9.673337949285771903e-04, 3.439654979233569554e-05, -1.925416448925952595e-02, 3.767765739038269127e-04,
             -9.812268923169720214e-04, -5.506280493043247404e-03, 1.028871648583701282e-03, 3.313612804757792851e-02, 4.559066896159180357e-03, 4.960497328579930969e-03, -1.467381299814064007e-03, 8.421687784977024104e-04,
             -5.717694136231898927e-01, 4.890900197501383412e-04, 9.673337949286010427e-04, 4.559066896159211582e-03, 4.040797021054944338e+01, -5.846090360514390183e-02, -1.085420985157333346e-01, -3.180813641674719583e-01,
             2.050778424409102976e-03, -6.440006029023184753e-03, 3.439654979236128271e-05, 4.960497328579911019e-03, -5.846090360514387407e-02, 2.126402459233173392e+00, -3.146107251594424947e-01, -6.121543155724997742e-01,
             1.603949458462595925e-03, 2.424866419664222558e-03, -1.925416448925637222e-02, -1.467381299814052732e-03, -1.085420985157333484e-01, -3.146107251594424947e-01, 1.882901146237779599e+00, 2.169498836024626176e-01,
             3.722816656088920491e-03, 5.966780984231585643e-03, 3.767765739038194317e-04, 8.421687785008474640e-04, -3.180813641674720138e-01, -6.121543155724996632e-01, 2.169498836024625898e-01, 1.810083574538941642e+00;

        Vector g(8);
        g << 2.136402586330743425e-05, -7.673385704108914851e-04, -7.553770299395495824e-03,
             -1.471254120976372981e-03, 1.610284961690960611e-04, 2.081839360532092287e-03,
             5.729823233163846077e-04, -1.148611361722669949e-03;

        Vector python_step(8);
        python_step << 4.991343675174342732e-04, 4.685482326040162071e-02, 2.267490802769254188e-01,
                       4.263763969071725918e-02, -3.988244278053273387e-06, -6.748678796205748393e-04,
                       1.878774490546386870e-03, -4.198348677718502866e-05;

        MinresResult result = minres_solve(H, -g, 1e-5);
        expect_near((result.x - python_step).norm(), 0.0, 1e-9,
                    "ill-conditioned hard_case==2: matches Python's real MINRES step");

        Vector exact = H.ldlt().solve(-g);
        const double exact_vs_python = (exact - python_step).norm();
        if (exact_vs_python < 1e-3) {
            std::printf("FAIL: ill-conditioned case: exact solve unexpectedly close to Python's "
                        "early-stopped answer (%.3e) -- test data may be stale\n", exact_vs_python);
            ++failures;
        } else {
            std::printf("PASS: ill-conditioned case: exact solve differs from Python's early-stopped "
                        "answer as expected (%.3e)\n", exact_vs_python);
        }
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
