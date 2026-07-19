// Smoke test for the casscf_c_backend link (ci_solver.c/orbital.c compiled
// from source + MKL + OpenMP, see CMakeLists.txt). Not a correctness test of
// any CASSCF physics -- just proves the extern "C" declarations in
// ci_orbital_backend.hpp actually match the real compiled symbols' ABI
// (argument types/order/register layout) and that the MKL/OpenMP runtime
// linkage resolves and runs without crashing. Real numerical validation of
// what these functions compute belongs to whatever calls them with real
// chemistry data (CasscfIntegralTransformer's own test, and eventually
// CasscfCiStateAverageSolver's).
#include "casscf/ci_orbital_backend.hpp"

#include <cstdio>
#include <vector>

int main() {
    // get_graph(N, n_o, Y): builds the arc-weight graph array for a
    // 2-electron-in-4-orbital active space (N=2, n_o=4) -- Y has size
    // N*(n_o-N+1)*3 = 2*3*3 = 18, per helper_PFCI.py's Y allocation formula
    // (see cpp_casscf/README.md's "What's still open" -> CiStateAverageSolver
    // notes). Just checking this doesn't crash and writes something
    // non-garbage-looking (Y is graph *offsets*, not a value with an
    // independently-derivable expected answer without re-implementing the
    // graph algorithm -- out of scope here).
    std::vector<int> Y(18, -1);
    get_graph(2, 4, Y.data());

    bool any_written = false;
    for (int v : Y) {
        if (v != -1) any_written = true;
    }

    if (!any_written) {
        std::printf("FAIL: get_graph did not write anything into Y\n");
        return 1;
    }

    std::printf("PASS: get_graph linked and ran without crashing, wrote into Y\n");
    return 0;
}
