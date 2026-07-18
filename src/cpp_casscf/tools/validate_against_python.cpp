// Replays real trust-region solver instances captured from a live Python
// helper_PFCI.py run (see cpp_casscf/validation/dump_lih_case.py and the
// CPP_CASSCF_VALIDATION_DIR-gated dump hooks in helper_PFCI.py) through the
// corresponding ported C++ solvers, and checks the step matches exactly.
//
// This is a standalone tool, not a ctest: it needs an external dump
// directory as input rather than being self-contained. Usage:
//   ./validate_against_python <dump_dir>   (defaults to ./dumps_lih)
//
// internal_lstrs_NNN/ cases (from internal_optimization3's inline LSTRS
// bisection, helper_PFCI.py:6996-7548) are replayed through LstrsSolver and
// expected to match to floating-point exactness -- both sides are dense,
// deterministic, and solve the identical bordered-eigenproblem bisection.
//
// gltr_NNN/ cases (from microiteration_optimization6's n_negative == 0
// dispatch to solve_gltr_trust_region, helper_PFCI.py:15045-15300ish) are
// replayed through GltrTrustRegionSolver with add_noise = false. Python's
// GLTR injects random gradient noise in production
// (helper_PFCI.py:15061-15064, "# NEW: Add microscopic noise to see hidden
// negative curvature") specifically to steer away from exact trust-region
// hard cases (gradient orthogonal to the most-negative-curvature
// eigenvector). An earlier version of this harness disabled that noise
// draw so both sides would see an identical, reproducible input -- but that
// removed the very mechanism the noise exists to provide, and sweeping
// across more molecules/active spaces surfaced real mismatches exactly on
// samples that were genuine hard cases with the noise turned off (tiny
// numpy/LAPACK-vs-Eigen floating-point differences sent the two sides down
// different, each individually valid, secular-equation branches). Fixed:
// the noise is left on (production behavior unchanged), and the dump hook
// instead captures the actual realized post-noise gradient
// (self._cpp_casscf_last_noised_gradient, set inside solve_gltr_trust_region
// / solve_gltr_with_operator right after the draw) and dumps *that* as
// "gradient" -- so replaying it here with add_noise = false reproduces the
// literal vector Python operated on, RNG mismatch sidestepped entirely
// rather than the noise's effect being removed.
//
// davidson_lstrs_NNN/ cases (from microiteration_optimization6's
// n_negative > 0 branch, the outer beta-bisection loop around
// Davidson_augmented_hessian_solve6) are trickier: Python's production path
// uses a genuinely subspace-approximate algorithm there, and this module's
// HessianGuessProvider has no real implementation yet (it needs
// build_orbital_hessian_guess, still unported -- see README "What's still
// open"), so there is no way to run DavidsonDrivenLstrsSolver the way
// Python actually would on this problem. Instead, following the same
// methodology already used in test_davidson_driven_lstrs_solver.cpp /
// test_davidson_expansion_loop.cpp (cross-validate against LstrsSolver's
// dense/exact answer under a guess provider forced to cover the whole
// space), this checks DavidsonDrivenLstrsSolver against LstrsSolver on the
// SAME real, materialized chemistry Hessian -- a genuine correctness check
// of the Davidson subspace/bisection machinery on real problem conditioning,
// just not a bit-exact reproduction of Python's own (approximate) answer.
// Python's actual step is also loaded and reported for information (not
// pass/fail): it's expected to be *close* to the dense reference, not
// identical, since production Davidson only sees a small guess subspace.
//
// qn_gltr_NNN/ and qn_bfgs_NNN/ cases (from microiteration_optimization6's
// qn_optimization == True dispatch, helper_PFCI.py:11144-11180) are the two
// QN sub-branches: qn_gltr is GLTR against the exact Hessian rebuilt at the
// QN reference point (solve_gltr_trust_region on U_zero/A_tilde_zero/G_blocks_zero,
// same function as the gltr_NNN cases, same noised-gradient capture); qn_bfgs
// is GLTR-with-operator against a running L-BFGS approximation
// (solve_gltr_with_operator wrapping get_bfgs_mv -- the *other* near-duplicate
// GLTR function, helper_PFCI.py:14850-15100ish, which needed its own
// identical noised-gradient-capture edit since it's not literal-duplicate
// code with solve_gltr_trust_region). Both are materialized to a dense
// matrix by unit-vector probing (orbital_sigma3 for qn_gltr, get_bfgs_mv
// directly for qn_bfgs -- no need to port L-BFGS to C++ at all, it's just
// some Hv function from the outside) and replayed exactly like gltr_NNN.
#include "casscf/davidson_driven_lstrs_solver.hpp"
#include "casscf/gltr_trust_region_solver.hpp"
#include "casscf/hessian_operator.hpp"
#include "casscf/lstrs_solver.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace casscf;
namespace fs = std::filesystem;

namespace {

int failures = 0;
int checked = 0;

// np.savetxt writes one row of whitespace-separated values per line for a
// 2D array, and one value per line for a 1D array -- both are handled by
// just reading whatever rows/columns are present.
Matrix load_text_matrix(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double v;
        while (iss >> v) row.push_back(v);
        if (!row.empty()) rows.push_back(row);
    }
    if (rows.empty()) return Matrix(0, 0);
    const int nrows = static_cast<int>(rows.size());
    const int ncols = static_cast<int>(rows[0].size());
    Matrix M(nrows, ncols);
    for (int i = 0; i < nrows; ++i) {
        if (static_cast<int>(rows[i].size()) != ncols) {
            throw std::runtime_error("ragged rows in " + path.string());
        }
        for (int j = 0; j < ncols; ++j) M(i, j) = rows[i][j];
    }
    return M;
}

Vector load_text_vector(const fs::path& path) {
    Matrix M = load_text_matrix(path);
    if (M.cols() == 1) return M.col(0);
    if (M.rows() == 1) return M.row(0).transpose();
    throw std::runtime_error("expected a vector at " + path.string());
}

double load_scalar(const fs::path& path) { return load_text_vector(path)(0); }

void report(const std::string& label, double step_err, double step_scale, double tol) {
    ++checked;
    const double rel_err = step_err / step_scale;
    if (rel_err > tol) {
        std::printf("FAIL: %-20s step error %.3e (rel %.3e, tol %.3e)\n",
                    label.c_str(), step_err, rel_err, tol);
        ++failures;
    } else {
        std::printf("PASS: %-20s step error %.3e (rel %.3e, tol %.3e)\n",
                    label.c_str(), step_err, rel_err, tol);
    }
}

void validate_internal_lstrs(const fs::path& dir) {
    const Matrix H = load_text_matrix(dir / "hessian.txt");
    const Vector g = load_text_vector(dir / "gradient.txt");
    const double trust_radius = load_scalar(dir / "trust_radius.txt");
    const Vector expected_step = load_text_vector(dir / "step.txt");

    auto op = std::make_shared<DenseHessianOperator>(H);
    LstrsSolver solver(op, /*max_iter=*/200);
    const TrustRegionResult result = solver.solve(g, trust_radius);

    const double err = (result.step - expected_step).norm();
    const double scale = std::max(1.0, expected_step.norm());
    report(dir.filename().string(), err, scale, 1e-6);
}

// Same pattern as the identically-named class in
// test_davidson_driven_lstrs_solver.cpp: supplies exact blocks of a
// materialized dense Hessian, so a guess provider configured for full
// subspace coverage makes DavidsonAugmentedHessianSolver's "iteration 1"
// exact rather than subspace-approximate.
class DenseGuessProviderWithGradient final : public HessianGuessProvider {
public:
    DenseGuessProviderWithGradient(Matrix hessian, Vector gradient)
        : hessian_(std::move(hessian)), gradient_(std::move(gradient)) {}

    void guess_block(const std::vector<int>& indices, Matrix& hessian_block, Vector& gradient_block) const override {
        const int m = static_cast<int>(indices.size());
        hessian_block = Matrix(m, m);
        gradient_block = Vector(m);
        for (int a = 0; a < m; ++a) {
            gradient_block(a) = gradient_(indices[a]);
            for (int b = 0; b < m; ++b) hessian_block(a, b) = hessian_(indices[a], indices[b]);
        }
    }

private:
    Matrix hessian_;
    Vector gradient_;
};

void validate_davidson_lstrs(const fs::path& dir) {
    const Matrix H = load_text_matrix(dir / "hessian.txt");
    const Vector g = load_text_vector(dir / "gradient.txt");
    const Vector m_diag = load_text_vector(dir / "m_diag.txt");
    const double trust_radius = load_scalar(dir / "trust_radius.txt");
    const Vector python_step = load_text_vector(dir / "step.txt");
    const int n = static_cast<int>(g.size());

    auto op = std::make_shared<DenseHessianOperator>(H);
    auto guess_provider = std::make_shared<DenseGuessProviderWithGradient>(H, g);

    DavidsonAugmentedHessianConfig davidson_config;
    davidson_config.large_problem_threshold = 0;
    davidson_config.large_problem_dim0 = n;
    davidson_config.max_guess_dimension = n;

    LstrsSolver lstrs_reference(op);
    DavidsonDrivenLstrsSolver davidson_lstrs(op, guess_provider, m_diag, {}, davidson_config);

    const TrustRegionResult r_lstrs = lstrs_reference.solve(g, trust_radius);
    const TrustRegionResult r_davidson = davidson_lstrs.solve(g, trust_radius);

    const double err = (r_davidson.step - r_lstrs.step).norm();
    const double scale = std::max(1.0, r_lstrs.step.norm());
    report(dir.filename().string() + " (vs dense LSTRS)", err, scale, 1e-4);

    const double python_vs_dense = (python_step - r_lstrs.step).norm();
    std::printf("INFO: %-20s Python's actual (subspace-approximate) step vs. dense "
                "LSTRS reference: %.3e (not pass/fail -- see file doc comment)\n",
                dir.filename().string().c_str(), python_vs_dense);
}

void validate_gltr(const fs::path& dir) {
    const Matrix H = load_text_matrix(dir / "hessian.txt");
    const Vector g = load_text_vector(dir / "gradient.txt");
    const Vector m_diag = load_text_vector(dir / "m_diag.txt");
    const double trust_radius = load_scalar(dir / "trust_radius.txt");
    const Vector expected_step = load_text_vector(dir / "step.txt");

    auto op = std::make_shared<DenseHessianOperator>(H);
    GltrConfig config;
    config.max_iter = 1000;
    config.tol = 1e-7;
    config.add_noise = false; // dump hook disabled Python's noise draw to match
    GltrTrustRegionSolver solver(op, m_diag, config);
    const TrustRegionResult result = solver.solve(g, trust_radius);

    const double err = (result.step - expected_step).norm();
    const double scale = std::max(1.0, expected_step.norm());
    report(dir.filename().string(), err, scale, 1e-6);
}

} // namespace

int main(int argc, char** argv) {
    const fs::path dump_dir = argc > 1 ? argv[1] : "dumps_lih";
    if (!fs::exists(dump_dir)) {
        std::printf("dump directory does not exist: %s\n", dump_dir.string().c_str());
        return 1;
    }

    std::vector<fs::path> internal_dirs, gltr_dirs, davidson_dirs, qn_gltr_dirs, qn_bfgs_dirs;
    for (const auto& entry : fs::directory_iterator(dump_dir)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("internal_lstrs_", 0) == 0) internal_dirs.push_back(entry.path());
        else if (name.rfind("davidson_lstrs_", 0) == 0) davidson_dirs.push_back(entry.path());
        else if (name.rfind("qn_gltr_", 0) == 0) qn_gltr_dirs.push_back(entry.path());
        else if (name.rfind("qn_bfgs_", 0) == 0) qn_bfgs_dirs.push_back(entry.path());
        else if (name.rfind("gltr_", 0) == 0) gltr_dirs.push_back(entry.path());
    }
    std::sort(internal_dirs.begin(), internal_dirs.end());
    std::sort(gltr_dirs.begin(), gltr_dirs.end());
    std::sort(davidson_dirs.begin(), davidson_dirs.end());
    std::sort(qn_gltr_dirs.begin(), qn_gltr_dirs.end());
    std::sort(qn_bfgs_dirs.begin(), qn_bfgs_dirs.end());

    for (const auto& d : internal_dirs) validate_internal_lstrs(d);
    for (const auto& d : gltr_dirs) validate_gltr(d);
    for (const auto& d : davidson_dirs) validate_davidson_lstrs(d);
    for (const auto& d : qn_gltr_dirs) validate_gltr(d);
    for (const auto& d : qn_bfgs_dirs) validate_gltr(d);

    std::printf("\n%d/%d cases passed.\n", checked - failures, checked);
    return failures == 0 ? 0 : 1;
}
