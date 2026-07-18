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
// internal_intermediates_NNN/, full_intermediates_NNN/, build_gradient_NNN/
// and build_hessian_diagonal_NNN/ cases (from internal_optimization3's
// build_intermediates_internal + build_gradient_and_hessian, and
// microiteration_optimization6's build_intermediates + build_gradient +
// build_hessian_diagonal) replay the ported intermediates-building
// functions (cpp_casscf/include/casscf/intermediates.hpp) against real
// captured (Hessian, gradient, RDM, ...) inputs and check the resulting
// A/G/gradient_tilde/hessian_tilde/hessian_diagonal tensors match Python's
// actual values -- this is the primary correctness check for that port,
// given how many einsum index-order derivations it required (see that
// header's doc comments for the specific reasoning behind each one).
#include "casscf/davidson_driven_lstrs_solver.hpp"
#include "casscf/gltr_trust_region_solver.hpp"
#include "casscf/hessian_guess.hpp"
#include "casscf/hessian_operator.hpp"
#include "casscf/intermediates.hpp"
#include "casscf/lstrs_solver.hpp"
#include "casscf/tensor_types.hpp"

#include <algorithm>
#include <cmath>
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

// Reads <base>.txt (flat, C-order raveled data, one value per line -- see
// _dump_cpp_casscf_validation_case's ndim>2 branch in helper_PFCI.py) plus
// <base>.shape.txt (the 4 dimensions, space-separated) and reconstructs a
// Tensor4. Tensor4 is RowMajor (see tensor_types.hpp), matching numpy's
// C-order ravel exactly, so this is a straight element-for-element copy via
// TensorMap, not a transpose.
Tensor4 load_tensor4(const fs::path& base_path) {
    std::ifstream shape_in(base_path.string() + ".shape.txt");
    if (!shape_in) throw std::runtime_error("cannot open shape file for " + base_path.string());
    int d0, d1, d2, d3;
    shape_in >> d0 >> d1 >> d2 >> d3;

    const Vector flat = load_text_vector(fs::path(base_path.string() + ".txt"));
    if (flat.size() != static_cast<long>(d0) * d1 * d2 * d3) {
        throw std::runtime_error("tensor size mismatch for " + base_path.string());
    }
    Eigen::TensorMap<const Tensor4> mapped(flat.data(), d0, d1, d2, d3);
    Tensor4 t = mapped; // copies out of `flat`'s storage before it goes out of scope
    return t;
}

Dimensions load_dims(const fs::path& path) {
    const Vector v = load_text_vector(path);
    Dimensions dims;
    dims.n_in_a = static_cast<int>(std::lround(v(0)));
    dims.n_act_orb = static_cast<int>(std::lround(v(1)));
    dims.n_virtual = static_cast<int>(std::lround(v(2)));
    dims.nmo = static_cast<int>(std::lround(v(3)));
    dims.n_occupied = static_cast<int>(std::lround(v(4)));
    return dims;
}

double tensor4_diff_norm(const Tensor4& a, const Tensor4& b) {
    double acc = 0.0;
    for (int i = 0; i < a.dimension(0); ++i)
        for (int j = 0; j < a.dimension(1); ++j)
            for (int k = 0; k < a.dimension(2); ++k)
                for (int l = 0; l < a.dimension(3); ++l) {
                    const double d = a(i, j, k, l) - b(i, j, k, l);
                    acc += d * d;
                }
    return std::sqrt(acc);
}

double tensor4_norm(const Tensor4& a) {
    double acc = 0.0;
    for (int i = 0; i < a.dimension(0); ++i)
        for (int j = 0; j < a.dimension(1); ++j)
            for (int k = 0; k < a.dimension(2); ++k)
                for (int l = 0; l < a.dimension(3); ++l) acc += a(i, j, k, l) * a(i, j, k, l);
    return std::sqrt(acc);
}

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

void validate_internal_intermediates(const fs::path& dir) {
    const Dimensions dims = load_dims(dir / "dims.txt");
    const Matrix occupied_fock_core = load_text_matrix(dir / "occupied_fock_core.txt");
    const Matrix occupied_d_cmo = load_text_matrix(dir / "occupied_d_cmo.txt");
    const Tensor4 occupied_J = load_tensor4(dir / "occupied_J");
    const Tensor4 occupied_K = load_tensor4(dir / "occupied_K");
    const Matrix D_tu_avg = load_text_matrix(dir / "D_tu_avg.txt");
    const Tensor4 D_tuvw_avg = load_tensor4(dir / "D_tuvw_avg");
    const Matrix Dpe_tu_avg = load_text_matrix(dir / "Dpe_tu_avg.txt");
    const double off_diagonal_constant = load_scalar(dir / "off_diagonal_constant.txt");
    const double omega = load_scalar(dir / "omega.txt");

    const Matrix expected_A1 = load_text_matrix(dir / "A1.txt");
    const Tensor4 expected_G1 = load_tensor4(dir / "G1");
    const Matrix expected_gradient_tilde1 = load_text_matrix(dir / "gradient_tilde1.txt");
    const Tensor4 expected_hessian_tilde1 = load_tensor4(dir / "hessian_tilde1");

    const SmallBlockIntermediates interm = build_intermediates_internal(
        occupied_fock_core, occupied_d_cmo, occupied_J, occupied_K, D_tu_avg, D_tuvw_avg,
        Dpe_tu_avg, off_diagonal_constant, omega, dims);

    const std::string label = dir.filename().string();
    report(label + " A", (interm.A - expected_A1).norm(), std::max(1.0, expected_A1.norm()), 1e-8);
    report(label + " G", tensor4_diff_norm(interm.G, expected_G1), std::max(1.0, tensor4_norm(expected_G1)), 1e-8);

    const GradientAndHessianResult gh = build_gradient_and_hessian(interm.A, interm.G, dims);
    report(label + " gradient_tilde", (gh.gradient_tilde - expected_gradient_tilde1).norm(),
           std::max(1.0, expected_gradient_tilde1.norm()), 1e-8);
    report(label + " hessian_tilde", tensor4_diff_norm(gh.hessian_tilde, expected_hessian_tilde1),
           std::max(1.0, tensor4_norm(expected_hessian_tilde1)), 1e-8);
}

void validate_full_intermediates(const fs::path& dir) {
    const Dimensions dims = load_dims(dir / "dims.txt");
    const Matrix H_spatial2 = load_text_matrix(dir / "H_spatial2.txt");
    const Matrix d_cmo = load_text_matrix(dir / "d_cmo.txt");
    const Tensor4 J = load_tensor4(dir / "J");
    const Tensor4 K = load_tensor4(dir / "K");
    const Matrix D_tu_avg = load_text_matrix(dir / "D_tu_avg.txt");
    const Tensor4 D_tuvw_avg = load_tensor4(dir / "D_tuvw_avg");
    const Matrix Dpe_tu_avg = load_text_matrix(dir / "Dpe_tu_avg.txt");
    const double off_diagonal_constant = load_scalar(dir / "off_diagonal_constant.txt");
    const double omega = load_scalar(dir / "omega.txt");

    const Matrix expected_A = load_text_matrix(dir / "A.txt");
    const Tensor4 expected_G = load_tensor4(dir / "G");

    const FullBlockIntermediates interm = build_intermediates(
        H_spatial2, d_cmo, J, K, D_tu_avg, D_tuvw_avg, Dpe_tu_avg, off_diagonal_constant, omega, dims);

    const std::string label = dir.filename().string();
    report(label + " A", (interm.A - expected_A).norm(), std::max(1.0, expected_A.norm()), 1e-8);
    report(label + " G", tensor4_diff_norm(interm.G, expected_G), std::max(1.0, tensor4_norm(expected_G)), 1e-8);
}

void validate_build_gradient(const fs::path& dir) {
    const Dimensions dims = load_dims(dir / "dims.txt");
    const Matrix U = load_text_matrix(dir / "U.txt");
    const Matrix A = load_text_matrix(dir / "A.txt");
    const Tensor4 G = load_tensor4(dir / "G");
    const Matrix expected_A_tilde_full = load_text_matrix(dir / "A_tilde.txt"); // (nmo, nmo)
    const Matrix expected_gradient_tilde = load_text_matrix(dir / "gradient_tilde.txt");

    const GradientResult result = build_gradient(U, A, G, dims);

    // Python's A_tilde is (nmo, nmo) but build_gradient only ever populates
    // its first n_occupied columns (see intermediates.hpp's doc comment) --
    // compare against just that block.
    const Matrix expected_A_tilde = expected_A_tilde_full.leftCols(dims.n_occupied);
    const std::string label = dir.filename().string();
    report(label + " A_tilde", (result.A_tilde - expected_A_tilde).norm(), std::max(1.0, expected_A_tilde.norm()), 1e-8);
    report(label + " gradient_tilde", (result.gradient_tilde - expected_gradient_tilde).norm(),
           std::max(1.0, expected_gradient_tilde.norm()), 1e-8);
}

void validate_build_hessian_diagonal(const fs::path& dir) {
    const Dimensions dims = load_dims(dir / "dims.txt");
    const Matrix U = load_text_matrix(dir / "U.txt");
    const Tensor4 G = load_tensor4(dir / "G");
    const Matrix A_tilde = load_text_matrix(dir / "A_tilde.txt"); // (nmo, nmo)
    const Matrix expected_hessian_diagonal = load_text_matrix(dir / "hessian_diagonal.txt");
    const Vector expected_reduced = load_text_vector(dir / "reduced_hessian_diagonal.txt");

    const HessianDiagonalResult result = build_hessian_diagonal(U, G, A_tilde, dims);

    const std::string label = dir.filename().string();
    report(label + " hessian_diagonal", (result.hessian_diagonal - expected_hessian_diagonal).norm(),
           std::max(1.0, expected_hessian_diagonal.norm()), 1e-8);
    report(label + " reduced_hessian_diagonal", (result.reduced_hessian_diagonal - expected_reduced).norm(),
           std::max(1.0, expected_reduced.norm()), 1e-8);
}

void validate_hessian_guess(const fs::path& dir) {
    const Dimensions dims = load_dims(dir / "dims.txt");
    const Matrix U = load_text_matrix(dir / "U.txt");
    const Matrix sym_A_tilde = load_text_matrix(dir / "sym_A_tilde.txt");
    const Vector reduced_gradient = load_text_vector(dir / "reduced_gradient.txt");
    const Tensor4 G = load_tensor4(dir / "G");
    const Vector idx_double = load_text_vector(dir / "idx.txt");

    std::vector<int> idx(idx_double.size());
    for (int i = 0; i < idx_double.size(); ++i) idx[i] = static_cast<int>(std::lround(idx_double(i)));

    const Matrix expected_hessian = load_text_matrix(dir / "guess_hessian.txt");
    const Vector expected_gradient = load_text_vector(dir / "guess_gradient.txt");

    const OrbitalHessianGuessProvider provider(U, sym_A_tilde, reduced_gradient, G,
                                                build_index_map(dims), dims.n_occupied);
    Matrix hessian_block;
    Vector gradient_block;
    provider.guess_block(idx, hessian_block, gradient_block);

    const std::string label = dir.filename().string();
    report(label + " guess_hessian", (hessian_block - expected_hessian).norm(),
           std::max(1.0, expected_hessian.norm()), 1e-8);
    report(label + " guess_gradient", (gradient_block - expected_gradient).norm(),
           std::max(1.0, expected_gradient.norm()), 1e-8);
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
    std::vector<fs::path> internal_interm_dirs, full_interm_dirs, build_gradient_dirs, hessian_diag_dirs;
    std::vector<fs::path> hessian_guess_dirs;
    for (const auto& entry : fs::directory_iterator(dump_dir)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("internal_lstrs_", 0) == 0) internal_dirs.push_back(entry.path());
        else if (name.rfind("davidson_lstrs_", 0) == 0) davidson_dirs.push_back(entry.path());
        else if (name.rfind("qn_gltr_", 0) == 0) qn_gltr_dirs.push_back(entry.path());
        else if (name.rfind("qn_bfgs_", 0) == 0) qn_bfgs_dirs.push_back(entry.path());
        else if (name.rfind("gltr_", 0) == 0) gltr_dirs.push_back(entry.path());
        else if (name.rfind("internal_intermediates_", 0) == 0) internal_interm_dirs.push_back(entry.path());
        else if (name.rfind("full_intermediates_", 0) == 0) full_interm_dirs.push_back(entry.path());
        else if (name.rfind("build_gradient_", 0) == 0) build_gradient_dirs.push_back(entry.path());
        else if (name.rfind("build_hessian_diagonal_", 0) == 0) hessian_diag_dirs.push_back(entry.path());
        else if (name.rfind("hessian_guess_", 0) == 0) hessian_guess_dirs.push_back(entry.path());
    }
    std::sort(internal_dirs.begin(), internal_dirs.end());
    std::sort(gltr_dirs.begin(), gltr_dirs.end());
    std::sort(davidson_dirs.begin(), davidson_dirs.end());
    std::sort(qn_gltr_dirs.begin(), qn_gltr_dirs.end());
    std::sort(qn_bfgs_dirs.begin(), qn_bfgs_dirs.end());
    std::sort(internal_interm_dirs.begin(), internal_interm_dirs.end());
    std::sort(full_interm_dirs.begin(), full_interm_dirs.end());
    std::sort(build_gradient_dirs.begin(), build_gradient_dirs.end());
    std::sort(hessian_diag_dirs.begin(), hessian_diag_dirs.end());
    std::sort(hessian_guess_dirs.begin(), hessian_guess_dirs.end());

    for (const auto& d : internal_interm_dirs) validate_internal_intermediates(d);
    for (const auto& d : full_interm_dirs) validate_full_intermediates(d);
    for (const auto& d : build_gradient_dirs) validate_build_gradient(d);
    for (const auto& d : hessian_diag_dirs) validate_build_hessian_diagonal(d);
    for (const auto& d : hessian_guess_dirs) validate_hessian_guess(d);
    for (const auto& d : internal_dirs) validate_internal_lstrs(d);
    for (const auto& d : gltr_dirs) validate_gltr(d);
    for (const auto& d : davidson_dirs) validate_davidson_lstrs(d);
    for (const auto& d : qn_gltr_dirs) validate_gltr(d);
    for (const auto& d : qn_bfgs_dirs) validate_gltr(d);

    std::printf("\n%d/%d cases passed.\n", checked - failures, checked);
    return failures == 0 ? 0 : 1;
}
