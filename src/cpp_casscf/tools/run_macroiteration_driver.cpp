// End-to-end integration runner: wires all 4 real MacroiterationDriver
// collaborators (CasscfCiStateAverageSolver, CasscfInternalOptimizationStep,
// CasscfMicroiterationOptimizationStep, CasscfIntegralTransformer) together
// and calls MacroiterationDriver::run on real captured LiH chemistry --
// the "macroiteration_bootstrap_000" dump (helper_PFCI.py, right before
// `while macroiteration < 1000:` starts) plus a "macroiteration_convergence_000"
// dump (captured at Python's own convergence break) for comparison.
//
// Standalone tool, not a ctest -- needs an external dump directory as
// input, same reasoning as validate_against_python.cpp. Usage:
//   ./run_macroiteration_driver <dump_dir> [--disable-qn]   (dump_dir
//   defaults to ../validation/dumps_macro_lih)
//
// --disable-qn constructs CasscfMicroiterationOptimizationStep with
// QuasiNewtonPolicy{enabled=false} instead of the class's own default
// (enabled=true) -- lets sweep_macroiterations.sh regression-check that
// disabling QN reproduces this port's pre-QN-wiring behavior exactly
// against the older dumps_macro_sweep_* fixtures (captured with Python's
// own QN trigger patched off), while the default (QN enabled) run compares
// against fixtures captured from Python's real, QN-enabled default.
//
// Generate the dump with (from cpp_casscf/validation/):
//   python dump_lih_case.py --dump-dir dumps_macro_lih
#include "casscf/ci_setup.hpp"
#include "casscf/ci_state_average_solver.hpp"
#include "casscf/integral_transformer.hpp"
#include "casscf/internal_optimization_step.hpp"
#include "casscf/macroiteration_driver.hpp"
#include "casscf/microiteration_optimization_step.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace casscf;
namespace fs = std::filesystem;

namespace {

// Same loading conventions as validate_against_python.cpp (np.savetxt-
// compatible text matrices/vectors, plus a <name>.shape.txt sidecar for
// rank>2 arrays -- see _dump_cpp_casscf_validation_case in helper_PFCI.py).
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
        if (static_cast<int>(rows[i].size()) != ncols) throw std::runtime_error("ragged rows in " + path.string());
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

Tensor4 load_tensor4(const fs::path& base_path) {
    std::ifstream shape_in(base_path.string() + ".shape.txt");
    if (!shape_in) throw std::runtime_error("cannot open shape file for " + base_path.string());
    int d0, d1, d2, d3;
    shape_in >> d0 >> d1 >> d2 >> d3;

    const Vector flat = load_text_vector(fs::path(base_path.string() + ".txt"));
    if (flat.size() != static_cast<long>(d0) * d1 * d2 * d3)
        throw std::runtime_error("tensor size mismatch for " + base_path.string());
    Eigen::TensorMap<const Tensor4> mapped(flat.data(), d0, d1, d2, d3);
    Tensor4 t = mapped;
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

} // namespace

int main(int argc, char** argv) {
    fs::path dump_dir = fs::path("../validation/dumps_macro_lih");
    bool disable_qn = false;
    // Output verbosity (logging.hpp). Default stays Silent so the existing
    // sweep scripts, which parse this tool's own summary lines, are unaffected.
    PrintLevel print_level = PrintLevel::Silent;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--disable-qn") {
            disable_qn = true;
        } else if (arg.rfind("--print-level=", 0) == 0) {
            const std::string v = arg.substr(14);
            if (v == "silent") print_level = PrintLevel::Silent;
            else if (v == "normal") print_level = PrintLevel::Normal;
            else if (v == "debug") print_level = PrintLevel::Debug;
            else if (v == "trace") print_level = PrintLevel::Trace;
            else { std::fprintf(stderr, "unknown --print-level=%s (silent|normal|debug|trace)\n", v.c_str()); return 2; }
        } else {
            dump_dir = fs::path(arg);
        }
    }
    fs::path case_dir = dump_dir / "macroiteration_bootstrap_000";
    fs::path convergence_dir = dump_dir / "macroiteration_convergence_000";

    Dimensions dims = load_dims(case_dir / "dims.txt");
    const int n_act = dims.n_act_orb;

    // config_int = [n_act_a, N_p, num_det, davidson_roots, davidson_indim,
    //               davidson_maxdim, davidson_maxiter, ignore_dse_terms]
    const Vector config_int = load_text_vector(case_dir / "config_int.txt");
    const int n_act_a = static_cast<int>(std::lround(config_int(0)));
    const int N_p = static_cast<int>(std::lround(config_int(1)));
    const int num_det = static_cast<int>(std::lround(config_int(2)));
    const int davidson_roots = static_cast<int>(std::lround(config_int(3)));
    const int davidson_indim = static_cast<int>(std::lround(config_int(4)));
    const int davidson_maxdim = static_cast<int>(std::lround(config_int(5)));
    const int davidson_maxiter = static_cast<int>(std::lround(config_int(6)));
    const bool ignore_dse_terms = config_int(7) != 0.0;

    // config_double = [omega, Enuc, d_c, d_exp, davidson_threshold, target_spin]
    const Vector config_double = load_text_vector(case_dir / "config_double.txt");
    const double omega = config_double(0);
    const double Enuc = config_double(1);
    const double d_c = config_double(2);
    const double d_exp = config_double(3);
    const double davidson_threshold = config_double(4);
    const double target_spin = config_double(5);

    Matrix H_spatial2 = load_text_matrix(case_dir / "H_spatial2.txt");
    Matrix d_cmo = load_text_matrix(case_dir / "d_cmo.txt");
    Tensor4 J = load_tensor4(case_dir / "J");
    Tensor4 K = load_tensor4(case_dir / "K");
    RowMajorMatrix twoeint = load_text_matrix(case_dir / "twoeint.txt");
    Matrix eigenvecs = load_text_matrix(case_dir / "eigenvecs.txt");
    const double avg_energy0 = load_scalar(case_dir / "avg_energy.txt");
    const double E_core = load_scalar(case_dir / "E_core.txt");
    Matrix D_tu_avg = load_text_matrix(case_dir / "D_tu_avg.txt");
    Tensor4 D_tuvw_avg = load_tensor4(case_dir / "D_tuvw_avg");
    Matrix Dpe_tu_avg = load_text_matrix(case_dir / "Dpe_tu_avg.txt");
    Vector weight = load_text_vector(case_dir / "weight.txt");

    const double python_avg_energy = load_scalar(convergence_dir / "avg_energy.txt");
    const int python_macroiterations = static_cast<int>(std::lround(load_scalar(convergence_dir / "macroiteration.txt")));

    std::printf("Loaded bootstrap case: nmo=%d n_occupied=%d n_in_a=%d n_act_orb=%d n_virtual=%d\n", dims.nmo,
                dims.n_occupied, dims.n_in_a, dims.n_act_orb, dims.n_virtual);
    std::printf("n_act_a=%d N_p=%d num_det=%d davidson_roots=%d target_spin=%.4f\n", n_act_a, N_p, num_det,
                davidson_roots, target_spin);
    std::printf("Python: initial avg_energy=%.12f, converged avg_energy=%.12f after %d macroiterations\n",
                avg_energy0, python_avg_energy, python_macroiterations);

    CasscfContext context;
    context.log.level = print_level;
    context.log.os = &std::cout;
    context.H_spatial2 = H_spatial2;
    context.d_cmo = d_cmo;
    context.J = J;
    context.K = K;
    context.twoeint = twoeint;
    context.E_core = E_core;
    context.D_tu_avg = D_tu_avg;
    context.D_tuvw_avg = D_tuvw_avg;
    context.Dpe_tu_avg = Dpe_tu_avg;

    // occupied_J/occupied_K are written via in-place operator() (only a
    // sub-block at a time, see commit_ci_solver_inputs in
    // microiteration_optimization_step.cpp) before
    // transform_macroiteration ever runs (which is what properly resizes
    // and refreshes them, once per macroiteration -- see
    // CasscfIntegralTransformer::transform_macroiteration), so they must
    // already be sized (n_occupied)^4 before the driver's first pass --
    // see CasscfContext::occupied_J's doc comment for the corrected shape.
    // A default-constructed (0-sized) Tensor4 here segfaults the first
    // time CasscfMicroiterationOptimizationStep commits into it.
    context.occupied_J = Tensor4(dims.n_occupied, dims.n_occupied, dims.n_occupied, dims.n_occupied);
    context.occupied_J.setZero();
    context.occupied_K = Tensor4(dims.n_occupied, dims.n_occupied, dims.n_occupied, dims.n_occupied);
    context.occupied_K.setZero();
    context.occupied_h1 = Matrix::Zero(dims.n_occupied, dims.n_occupied);
    context.occupied_d_cmo = Matrix::Zero(dims.n_occupied, dims.n_occupied);
    context.occupied_fock_core = Matrix::Zero(dims.n_occupied, dims.n_occupied);
    context.gkl2 = Matrix::Zero(n_act, n_act);
    context.U_total = Matrix::Identity(dims.nmo, dims.nmo);

    CasscfPhysicalConstants constants;
    constants.N_p = N_p;
    constants.num_det = num_det;
    constants.omega = omega;
    constants.Enuc = Enuc;
    constants.d_c = d_c;
    constants.d_exp = d_exp;
    constants.weight = weight;

    CasscfCiConfig ci_config;
    ci_config.n_act_a = n_act_a;
    ci_config.davidson_roots = davidson_roots;
    ci_config.davidson_threshold = davidson_threshold;
    ci_config.davidson_indim = davidson_indim;
    ci_config.davidson_maxdim = davidson_maxdim;
    ci_config.davidson_maxiter = davidson_maxiter;
    ci_config.target_spin = target_spin;
    ci_config.ignore_dse_terms = ignore_dse_terms;

    std::printf("Constructing CasscfCiSetup...\n");
    CasscfCiSetup setup(dims, ci_config, constants, H_spatial2, J, K, E_core);
    std::printf("H_dim=%d indim=%d maxdim=%d\n", setup.H_dim(), setup.indim(), setup.maxdim());

    CasscfCiStateAverageSolver ci_solver(dims, ci_config, constants, setup, context);
    CasscfIntegralTransformer integral_transformer(context, dims);
    CasscfInternalOptimizationStep internal_step(dims, constants, ci_solver, integral_transformer);
    QuasiNewtonPolicy qn_policy;
    qn_policy.enabled = !disable_qn;
    std::printf("QuasiNewtonPolicy::enabled=%d (%s)\n", qn_policy.enabled, disable_qn ? "--disable-qn passed" : "default");
    CasscfMicroiterationOptimizationStep microiteration_step(dims, constants, ci_solver, /*max_microiterations=*/20,
                                                              qn_policy);

    MacroiterationDriverConfig driver_config;
    driver_config.dims = dims;
    // Matches Python's own per-macroiteration print exactly (helper_PFCI.py:
    // 2597-2600, "Macroiteration %d old CI energy %f new CI energy %f") so a
    // caller can grep/diff the two logs directly -- see
    // validation/compare_macroiterations.py.
    driver_config.on_macroiteration_end = [](int macroiteration, double old_avg_energy, double new_avg_energy) {
        std::printf("Macroiteration %d old CI energy %.12f new CI energy %.12f\n", macroiteration, old_avg_energy,
                    new_avg_energy);
    };

    MacroiterationDriver driver(driver_config, ci_solver, internal_step, microiteration_step, integral_transformer);

    std::printf("\nRunning MacroiterationDriver::run...\n\n");
    MacroiterationResult result = driver.run(eigenvecs, avg_energy0, context);

    std::printf("\n--- Result ---\n");
    std::printf("converged=%d macroiterations_run=%d\n", result.converged, result.macroiterations_run);
    std::printf("C++  avg_energy=%.12f\n", result.avg_energy);
    std::printf("Python avg_energy=%.12f\n", python_avg_energy);
    std::printf("|diff|=%.3e\n", std::abs(result.avg_energy - python_avg_energy));

    return 0;
}
