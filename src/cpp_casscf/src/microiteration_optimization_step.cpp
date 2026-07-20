#include "casscf/microiteration_optimization_step.hpp"

#include "casscf/davidson_driven_lstrs_solver.hpp"
#include "casscf/gltr_trust_region_solver.hpp"
#include "casscf/hessian_guess.hpp"
#include "casscf/hessian_operator.hpp"
#include "casscf/internal_optimization.hpp" // step_control
#include "casscf/intermediates.hpp"
#include "casscf/linear_equation_solve.hpp"
#include "casscf/microiteration_ci_integrals_transform.hpp"
#include "casscf/microiteration_energy.hpp"
#include "casscf/minres_solver.hpp"
#include "casscf/orbital_rotation.hpp"
#include "casscf/orbital_sigma.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>

namespace casscf {
namespace {

// helper_PFCI.py:12111-12119: dense materialization of the matrix-free
// operator via unit-vector probing, needed because minres_solve (this
// port's faithful scipy.sparse.linalg.minres port, see minres_solver.hpp)
// only accepts a dense matrix -- same materialization approach already
// used elsewhere in this codebase for a matrix-free operator feeding a
// dense-only routine (e.g. validate_against_python.cpp's qn_gltr/qn_bfgs
// cases). Flagged here, same as orbital_sigma3's own doc comment, as
// worth revisiting for performance once correctness is settled: this is
// O(index_map_size) calls to the already-expensive orbital_sigma3, only
// reached on the (rare) gradient-small-Newton fallback when
// linear_equation_solve itself didn't converge.
Matrix materialize_dense_hessian(const std::function<Vector(const Vector&)>& apply, int n) {
    Matrix H(n, n);
    for (int i = 0; i < n; ++i) {
        Vector e_i = Vector::Zero(n);
        e_i(i) = 1.0;
        H.col(i) = apply(e_i);
    }
    return H;
}

StateAverageData make_state_average_data(const CasscfContext& context, const CasscfPhysicalConstants& constants) {
    StateAverageData sad;
    sad.D_tu_avg = context.D_tu_avg;
    sad.D_tuvw_avg = context.D_tuvw_avg;
    sad.Dpe_tu_avg = context.Dpe_tu_avg;
    sad.weight = constants.weight;
    sad.N_p = constants.N_p;
    sad.num_det = constants.num_det;
    sad.omega = constants.omega;
    sad.Enuc = constants.Enuc;
    sad.d_c = constants.d_c;
    sad.d_exp = constants.d_exp;
    return sad;
}

// helper_PFCI.py:11085-11119 (first occurrence) / 12246-12260 (post-accept
// rebuild): zero-pad build_gradient's (nmo, n_occupied) A_tilde into the
// full (nmo, nmo) shape build_hessian_diagonal expects, WITHOUT
// symmetrizing -- deliberately not embed_and_symmetrize_A_tilde
// (intermediates.hpp), which additionally adds the transpose and is meant
// for a different consumer (OrbitalHessianGuessProvider's sym_A_tilde).
Matrix embed_A_tilde_full(const Matrix& A_tilde_occ_cols, const Dimensions& dims) {
    Matrix full = Matrix::Zero(dims.nmo, dims.nmo);
    full.leftCols(dims.n_occupied) = A_tilde_occ_cols;
    return full;
}

// helper_PFCI.py:11113-11120 / 12250-12262: reduced_gradient[i] =
// gradient_tilde[s][r] where (s, r) = index_map[i] (s the larger index, r
// the smaller) -- exactly build_index_map's own enumeration, same
// correspondence build_hessian_diagonal's doc comment already establishes
// between its reduction loop and build_index_map.
Vector extract_reduced_gradient(const Matrix& gradient_tilde, const std::vector<std::pair<int, int>>& index_map) {
    Vector reduced(static_cast<int>(index_map.size()));
    for (size_t i = 0; i < index_map.size(); ++i) {
        const auto [s, r] = index_map[i];
        reduced(static_cast<int>(i)) = gradient_tilde(s, r);
    }
    return reduced;
}

// helper_PFCI.py:12007-12175 (QN branch) / 12312-12325 (non-QN branch,
// structurally identical): unpack the reduced-space step into the three
// off-diagonal rotation-generator blocks build_unitary_matrix expects.
void step_to_rotation_blocks(const Vector& step, const std::vector<std::pair<int, int>>& index_map,
                              const Dimensions& dims, Matrix& Rai, Matrix& Rvi, Matrix& Rva) {
    Rai = Matrix::Zero(dims.n_act_orb, dims.n_in_a);
    Rvi = Matrix::Zero(dims.n_virtual, dims.n_in_a);
    Rva = Matrix::Zero(dims.n_virtual, dims.n_act_orb);
    for (size_t i = 0; i < index_map.size(); ++i) {
        const auto [s, l] = index_map[i];
        if (s >= dims.n_in_a && s < dims.n_occupied && l < dims.n_in_a) {
            Rai(s - dims.n_in_a, l) = step(static_cast<int>(i));
        } else if (s >= dims.n_occupied && l < dims.n_in_a) {
            Rvi(s - dims.n_occupied, l) = step(static_cast<int>(i));
        } else {
            Rva(s - dims.n_occupied, l - dims.n_in_a) = step(static_cast<int>(i));
        }
    }
}

// helper_PFCI.py:11017-11031: zero_energy, evaluated against THIS outer
// iteration's fixed reference point (fi.E_core/active_fock_core/
// active_twoeint, set as a side effect of build_intermediates -- see
// FullBlockIntermediates's doc comment) and context.d_cmo (the outer,
// macroiteration-level dipole integrals -- unchanged across microiterations,
// confirmed by grep: self.d_cmo is never assigned anywhere in
// microiteration_optimization6). Same structural pattern as
// internal_optimization_exact_energy's sum_energy, just fed the
// already-active-restricted fi.active_fock_core/active_twoeint directly.
double compute_zero_energy(const FullBlockIntermediates& fi, const CasscfContext& context,
                            const StateAverageData& sad, const Matrix& eigenvecs, const Dimensions& dims) {
    double energy = fi.E_core;
    energy += fi.active_fock_core.cwiseProduct(sad.D_tu_avg).sum();

    double two_e = 0.0;
    for (int t = 0; t < dims.n_act_orb; ++t)
        for (int u = 0; u < dims.n_act_orb; ++u)
            for (int v = 0; v < dims.n_act_orb; ++v)
                for (int w = 0; w < dims.n_act_orb; ++w)
                    two_e += fi.active_twoeint(t, u, v, w) * sad.D_tuvw_avg(t, u, v, w);
    energy += 0.5 * two_e;

    const Matrix d_cmo_active = context.d_cmo.block(dims.n_in_a, dims.n_in_a, dims.n_act_orb, dims.n_act_orb);
    energy += -std::sqrt(sad.omega / 2.0) * d_cmo_active.cwiseProduct(sad.Dpe_tu_avg).sum();

    energy += calculate_ci_dependent_energy(eigenvecs, context.d_cmo, sad.weight, sad.N_p, sad.num_det, sad.omega,
                                             sad.d_exp, dims.n_in_a);
    energy += sad.Enuc;
    energy += sad.d_c;
    return energy;
}

// The local (active_fock_core, active_twoeint, d_cmo, E_core2) tuple --
// helper_PFCI.py's bare (not self.-prefixed) locals of the same names,
// function-scope-persistent across the whole run() call, refreshed only by
// an accepted inner step (via microiteration_ci_integrals_transform) or the
// convergence-time reference-point fallback below.
struct CiInputStaging {
    Matrix active_fock_core;
    Tensor4 active_twoeint;
    Matrix d_cmo;
    double E_core2 = 0.0;
};

// helper_PFCI.py:12306-12325: build the occupied-sized gkl2/occupied_J/
// occupied_fock_core/occupied_d_cmo the CI solver needs from the staging
// tuple, and commit them into context (see this class's header doc comment
// for why context, not a fresh return channel, is the right home).
void commit_ci_solver_inputs(const CiInputStaging& staging, const Dimensions& dims, CasscfContext& context) {
    const int n_act = dims.n_act_orb;
    const int n_in_a = dims.n_in_a;
    const int n_occ = dims.n_occupied;

    // gkl2(k,l) = active_fock_core(k,l) - 0.5 * sum_j active_twoeint(k,j,j,l)
    Matrix gkl2(n_act, n_act);
    for (int k = 0; k < n_act; ++k) {
        for (int l = 0; l < n_act; ++l) {
            double acc = 0.0;
            for (int j = 0; j < n_act; ++j) acc += staging.active_twoeint(k, j, j, l);
            gkl2(k, l) = staging.active_fock_core(k, l) - 0.5 * acc;
        }
    }
    context.gkl2 = gkl2;

    // occupied_J: (n_occupied)^4, zero everywhere except the active-active
    // block, which holds active_twoeint -- committed into context.occupied_J's
    // [:, :, :n_occupied, :n_occupied] sub-block, matching the sub-block-write
    // convention CasscfContext's own doc comment establishes for this field.
    for (int p = 0; p < n_occ; ++p)
        for (int q = 0; q < n_occ; ++q)
            for (int r = 0; r < n_occ; ++r)
                for (int s = 0; s < n_occ; ++s) context.occupied_J(p, q, r, s) = 0.0;
    for (int t = 0; t < n_act; ++t)
        for (int u = 0; u < n_act; ++u)
            for (int v = 0; v < n_act; ++v)
                for (int w = 0; w < n_act; ++w)
                    context.occupied_J(n_in_a + t, n_in_a + u, n_in_a + v, n_in_a + w) =
                        staging.active_twoeint(t, u, v, w);

    Matrix occ_fock = Matrix::Zero(n_occ, n_occ);
    occ_fock.block(n_in_a, n_in_a, n_act, n_act) = staging.active_fock_core;
    context.occupied_fock_core = occ_fock;

    context.occupied_d_cmo = staging.d_cmo.topLeftCorner(n_occ, n_occ);
    context.E_core2 = staging.E_core2;
}

} // namespace

CasscfMicroiterationOptimizationStep::CasscfMicroiterationOptimizationStep(Dimensions dims,
                                                                             CasscfPhysicalConstants constants,
                                                                             CiStateAverageSolver& ci_solver,
                                                                             int max_microiterations)
    : dims_(dims), constants_(std::move(constants)), ci_solver_(&ci_solver),
      max_microiterations_(max_microiterations) {}

void CasscfMicroiterationOptimizationStep::run(CasscfContext& context, const Matrix& U, Matrix& eigenvecs,
                                                double convergence_threshold) {
    const Dimensions& dims = dims_;
    const auto index_map = build_index_map(dims);

    U2_ = U; // helper_PFCI.py:10910 (self.U2 = copy.deepcopy(U))

    // helper_PFCI.py's function-scope locals, persistent across outer
    // iterations (NOT reset at the top of each pass) -- see
    // CiInputStaging's own doc comment.
    CiInputStaging staging;
    staging.active_fock_core = Matrix::Zero(dims.n_act_orb, dims.n_act_orb);
    staging.active_twoeint = Tensor4(dims.n_act_orb, dims.n_act_orb, dims.n_act_orb, dims.n_act_orb);
    staging.active_twoeint.setZero();
    staging.d_cmo = Matrix::Zero(dims.nmo, dims.nmo);
    staging.E_core2 = 0.0;

    double current_energy = 0.0;
    double old_energy = 0.0;
    int N_orbital_optimization_steps = 1;
    int N_microiterations = max_microiterations_;

    int microiteration = 0;
    while (microiteration < N_microiterations) {
        // helper_PFCI.py:11019: trust_radius reset to 0.5 once per OUTER
        // ("microiteration") pass -- NOT once per inner
        // ("orbital_optimization_step") solve. Threaded forward across
        // inner iterations below via step_control on accept / *0.5 on
        // reject (helper_PFCI.py:12267, 12339) -- a real, previously
        // mistaken assumption in this class corrected here (an earlier
        // version reset to 0.5 at every inner solve and discarded
        // step_control's return value entirely; found and fixed via a
        // direct side-by-side trace comparison against the real Python on
        // real LiH chemistry, not by re-reading the Python alone -- see
        // this project's session notes for how the discrepancy was found).
        double trust_radius = 0.5;

        // Step 1: build_intermediates, once per outer iteration -- sets this
        // iteration's fixed reference point.
        const double off_diagonal_constant =
            calculate_off_diagonal_photon_constant(eigenvecs, constants_.weight, constants_.N_p, constants_.num_det,
                                                    constants_.omega);
        StateAverageData sad = make_state_average_data(context, constants_);
        FullBlockIntermediates fi = build_intermediates(context.H_spatial2, context.d_cmo, context.J, context.K,
                                                          sad.D_tu_avg, sad.D_tuvw_avg, sad.Dpe_tu_avg,
                                                          off_diagonal_constant, sad.omega, dims);
        context.E_core = fi.E_core; // helper_PFCI.py: self.E_core reassigned inside build_intermediates

        // Step 2: zero_energy / current_energy / small-energy-change break.
        const double zero_energy = compute_zero_energy(fi, context, sad, eigenvecs, dims);
        const double new_current_energy = zero_energy + microiteration_exact_energy(U2_, fi.A, fi.G, dims);
        if (std::abs(new_current_energy - old_energy) < std::max(0.01 * convergence_threshold, 1e-10) &&
            microiteration >= 2) {
            break;
        }
        old_energy = new_current_energy;
        current_energy = new_current_energy;

        // Step 3: gradient / hessian diagonal / reduced_gradient / n_negative.
        GradientResult gr = build_gradient(U2_, fi.A, fi.G, dims);
        Matrix A_tilde_full = embed_A_tilde_full(gr.A_tilde, dims);
        HessianDiagonalResult hd = build_hessian_diagonal(U2_, fi.G, A_tilde_full, dims);
        Vector reduced_gradient = extract_reduced_gradient(gr.gradient_tilde, index_map);
        int n_negative = static_cast<int>((hd.reduced_hessian_diagonal.array() < 0.0).count());

        // Step 4: inner orbital-optimization-step loop (non-QN only; see
        // header doc comment, deviation 1).
        int orbital_optimization_step = 0;
        int accepted_count = 0;
        bool small_gradient_convergence = false;

        auto hessian_op = std::make_shared<MatrixFreeHessianOperator>(
            [&](const Vector& v) { return orbital_sigma3(U2_, A_tilde_full, fi.G, v, dims); },
            dims.index_map_size());

        while (orbital_optimization_step < N_orbital_optimization_steps) {
            const double gradient_norm = reduced_gradient.norm();
            if (gradient_norm < 0.1 * convergence_threshold && microiteration > 0) {
                small_gradient_convergence = true;
                break;
            }
            if (gradient_norm < 1e-7) {
                small_gradient_convergence = true;
                break;
            }

            Vector step;
            int hard_case = 0;
            if (gradient_norm > 1e-3) {
                if (n_negative == 0) {
                    GltrTrustRegionSolver solver(hessian_op, hd.reduced_hessian_diagonal);
                    TrustRegionResult result = solver.solve(reduced_gradient, trust_radius);
                    step = result.step;
                    hard_case = 0;
                } else {
                    Matrix sym_A_tilde = embed_and_symmetrize_A_tilde(gr.A_tilde, dims);
                    auto guess_provider = std::make_shared<OrbitalHessianGuessProvider>(
                        U2_, sym_A_tilde, reduced_gradient, fi.G, index_map, dims.n_occupied);
                    DavidsonDrivenLstrsSolver solver(hessian_op, guess_provider, hd.reduced_hessian_diagonal);
                    TrustRegionResult result = solver.solve(reduced_gradient, trust_radius);
                    step = result.step;
                    hard_case = (result.reason == TerminationReason::SuccessInteriorSolution) ? 2 : 0;
                }
            } else {
                // helper_PFCI.py:12103-12137: gradient-small Newton fallback
                // -- linear_equation_solve (LinearRMSolver), falling back to
                // real MINRES on non-convergence. See linear_equation_solve.hpp's
                // own doc comment for the one remaining, inherent (not
                // fixable) non-reproducibility (the random initial-probe
                // draw, always exercised at this call site).
                LinearEquationSolveResult solve_result = linear_equation_solve(
                    U2_, A_tilde_full, fi.G, reduced_gradient, hd.reduced_hessian_diagonal,
                    /*max_iter=*/20, /*conv_thresh=*/1e-6, dims);
                if (solve_result.converged) {
                    step = solve_result.solution;
                } else {
                    Matrix dense_hessian = materialize_dense_hessian(
                        [&](const Vector& v) { return orbital_sigma3(U2_, A_tilde_full, fi.G, v, dims); },
                        dims.index_map_size());
                    MinresResult minres_result = minres_solve(dense_hessian, -reduced_gradient, /*rtol=*/1e-6);
                    step = minres_result.x;
                }
                hard_case = 2;
            }

            Matrix Rai, Rvi, Rva;
            step_to_rotation_blocks(step, index_map, dims, Rai, Rvi, Rva);
            Matrix U_delta = build_unitary_matrix(Rai, Rvi, Rva, dims);
            Matrix U3 = U2_ * U_delta;

            const double second_order_energy_change = microiteration_exact_energy(U3, fi.A, fi.G, dims);
            const double energy_change = zero_energy + second_order_energy_change - current_energy;
            const double predicted_energy2 =
                microiteration_predicted_energy2(U2_, reduced_gradient, A_tilde_full, fi.G, step, dims);

            // helper_PFCI.py:12169-12172: unconditional on the very first
            // inner iteration overall, whether or not this trial is accepted.
            if (microiteration == 0 && orbital_optimization_step == 0) {
                convergence_threshold = std::min(0.01 * gradient_norm, gradient_norm * gradient_norm);
            }

            if (energy_change < 0.0 || hard_case == 2) {
                // Accept.
                U2_ = U3;

                if (microiteration == 0 && orbital_optimization_step == 0) {
                    const double step_norm = step.norm();
                    if (step_norm > 0.1) {
                        N_microiterations = 5;
                        N_orbital_optimization_steps = 4;
                    } else if (step_norm > 0.05) {
                        N_microiterations = 7;
                        N_orbital_optimization_steps = 3;
                    }
                }
                ++orbital_optimization_step;

                const double ratio = energy_change / predicted_energy2;
                // helper_PFCI.py:12267: trust_radius carries forward across
                // inner ("orbital_optimization_step") iterations within one
                // outer pass, updated by step_control on every accept (and
                // halved on every reject, below) -- NOT reset to 0.5 fresh
                // each inner solve. See this loop's own trust_radius
                // declaration comment for how this was found and corrected.
                trust_radius = step_control(ratio, trust_radius);

                // Rebuild gradient/hessian/reduced_gradient on the SAME
                // fi.A/fi.G (not a fresh build_intermediates call).
                gr = build_gradient(U2_, fi.A, fi.G, dims);
                A_tilde_full = embed_A_tilde_full(gr.A_tilde, dims);
                hd = build_hessian_diagonal(U2_, fi.G, A_tilde_full, dims);
                reduced_gradient = extract_reduced_gradient(gr.gradient_tilde, index_map);

                MicroiterationCiIntegralsResult ci_int = microiteration_ci_integrals_transform(
                    U2_, fi.E_core, fi.fock_core, fi.L, context.J, context.K, fi.active_twoeint, context.d_cmo, dims);
                staging.active_fock_core = ci_int.active_fock_core;
                staging.active_twoeint = ci_int.active_twoeint;
                staging.d_cmo = ci_int.d_cmo;
                staging.E_core2 = ci_int.E_core2;

                ++accepted_count;
                current_energy = zero_energy + second_order_energy_change;
            } else {
                // Reject: helper_PFCI.py:12339 -- trust_radius is halved
                // and carries forward into the next inner iteration (same
                // trust_radius variable the accept branch updates above);
                // nothing else changes, loop continues with the same
                // reduced_gradient.
                trust_radius = 0.5 * trust_radius;
            }
        }

        // Step 5: after the inner loop, unconditionally once per outer pass.
        if (small_gradient_convergence && accepted_count == 0) {
            // helper_PFCI.py:12300-12304.
            staging.active_fock_core = fi.active_fock_core;
            staging.active_twoeint = fi.active_twoeint;
            staging.d_cmo = context.d_cmo;
            staging.E_core2 = fi.E_core;
        }

        commit_ci_solver_inputs(staging, dims, context);

        CiStateAverageResult ci_result = ci_solver_->solve(eigenvecs);
        eigenvecs = ci_result.eigenvectors;
        context.D_tu_avg = ci_result.D_tu_avg;
        context.D_tuvw_avg = ci_result.D_tuvw_avg;
        context.Dpe_tu_avg = ci_result.Dpe_tu_avg;

        ++microiteration;
    }
}

} // namespace casscf
