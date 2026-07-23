#include "casscf/internal_optimization_step.hpp"

#include "casscf/hessian_operator.hpp"
#include "casscf/intermediates.hpp"
#include "casscf/internal_optimization.hpp"
#include "casscf/lstrs_solver.hpp"
#include "casscf/orbital_rotation.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

namespace casscf {
namespace {

// Scientific-notation formatting for log records (see logging.hpp).
std::string fmte(double x) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.3e", x);
    return buf;
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

// helper_PFCI.py:7033-7035
// (gradient_tilde_ai[:, :] = gradient_tilde1[n_in_a:n_occupied, :n_in_a]).
Vector extract_gradient_ai(const Matrix& gradient_tilde, const Dimensions& dims) {
    Vector g(dims.n_act_orb * dims.n_in_a);
    for (int a = 0; a < dims.n_act_orb; ++a)
        for (int b = 0; b < dims.n_in_a; ++b) g(a * dims.n_in_a + b) = gradient_tilde(dims.n_in_a + a, b);
    return g;
}

// helper_PFCI.py:7036-7043: hessian_tilde1.transpose(2, 0, 3, 1), sliced
// [n_in_a:n_occupied, :n_in_a, n_in_a:n_occupied, :n_in_a], reshaped to
// (n_act_orb*n_in_a, n_act_orb*n_in_a). Worked out by hand from numpy's
// transpose(axes) semantics (result[i0,i1,i2,i3] = original[j] where
// j[axes[k]] = i[k] for axes=(2,0,3,1), i.e. j = (i1, i3, i0, i2)):
// result[a,b,c,d] = hessian_tilde1[b, d, n_in_a+a, n_in_a+c].
Matrix extract_hessian_ai(const Tensor4& hessian_tilde, const Dimensions& dims) {
    Matrix h(dims.n_act_orb * dims.n_in_a, dims.n_act_orb * dims.n_in_a);
    for (int a = 0; a < dims.n_act_orb; ++a)
        for (int b = 0; b < dims.n_in_a; ++b)
            for (int c = 0; c < dims.n_act_orb; ++c)
                for (int d = 0; d < dims.n_in_a; ++d)
                    h(a * dims.n_in_a + b, c * dims.n_in_a + d) =
                        hessian_tilde(b, d, dims.n_in_a + a, dims.n_in_a + c);
    return h;
}

// helper_PFCI.py:7250 (Rai = step.reshape(n_act_orb, n_in_a)).
Matrix step_to_Rai(const Vector& step, const Dimensions& dims) {
    Matrix Rai(dims.n_act_orb, dims.n_in_a);
    for (int a = 0; a < dims.n_act_orb; ++a)
        for (int b = 0; b < dims.n_in_a; ++b) Rai(a, b) = step(a * dims.n_in_a + b);
    return Rai;
}

} // namespace

CasscfInternalOptimizationStep::CasscfInternalOptimizationStep(Dimensions dims, CasscfPhysicalConstants constants,
                                                                 CiStateAverageSolver& ci_solver,
                                                                 IntegralTransformer& integral_transformer,
                                                                 int max_microiterations, int lstrs_max_iter)
    : dims_(dims),
      constants_(std::move(constants)),
      ci_solver_(&ci_solver),
      integral_transformer_(&integral_transformer),
      max_microiterations_(max_microiterations),
      lstrs_max_iter_(lstrs_max_iter) {}

// Precondition (caller's responsibility, same as MacroiterationDriver::run's
// on context.H_spatial2/d_cmo): context.occupied_fock_core/occupied_d_cmo/
// occupied_h1/occupied_J/occupied_K/D_tu_avg/D_tuvw_avg/Dpe_tu_avg and
// context.H_spatial2/d_cmo/J/K must already be populated for this
// macroiteration before run() is called.
void CasscfInternalOptimizationStep::run(CasscfContext& context, double /*E0*/, Matrix& eigenvecs) {
    const Dimensions& dims = dims_;
    StateAverageData sad = make_state_average_data(context, constants_);

    // helper_PFCI.py:6884-6919 ("test energy" at the top of
    // internal_optimization3): the initial sum_energy, used only to seed
    // the `current_energy` baseline for the ratio test below. Reuses
    // internal_optimization_exact_energy (E0=0, so its returned
    // energy_change *is* this sum_energy; hard_case=-1 so the hard_case==2
    // branch never fires, matching this block's own un-committing formula)
    // instead of re-deriving the identical formula a third time -- see this
    // class's header doc comment for why that reuse is safe (the
    // accept-branch side effects it would otherwise trigger are computed
    // but simply never read here, since only .energy_change is used).
    double off_diagonal_constant =
        calculate_off_diagonal_photon_constant(eigenvecs, sad.weight, sad.N_p, sad.num_det, sad.omega);
    double current_energy = internal_optimization_exact_energy(0.0, eigenvecs, context.H_spatial2, context.d_cmo,
                                                                 context.J, context.K, /*hard_case=*/-1, sad, dims)
                                 .energy_change;

    // helper_PFCI.py:6970, 6975.
    Matrix U1 = Matrix::Identity(dims.nmo, dims.nmo);
    double trust_radius = 0.5;

    // helper_PFCI.py:6984-7043. BLAS-backed production path; the legacy
    // explicit-loop build_intermediates_internal stays in intermediates.cpp as
    // the line-for-line Python correspondence, the TAMM-retarget reference,
    // and this path's oracle (test_intermediates_internal_fast.cpp ties the
    // two elementwise). Same arrangement as build_intermediates_fast.
    SmallBlockIntermediates si = build_intermediates_internal_fast(
        context.occupied_fock_core, context.occupied_d_cmo, context.occupied_J, context.occupied_K, sad.D_tu_avg,
        sad.D_tuvw_avg, sad.Dpe_tu_avg, off_diagonal_constant, sad.omega, dims);
    GradientAndHessianResult gh = build_gradient_and_hessian(si.A, si.G, dims);
    Vector gradient_ai = extract_gradient_ai(gh.gradient_tilde, dims);
    Matrix hessian_ai = extract_hessian_ai(gh.hessian_tilde, dims);

    // See this class's header doc comment, deviation (2).
    bool ci_converged = false;

    for (int microiteration = 0;; ++microiteration) {
        // helper_PFCI.py:6996-7548 (the outer LSTRS bisection, minus the
        // cross-microiteration warm-start shortcut -- see this class's
        // header doc comment, deviation (1)).
        LstrsSolver solver(std::make_shared<DenseHessianOperator>(hessian_ai), lstrs_max_iter_);
        TrustRegionResult lstrs_result = solver.solve(gradient_ai, trust_radius);
        const Vector& step = lstrs_result.step;
        const bool hard_case_is_interior = (lstrs_result.reason == TerminationReason::SuccessInteriorSolution);

        // helper_PFCI.py:7250-7255.
        Matrix Rai = step_to_Rai(step, dims);
        Matrix Rvi = Matrix::Zero(dims.n_virtual, dims.n_in_a);
        Matrix Rva = Matrix::Zero(dims.n_virtual, dims.n_act_orb);
        Matrix U_delta = build_unitary_matrix(Rai, Rvi, Rva, dims);

        // helper_PFCI.py:7256-7278.
        InternalTransformationResult it =
            internal_transformation(U_delta, context.occupied_h1, context.occupied_d_cmo, context.occupied_J, dims);
        InternalOptimizationEnergyResult er = internal_optimization_exact_energy(
            current_energy, eigenvecs, it.h1, it.d_cmo1, it.J, it.K, hard_case_is_interior ? 2 : 0, sad, dims);
        const double predicted_energy = internal_optimization_predicted_energy(gradient_ai, hessian_ai, step);

        const double trust_radius_used = trust_radius; // before step_control/halving below
        if (er.accepted) {
            // helper_PFCI.py:7794 (self.U1 = einsum("pq,qs->ps", self.U1, self.U_delta)).
            U1 = U1 * U_delta;

            // helper_PFCI.py:6816-6828: commit into the occupied_*
            // sub-block context owns -- only the leading n_occupied of
            // occupied_J/K's trailing (nmo-sized) axes, matching
            // CasscfContext's doc comment on occupied_J/K's shape.
            for (int p = 0; p < dims.n_occupied; ++p)
                for (int q = 0; q < dims.n_occupied; ++q)
                    for (int r = 0; r < dims.n_occupied; ++r)
                        for (int s = 0; s < dims.n_occupied; ++s) {
                            context.occupied_J(p, q, r, s) = er.occupied_J(p, q, r, s);
                            context.occupied_K(p, q, r, s) = er.occupied_K(p, q, r, s);
                        }
            context.occupied_h1 = er.occupied_h1;
            context.occupied_d_cmo = er.occupied_d_cmo;
            context.occupied_fock_core = er.occupied_fock_core;
            context.E_core = er.E_core;
            context.gkl2 = er.gkl2;
            current_energy += er.energy_change;

            // helper_PFCI.py:7740: internal_optimization3's own c_get_roots
            // call feeds self.E_core (just committed above, reflecting the
            // internal rotation accumulated so far) into constdouble[5] --
            // stage it into context.E_core2, the shared slot
            // CiStateAverageSolver::solve(use_staged_inputs=true) reads
            // uniformly for both this class's and
            // MicroiterationOptimizationStep's own staged-mode calls (see
            // that field's own doc comment in casscf_context.hpp).
            context.E_core2 = context.E_core;

            // helper_PFCI.py:7699-7728: re-diagonalize the CI problem and
            // refresh the state-averaged RDMs.
            CiStateAverageResult ci_result = ci_solver_->solve(eigenvecs, /*use_staged_inputs=*/true);
            eigenvecs = ci_result.eigenvectors;
            context.D_tu_avg = ci_result.D_tu_avg;
            context.D_tuvw_avg = ci_result.D_tuvw_avg;
            context.Dpe_tu_avg = ci_result.Dpe_tu_avg;
            sad.D_tu_avg = ci_result.D_tu_avg;
            sad.D_tuvw_avg = ci_result.D_tuvw_avg;
            sad.Dpe_tu_avg = ci_result.Dpe_tu_avg;
            ci_converged = ci_result.ci_diagonalization_converged;

            // helper_PFCI.py:7756-7758.
            if (predicted_energy != 0.0) {
                trust_radius = step_control(er.energy_change / predicted_energy, trust_radius);
            }

            // helper_PFCI.py:7846-7862: rebuild intermediates on the newly
            // committed state for the next microiteration.
            off_diagonal_constant =
                calculate_off_diagonal_photon_constant(eigenvecs, sad.weight, sad.N_p, sad.num_det, sad.omega);
            si = build_intermediates_internal_fast(context.occupied_fock_core, context.occupied_d_cmo,
                                                    context.occupied_J, context.occupied_K, sad.D_tu_avg,
                                                    sad.D_tuvw_avg, sad.Dpe_tu_avg, off_diagonal_constant, sad.omega,
                                                    dims);
            gh = build_gradient_and_hessian(si.A, si.G, dims);
            gradient_ai = extract_gradient_ai(gh.gradient_tilde, dims);
            hessian_ai = extract_hessian_ai(gh.hessian_tilde, dims);
        } else {
            // helper_PFCI.py:7815-7819.
            trust_radius = 0.5 * trust_radius;
        }

        // Per-internal-microiteration (inactive-active rotation) trace.
        CASSCF_LOG(context.log, PrintLevel::Debug,
                   "[internal] macro=" << context.macroiteration << " micro=" << microiteration
                   << " solver=LSTRS trust=" << fmte(trust_radius_used)
                   << " hc=" << (hard_case_is_interior ? 2 : 0) << " snorm=" << fmte(step.norm())
                   << " dE=" << fmte(er.energy_change) << " pred=" << fmte(predicted_energy)
                   << " grad=" << fmte(gradient_ai.norm()) << " accept=" << (er.accepted ? 1 : 0));

        // helper_PFCI.py:7820-7824.
        if ((gradient_ai.norm() < 1e-4 && ci_converged) || microiteration == max_microiterations_) {
            // helper_PFCI.py:7788-7806.
            integral_transformer_->transform_internal_rotation(U1);
            context.U_total = context.U_total * U1;
            break;
        }
    }
}

} // namespace casscf
