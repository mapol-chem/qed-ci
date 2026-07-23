#include "casscf/macroiteration_driver.hpp"

#include <cstdio>
#include <string>
#include "casscf/orbital_rotation.hpp"

#include <cmath>
#include <utility>

namespace casscf {

MacroiterationDriver::MacroiterationDriver(MacroiterationDriverConfig config,
                                            CiStateAverageSolver& ci_solver,
                                            InternalOptimizationStep& internal_step,
                                            MicroiterationOptimizationStep& microiteration_step,
                                            IntegralTransformer& integral_transformer)
    : config_(std::move(config)),
      ci_solver_(&ci_solver),
      internal_step_(&internal_step),
      microiteration_step_(&microiteration_step),
      integral_transformer_(&integral_transformer) {}

namespace {
// Fixed-width numeric formatting so the records stay column-stable and
// greppable (see logging.hpp): energies at 12 decimals, small quantities in
// scientific notation.
std::string fmt12(double x) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.12f", x);
    return buf;
}
std::string fmte(double x) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.3e", x);
    return buf;
}
} // namespace

MacroiterationResult MacroiterationDriver::run(Matrix eigenvecs0, double avg_energy0, CasscfContext& context) {
    const Dimensions& dims = config_.dims;

    Matrix eigenvecs = std::move(eigenvecs0);
    Vector eigenvalues; // only populated once the CI solver has run at least once -- see below

    // helper_PFCI.py:2390. context.H_spatial2/context.d_cmo are left as the
    // caller set them; context.U_total is reset here, matching self.U_total
    // = eye(nmo) at the top of the macroiteration loop. Using context (not
    // a local) means internal_step's own rotation of these on acceptance
    // (helper_PFCI.py:7787-7805) is visible to -- and composes with -- the
    // rotation applied below (helper_PFCI.py:2925-2937), rather than being a
    // documented gap. See CasscfContext's doc comment.
    context.U_total = Matrix::Identity(dims.nmo, dims.nmo);

    // helper_PFCI.py:2391-2393. `convergence` is intentionally never reset
    // once set -- matches the Python exactly, including the (mild) wart that
    // this lets a single coincidental near-zero |new-old| gap latch
    // convergence permanently even if a later, real comparison disagrees.
    double old_avg_energy = 0.0;
    double new_avg_energy = avg_energy0;
    bool convergence = false;

    int macroiteration = 0;
    for (; macroiteration < config_.max_macroiterations; ++macroiteration) {
        // Once-per-macroiteration banner: the macroiteration number is written
        // here, in caps, so the [ci]/[macro]/[internal]/[micro]/[orb] records
        // below don't repeat macro= on every line.
        CASSCF_LOG(context.log, PrintLevel::Normal,
                   "\n========== MACROITERATION " << macroiteration << " ==========");

        if (macroiteration > 0) {
            // helper_PFCI.py:2395-2521: CI diagonalization + state-averaged
            // RDM build, behind CiStateAverageSolver.
            CiStateAverageResult ci_result = ci_solver_->solve(eigenvecs);
            eigenvalues = ci_result.eigenvalues;
            eigenvecs = ci_result.eigenvectors;
            old_avg_energy = new_avg_energy;
            new_avg_energy = ci_result.avg_energy;

            // helper_PFCI.py:7992+ (build_state_average_rdms): writes
            // directly into the persistent self.D_tu_avg/D_tuvw_avg/
            // Dpe_tu_avg instance attributes right after the CI solve --
            // context is this driver's equivalent of that persistent
            // instance state, read later in this same macroiteration by
            // InternalOptimizationStep. See CiStateAverageResult's and
            // CasscfContext's doc comments.
            context.D_tu_avg = ci_result.D_tu_avg;
            context.D_tuvw_avg = ci_result.D_tuvw_avg;
            context.Dpe_tu_avg = ci_result.Dpe_tu_avg;

            // [ci] -- the macroiteration-level ("runaway") CI solve: run to
            // convergence against the freshly-rotated integrals from the
            // previous macroiteration.
            CASSCF_LOG(context.log, PrintLevel::Normal,
                       "[ci]    phase=macro E=" << fmt12(ci_result.avg_energy)
                       << " res=" << fmte(ci_result.residual_norm)
                       << " roots=" << ci_result.eigenvalues.size()
                       << " conv=" << (ci_result.ci_diagonalization_converged ? 1 : 0));

            // [ci.root] -- per-root detail is Debug and above; at Normal the
            // individual roots are not listed (only spin exceptions are).
            for (int r = 0; r < ci_result.eigenvalues.size(); ++r) {
                CASSCF_LOG(context.log, PrintLevel::Debug,
                           "  [ci.root] root=" << r << " E=" << fmt12(ci_result.eigenvalues(r)));
            }
        }

        // helper_PFCI.py:2597-2600's print call site, same values, same
        // position relative to the convergence check below.
        if (config_.on_macroiteration_end) {
            config_.on_macroiteration_end(macroiteration, old_avg_energy, new_avg_energy);
        }

        // [macro] -- the top-level per-macroiteration energy summary (the
        // iteration number is in the banner above).
        CASSCF_LOG(context.log, PrintLevel::Normal,
                   "[macro] E=" << fmt12(new_avg_energy)
                   << " dE=" << fmte(new_avg_energy - old_avg_energy));

        if (std::abs(new_avg_energy - old_avg_energy) < config_.energy_convergence) {
            convergence = true;
        }

        // helper_PFCI.py:2802: the break sits inside
        // `if macroiteration > 0 and convergence == 1`, before the internal
        // optimization call. The convergence-time state-analysis / dipole /
        // natural-orbital reporting block (helper_PFCI.py:2538-2801) between
        // the convergence check and the break is deliberately not ported
        // here -- see the class doc comment.
        if (macroiteration > 0 && convergence) {
            break;
        }

        // helper_PFCI.py:2804-2809.
        if (macroiteration > 0 && dims.n_in_a > 0) {
            internal_step_->run(context, new_avg_energy, eigenvecs);
        }

        // helper_PFCI.py:2841-2846.
        const double convergence_threshold = macroiteration == 0
            ? config_.first_iteration_convergence_threshold
            : config_.later_iteration_convergence_threshold;

        // helper_PFCI.py:2847-2850.
        const Matrix U0 = Matrix::Identity(dims.nmo, dims.nmo);
        microiteration_step_->run(context, U0, eigenvecs, convergence_threshold);
        Matrix U2 = microiteration_step_->last_U2();

        // helper_PFCI.py:2857-2860: check whether the microiteration step
        // introduced a non-negligible active-inactive ("internal") rotation
        // that needs to be corrected via a dedicated restart before it's
        // folded into H_spatial2/d_cmo below.
        Matrix R = 0.5 * (U2 - U2.transpose());
        Matrix Rai = R.block(dims.n_in_a, 0, dims.n_act_orb, dims.n_in_a);

        if (Rai.norm() > config_.internal_rotation_restart_threshold) {
            // helper_PFCI.py:2864-2896: "RESTART MICROITERATION TO CORRECT
            // INTERNAL ROTATION" -- apply the internal-rotation-only
            // correction to the integrals, fold it into U_total, then rerun
            // the microiteration step starting from the complementary
            // (virtual-inactive / virtual-active) rotation instead of
            // identity.
            CASSCF_LOG(context.log, PrintLevel::Debug,
                       "[restart] internal-rotation norm=" << fmte(Rai.norm())
                       << " > " << fmte(config_.internal_rotation_restart_threshold)
                       << " -- absorbing internal rotation, rerunning microiteration");

            Matrix Rvi = Matrix::Zero(dims.n_virtual, dims.n_in_a);
            Matrix Rva = Matrix::Zero(dims.n_virtual, dims.n_act_orb);
            Matrix U_delta = build_unitary_matrix(Rai, Rvi, Rva, dims);

            integral_transformer_->transform_internal_rotation(U_delta);
            context.U_total = context.U_total * U_delta;

            Rai.setZero();
            Rvi = R.block(dims.n_occupied, 0, dims.n_virtual, dims.n_in_a);
            Rva = R.block(dims.n_occupied, dims.n_in_a, dims.n_virtual, dims.n_act_orb);
            U_delta = build_unitary_matrix(Rai, Rvi, Rva, dims);

            microiteration_step_->run(context, U_delta, eigenvecs, convergence_threshold);
            U2 = microiteration_step_->last_U2();
        }

        // helper_PFCI.py:2925-2937: rotate H_spatial2/d_cmo by the final U2
        // and fold U2 into U_total. (self.d_cmo0, the Python's pre-rotation
        // copy of d_cmo, is never read again in the ported range -- dropped
        // here as dead state.)
        context.H_spatial2 = U2.transpose() * context.H_spatial2 * U2;
        context.d_cmo = U2.transpose() * context.d_cmo * U2;
        context.U_total = context.U_total * U2;

        // helper_PFCI.py:2944-2973: rebuild J/K (density-fitted or not) on
        // the accumulated U_total. fock_core/E_core bookkeeping that follows
        // in the Python (helper_PFCI.py:2990-3057, consumed by next
        // iteration's CI solve) is intentionally not modeled here -- it
        // depends on the J/K four-index tensors, which this driver never
        // holds directly; it belongs behind this same call, as an
        // implementation detail of IntegralTransformer / CiStateAverageSolver
        // operating on context.J/context.K rather than something threaded
        // through this signature.
        integral_transformer_->transform_macroiteration(context.U_total);
    }

    MacroiterationResult result;
    result.converged = convergence && macroiteration > 0;
    result.macroiterations_run = macroiteration;
    result.avg_energy = new_avg_energy;
    result.eigenvalues = eigenvalues;
    result.eigenvectors = eigenvecs;
    return result;
}

} // namespace casscf
