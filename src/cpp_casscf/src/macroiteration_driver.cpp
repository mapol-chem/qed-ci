#include "casscf/macroiteration_driver.hpp"
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

MacroiterationResult MacroiterationDriver::run(Matrix eigenvecs0, double avg_energy0,
                                                Matrix H_spatial2, Matrix d_cmo) {
    const Dimensions& dims = config_.dims;

    Matrix eigenvecs = std::move(eigenvecs0);
    Vector eigenvalues; // only populated once the CI solver has run at least once -- see below

    // helper_PFCI.py:2390. NOTE: this only tracks the two rotations the
    // driver itself applies below (the restart-branch internal correction
    // and the main microiteration step, helper_PFCI.py:2895/2937) -- it does
    // NOT include whatever internal_optimization3 does to its own U_total
    // internally (helper_PFCI.py:7805). See InternalOptimizationStep's doc
    // comment for why that's a documented gap of this interface shape, not
    // an oversight here.
    Matrix U_total = Matrix::Identity(dims.nmo, dims.nmo);

    // helper_PFCI.py:2391-2393. `convergence` is intentionally never reset
    // once set -- matches the Python exactly, including the (mild) wart that
    // this lets a single coincidental near-zero |new-old| gap latch
    // convergence permanently even if a later, real comparison disagrees.
    double old_avg_energy = 0.0;
    double new_avg_energy = avg_energy0;
    bool convergence = false;

    int macroiteration = 0;
    for (; macroiteration < config_.max_macroiterations; ++macroiteration) {
        if (macroiteration > 0) {
            // helper_PFCI.py:2395-2521: CI diagonalization + state-averaged
            // RDM build, behind CiStateAverageSolver.
            CiStateAverageResult ci_result = ci_solver_->solve(eigenvecs);
            eigenvalues = ci_result.eigenvalues;
            eigenvecs = ci_result.eigenvectors;
            old_avg_energy = new_avg_energy;
            new_avg_energy = ci_result.avg_energy;
        }

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
            internal_step_->run(new_avg_energy, eigenvecs);
        }

        // helper_PFCI.py:2841-2846.
        const double convergence_threshold = macroiteration == 0
            ? config_.first_iteration_convergence_threshold
            : config_.later_iteration_convergence_threshold;

        // helper_PFCI.py:2847-2850.
        const Matrix U0 = Matrix::Identity(dims.nmo, dims.nmo);
        microiteration_step_->run(U0, eigenvecs, convergence_threshold);
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
            Matrix Rvi = Matrix::Zero(dims.n_virtual, dims.n_in_a);
            Matrix Rva = Matrix::Zero(dims.n_virtual, dims.n_act_orb);
            Matrix U_delta = build_unitary_matrix(Rai, Rvi, Rva, dims);

            integral_transformer_->transform_internal_rotation(U_delta);
            U_total = U_total * U_delta;

            Rai.setZero();
            Rvi = R.block(dims.n_occupied, 0, dims.n_virtual, dims.n_in_a);
            Rva = R.block(dims.n_occupied, dims.n_in_a, dims.n_virtual, dims.n_act_orb);
            U_delta = build_unitary_matrix(Rai, Rvi, Rva, dims);

            microiteration_step_->run(U_delta, eigenvecs, convergence_threshold);
            U2 = microiteration_step_->last_U2();
        }

        // helper_PFCI.py:2925-2937: rotate H_spatial2/d_cmo by the final U2
        // and fold U2 into U_total. (self.d_cmo0, the Python's pre-rotation
        // copy of d_cmo, is never read again in the ported range -- dropped
        // here as dead state.)
        H_spatial2 = U2.transpose() * H_spatial2 * U2;
        d_cmo = U2.transpose() * d_cmo * U2;
        U_total = U_total * U2;

        // helper_PFCI.py:2944-2973: rebuild J/K (density-fitted or not) on
        // the accumulated U_total. fock_core/E_core bookkeeping that follows
        // in the Python (helper_PFCI.py:2990-3057, consumed by next
        // iteration's CI solve) is intentionally not modeled here -- it
        // depends on the J/K four-index tensors, which this driver never
        // holds; it belongs behind this same call, as an implementation
        // detail of IntegralTransformer / CiStateAverageSolver's shared
        // state rather than something threaded through this signature.
        integral_transformer_->transform_macroiteration(U_total);
    }

    MacroiterationResult result;
    result.converged = convergence && macroiteration > 0;
    result.macroiterations_run = macroiteration;
    result.avg_energy = new_avg_energy;
    result.eigenvalues = eigenvalues;
    result.eigenvectors = eigenvecs;
    result.U_total = U_total;
    return result;
}

} // namespace casscf
