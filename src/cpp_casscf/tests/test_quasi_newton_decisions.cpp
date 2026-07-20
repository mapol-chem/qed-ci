// Tests for the pure QN decision-logic functions that had no dedicated
// coverage of their own before this: should_activate_qn (quasi_newton_policy.hpp,
// helper_PFCI.py:10417-10423/11988-11994) and should_reset_bfgs_reference /
// should_reset_bfgs_reference_point (solver_selector.hpp, helper_PFCI.py:11118,
// 11227). All three are small pure functions over plain structs -- exact
// truth-table checks, no chemistry/hand-solvable-system needed.
#include "casscf/quasi_newton_policy.hpp"
#include "casscf/solver_selector.hpp"

#include <cstdio>

using namespace casscf;

namespace {

int failures = 0;

void expect(bool actual, bool expected, const char* label) {
    if (actual != expected) {
        std::printf("FAIL: %s -- expected %s, got %s\n", label, expected ? "true" : "false",
                    actual ? "true" : "false");
        ++failures;
    } else {
        std::printf("PASS: %s (%s)\n", label, actual ? "true" : "false");
    }
}

} // namespace

int main() {
    // --- should_activate_qn: helper_PFCI.py:10417-10423 --
    // (energy_change < 0.0 or hard_case == 2) and step_norm < threshold,
    // gated by policy.enabled. ---
    {
        QuasiNewtonPolicy policy; // enabled = true, threshold = 0.05
        expect(should_activate_qn(policy, /*energy_change=*/-1.0, /*hard_case=*/0, /*step_norm=*/0.01), true,
               "activate: energy_change<0, small step -> true");
        expect(should_activate_qn(policy, /*energy_change=*/1.0, /*hard_case=*/2, /*step_norm=*/0.01), true,
               "activate: hard_case==2 (energy_change>0), small step -> true (either trigger clause suffices)");
        expect(should_activate_qn(policy, /*energy_change=*/1.0, /*hard_case=*/0, /*step_norm=*/0.01), false,
               "activate: energy_change>0 and hard_case!=2 -> false regardless of step_norm");
        expect(should_activate_qn(policy, /*energy_change=*/-1.0, /*hard_case=*/0, /*step_norm=*/0.05), false,
               "activate: step_norm exactly at threshold -> false (strict <)");
        expect(should_activate_qn(policy, /*energy_change=*/-1.0, /*hard_case=*/0, /*step_norm=*/0.0499), true,
               "activate: step_norm just under threshold -> true");
        expect(should_activate_qn(policy, /*energy_change=*/-1.0, /*hard_case=*/0, /*step_norm=*/0.1), false,
               "activate: step_norm above threshold -> false");

        QuasiNewtonPolicy disabled;
        disabled.enabled = false;
        expect(should_activate_qn(disabled, /*energy_change=*/-1.0, /*hard_case=*/0, /*step_norm=*/0.001), false,
               "activate: policy.enabled == false -> always false regardless of everything else");

        QuasiNewtonPolicy custom_threshold;
        custom_threshold.activation_step_norm_threshold = 0.2;
        expect(should_activate_qn(custom_threshold, /*energy_change=*/-1.0, /*hard_case=*/0, /*step_norm=*/0.15),
               true, "activate: custom threshold respected (0.15 < 0.2)");
    }

    // --- should_reset_bfgs_reference: 4-clause dispatch condition,
    //     helper_PFCI.py:11118 -- (density_norm_change > 0.025 and
    //     qn_optimization) or predicted_energy > 0 or qn_count == 1 or
    //     consecutive_skips >= 3. Each disjunct checked independently, with
    //     the other three held false, to confirm none of them are silently
    //     ANDed together or missing. ---
    {
        BfgsReferenceState all_false{/*density_norm_change=*/0.0, /*predicted_energy=*/-1.0, /*qn_count=*/0,
                                      /*consecutive_skips=*/0};
        expect(should_reset_bfgs_reference(all_false, /*qn_optimization=*/true), false,
               "reset_dispatch: all four clauses false -> false");

        BfgsReferenceState density{/*density_norm_change=*/0.03, /*predicted_energy=*/-1.0, /*qn_count=*/0,
                                    /*consecutive_skips=*/0};
        expect(should_reset_bfgs_reference(density, /*qn_optimization=*/true), true,
               "reset_dispatch: density_norm_change>0.025 and qn_optimization=true -> true");
        expect(should_reset_bfgs_reference(density, /*qn_optimization=*/false), false,
               "reset_dispatch: density_norm_change>0.025 but qn_optimization=false -> false "
               "('and' binds tighter than 'or')");

        BfgsReferenceState predicted{/*density_norm_change=*/0.0, /*predicted_energy=*/0.5, /*qn_count=*/0,
                                      /*consecutive_skips=*/0};
        expect(should_reset_bfgs_reference(predicted, /*qn_optimization=*/false), true,
               "reset_dispatch: predicted_energy>0 -> true regardless of qn_optimization");

        BfgsReferenceState qn_count_one{/*density_norm_change=*/0.0, /*predicted_energy=*/-1.0, /*qn_count=*/1,
                                         /*consecutive_skips=*/0};
        expect(should_reset_bfgs_reference(qn_count_one, /*qn_optimization=*/false), true,
               "reset_dispatch: qn_count==1 -> true regardless of qn_optimization");
        BfgsReferenceState qn_count_two{/*density_norm_change=*/0.0, /*predicted_energy=*/-1.0, /*qn_count=*/2,
                                         /*consecutive_skips=*/0};
        expect(should_reset_bfgs_reference(qn_count_two, /*qn_optimization=*/true), false,
               "reset_dispatch: qn_count==2 (not 1) -> false (other clauses held false)");

        BfgsReferenceState skips{/*density_norm_change=*/0.0, /*predicted_energy=*/-1.0, /*qn_count=*/0,
                                  /*consecutive_skips=*/3};
        expect(should_reset_bfgs_reference(skips, /*qn_optimization=*/false), true,
               "reset_dispatch: consecutive_skips>=3 -> true (dead in the active Python path, but the "
               "function itself must still implement it correctly)");
    }

    // --- should_reset_bfgs_reference_point: the DIFFERENT 3-clause
    //     top-of-loop condition, helper_PFCI.py:11227 -- same first three
    //     clauses as should_reset_bfgs_reference, but explicitly WITHOUT the
    //     consecutive_skips>=3 disjunct. The two functions must actually
    //     differ on a case only distinguished by that clause. ---
    {
        BfgsReferenceState skips_only{/*density_norm_change=*/0.0, /*predicted_energy=*/-1.0, /*qn_count=*/0,
                                       /*consecutive_skips=*/3};
        expect(should_reset_bfgs_reference(skips_only, /*qn_optimization=*/false), true,
               "reference_point vs dispatch: consecutive_skips>=3 alone -> should_reset_bfgs_reference true");
        expect(should_reset_bfgs_reference_point(skips_only, /*qn_optimization=*/false), false,
               "reference_point vs dispatch: consecutive_skips>=3 alone -> should_reset_bfgs_reference_point "
               "false (no consecutive_skips disjunct) -- the two functions must genuinely differ here");

        BfgsReferenceState density{/*density_norm_change=*/0.03, /*predicted_energy=*/-1.0, /*qn_count=*/0,
                                    /*consecutive_skips=*/0};
        expect(should_reset_bfgs_reference_point(density, /*qn_optimization=*/true), true,
               "reference_point: density_norm_change>0.025 and qn_optimization=true -> true");
        expect(should_reset_bfgs_reference_point(density, /*qn_optimization=*/false), false,
               "reference_point: density_norm_change>0.025 but qn_optimization=false -> false");

        BfgsReferenceState predicted{/*density_norm_change=*/0.0, /*predicted_energy=*/0.5, /*qn_count=*/0,
                                      /*consecutive_skips=*/0};
        expect(should_reset_bfgs_reference_point(predicted, /*qn_optimization=*/false), true,
               "reference_point: predicted_energy>0 -> true regardless of qn_optimization");

        BfgsReferenceState qn_count_one{/*density_norm_change=*/0.0, /*predicted_energy=*/-1.0, /*qn_count=*/1,
                                         /*consecutive_skips=*/0};
        expect(should_reset_bfgs_reference_point(qn_count_one, /*qn_optimization=*/false), true,
               "reference_point: qn_count==1 -> true regardless of qn_optimization");

        BfgsReferenceState all_false{/*density_norm_change=*/0.0, /*predicted_energy=*/-1.0, /*qn_count=*/0,
                                      /*consecutive_skips=*/0};
        expect(should_reset_bfgs_reference_point(all_false, /*qn_optimization=*/true), false,
               "reference_point: all three clauses false -> false");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
