// Tests the print-level machinery: that Silent really is silent, that levels
// gate correctly, that disabled records do not evaluate their arguments (the
// reason CASSCF_LOG is a macro), and that spin deviation is reported by
// exception the way Normal level requires.
#include "casscf/logging.hpp"

#include <cstdio>
#include <sstream>

using namespace casscf;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) {
        std::printf("PASS: %s\n", what);
    } else {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

int side_effect_count = 0;
int counted() { return ++side_effect_count; }

} // namespace

int main() {
    // --- level gating ---
    std::ostringstream out;
    Logger lg{PrintLevel::Normal, &out};
    CASSCF_LOG(lg, PrintLevel::Normal, "[macro] iter=1");
    CASSCF_LOG(lg, PrintLevel::Debug, "[orb] should not appear");
    CASSCF_LOG(lg, PrintLevel::Trace, "[ci] should not appear");
    check(out.str() == "[macro] iter=1\n", "Normal emits Normal, suppresses Debug/Trace");

    out.str("");
    lg.level = PrintLevel::Trace;
    CASSCF_LOG(lg, PrintLevel::Normal, "a");
    CASSCF_LOG(lg, PrintLevel::Debug, "b");
    CASSCF_LOG(lg, PrintLevel::Trace, "c");
    check(out.str() == "a\nb\nc\n", "Trace emits every level");

    // --- Silent is the default, and is genuinely silent ---
    out.str("");
    Logger def;
    check(def.level == PrintLevel::Silent, "default level is Silent");
    check(!def.enabled(PrintLevel::Normal), "Silent disables Normal");
    Logger silent{PrintLevel::Silent, &out};
    CASSCF_LOG(silent, PrintLevel::Normal, "nope");
    check(out.str().empty(), "Silent with a live stream still emits nothing");

    // A null stream discards regardless of level -- this is what keeps the
    // ctest binaries and validate_against_python output clean.
    Logger nullsink{PrintLevel::Trace, nullptr};
    check(!nullsink.enabled(PrintLevel::Normal), "null stream disables all output");
    CASSCF_LOG(nullsink, PrintLevel::Normal, "nope"); // must not crash

    // --- disabled records must not evaluate their arguments ---
    // This is the whole reason CASSCF_LOG is a macro rather than a function
    // returning a null sink: Debug/Trace records format per-root data inside
    // hot loops, and formatting-then-discarding would cost more than the work
    // being described.
    side_effect_count = 0;
    Logger quiet{PrintLevel::Normal, &out};
    out.str("");
    CASSCF_LOG(quiet, PrintLevel::Debug, "value=" << counted());
    check(side_effect_count == 0, "disabled record does not evaluate its arguments");
    CASSCF_LOG(quiet, PrintLevel::Normal, "value=" << counted());
    check(side_effect_count == 1, "enabled record does evaluate its arguments");

    // --- spin reporting by exception ---
    RootSpin clean;
    clean.s2 = 1.3e-17; // the observed noise floor in the reference MgH+ logs
    clean.target_s2 = 0.0;
    clean.deviation = 1.3e-17;
    check(!spin_deviates(clean), "a converged singlet root is not flagged");

    RootSpin contaminated;
    contaminated.s2 = 0.0142;
    contaminated.target_s2 = 0.0;
    contaminated.deviation = 0.0142;
    check(spin_deviates(contaminated), "a visibly contaminated root is flagged");

    // A root can sit under the deviation threshold and still be a problem if
    // first_order_spin_projection exhausted its 20-iteration cap -- that is the
    // silent failure mode in ci_solver.c:1355, so it must flag independently.
    RootSpin capped;
    capped.deviation = 1e-9;
    capped.projection_iters = 20;
    capped.projection_failed = true;
    check(spin_deviates(capped), "projection failure flags even when deviation is small");

    // Threshold is a calibration knob, not a constant of nature.
    RootSpin marginal;
    marginal.deviation = 5e-4;
    check(spin_deviates(marginal), "5e-4 deviates at the default 1e-4 threshold");
    check(!spin_deviates(marginal, 1e-3), "the same root passes at a looser threshold");

    if (failures == 0) {
        std::printf("\nAll logging checks passed.\n");
        return 0;
    }
    std::printf("\n%d check(s) FAILED.\n", failures);
    return 1;
}
