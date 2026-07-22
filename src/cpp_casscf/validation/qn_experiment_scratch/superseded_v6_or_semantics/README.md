# Superseded v6 runs -- OR-semantics QED_RESET_ON_BAD_STEP

These three MgH+ runs (seeds 1-3) were made with the ORIGINAL v6 toggle, which
OR'd `_qn_bad_step` into a reset condition that already contained
`_energy_up = self.predicted_energy > 0`.

They are byte-for-byte equivalent to their v0 counterparts because that term
was never decisive -- across all MgH+ v0/v6 runs, 372 of 523 QN steps raised
the actual energy and in ZERO of them was `predicted_energy <= 0`, so the
default trigger had already fired every time. Confirmed independently with
QED_TRACE_RESET on lih/h2o/beh2 x 3 seeds: `bad=1` in 8 of 400 reset checks,
another trigger also true in all 8, reset fired because of `bad` in 0.

Kept as the evidence for that finding. NOT comparable to the rewritten v6
(which replaces the trigger instead of OR-ing into it), so they are out of
mgh_det_logs/ to keep tabulate_mgh_det.py from mixing the two semantics.
