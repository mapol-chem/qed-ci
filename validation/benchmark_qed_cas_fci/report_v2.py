"""Write a compact scientific report from completed v2 artifacts."""
import csv
import json
import numpy as np
from config import RESULTS, LAMBDA_MAGNITUDES, SYSTEMS, lambda_key
from plot_combined import read, selected


def main():
    # Additional validation with nonzero displacement; use completed caches.
    old_cache=[json.loads(x) for x in (RESULTS/'full_cache.jsonl').read_text().splitlines()]
    new_cache=[json.loads(x) for x in (RESULTS/'coherent_v2_cache.jsonl').read_text().splitlines()]
    def pick(rows):
        return max((r for r in rows if r['system']=='LiH' and r['geometry']=='equilibrium' and r['state']=='photon_convergence' and np.isclose(np.linalg.norm(r['lambda_vector']),.01)),key=lambda r:r['photon_states'])
    a,b=pick(old_cache),pick(new_cache)
    extra=[]
    for state,ar,br in [('ground',0,0),('LP',a['lp_root'],b['lp_root']),('UP',a['up_root'],b['up_root'])]:
        de=abs(a['energies'][ar]-b['energies'][br]); dr=float(np.linalg.norm(np.array(a['roots'][ar]['rdm'])-b['roots'][br]['rdm']))
        extra.append(dict(system='LiH',geometry='equilibrium',lambda_value=.01,state=state,displacement_alpha=b['displacement_alpha'],energy_abs_diff=de,rdm_frobenius=dr,passed=de<=1e-6 and dr<=1e-6))
    with (RESULTS/'coherent_v2_nonzero_displacement_check.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(extra[0]));writer.writeheader();writer.writerows(extra)
    assert all(r['passed'] for r in extra)
    summary=read(RESULTS/'full_v2_summary.csv')
    audit=read(RESULTS/'lambda_recovery_v2.csv')
    checks=read(RESULTS/'coherent_v2_validation.csv')
    diagnosis=read(RESULTS/'beh2_diagnosis_v2.csv')
    groups=read(RESULTS/'coherent_v2_groups.csv')
    old=read(RESULTS/'full_metrics.csv')
    merged=read(RESULTS/'full_v2_metrics.csv')
    selected_rows=selected(merged,'cavity_free_frobenius','ratio',ratio=True)
    present=sorted({r['system'] for r in selected_rows})
    lines=['# QED-CAS/QED-FCI 1-RDM follow-up report','',
        'All calculations and figures used the `qed-ci` conda environment. Original results and figures were verified unchanged by SHA-256. New data, logs, tables, and figures have v2 names.',
        '', '## λ matching and state selection','',
        'Selection uses `np.isclose` with absolute tolerance 1e-8; grouping keys round to four decimals. The counts below isolate this fix on the original cached data, before rerun replacement. Both metrics have the same gains:']
    for selection in dict.fromkeys(r['selection'] for r in audit):
        rows=[r for r in audit if r['metric']=='1-RDM Frobenius' and r['selection']==selection]
        lines.append(f"\n- {selection}: +"+' / +'.join(r['rows_recovered'] for r in rows)+f" at λ=0.01 / 0.05 / 0.10; total +{sum(int(r['rows_recovered']) for r in rows)}.")
    lines+=['','Ratios and all summary statistics include resonant ground + blue-detuned bright-connected states only. Variant A uses exactly this set in parity. Variant B adds light-gray resonant LP/UP in parity only, behind colored points, labeled “resonant polaritons (not in ratio)”.',
        '', '## BeH2 diagnosis','',
        'The original BeH2 failure is an assignment gate; its ground energy is already photon-converged at 10 states. Convergence of the actual LP/UP cannot be established from the ambiguously assigned pair. At every ladder size and coupling, one of the two selected roots has projection score far below 0.1. `run_pilot.py` resets the previous-energy comparison to None, so convergence cannot pass even when energies stabilize. The calculation requests eight QED roots; the selected pair does not contain two bright/photon-character states. The rerun retains that protocol rather than increasing the root window or weakening the gate. The λ=0.01 maximum selected-energy change below includes an unstable near-zero-character candidate; it must not be interpreted as a tracked polariton truncation error.',
        '', '| λ | 6→10 ground ΔE (Eh) | 6→10 maximum selected ΔE (Eh) | Weak assignment score at 10 |','|---:|---:|---:|---:|']
    for r in diagnosis:
        if r['photon_states']=='10':lines.append(f"| {float(r['lambda_value']):.2f} | {float(r['ground_change']):.3e} | {float(r['max_selected_energy_change']):.3e} | {float(r['lp_score']):.3e} |")
    lines+=['', '## Coherent basis implementation and validation','',
        'The checked API option is `CQEDConfig(photon_basis="coherent")`. Its validation currently couples this to `reference_orbitals="qed_rhf"`; the legacy flags also forcibly pair coherent photons with QED-RHF orbitals. The orbital-guess path semicanonicalizes with the QED Fock matrix, so it cannot guarantee the requested fixed canonical orbitals.',
        '', '`coherent_v2.py` therefore supplies a canonical RHF reference through a scoped reference-builder adapter, with the package’s coherent constants: d_exp=2 Tr(D d), d_c=d_exp²/2, and d_PF=-d_exp d. The actual electronic coefficients remain canonical cavity-free RHF (asserted against the bare run at 1e-10). No package source was modified and no orbital optimization was performed. Metadata explicitly records the effective canonical orbitals. The informational mean-field energy fields retain RHF values; CI energies use the displaced Hamiltonian constants.',
        '', 'Physical number-state character targets are transformed into the displaced oscillator before LP/UP assignment: a=b+α, α=(d_exp+λ·μ_nuc)/sqrt(2ω). Photon-number expectations are deliberately blank/null in coherent metrics/cache; displaced-sector occupation is not reported as physical photon number.',
        '', 'H2 equilibrium, λ=0.10, same cached resonant ω, 10 photon states:',
        '', '| State | Number/coherent absolute ΔE (Eh) | Electronic 1-RDM Frobenius difference |','|---|---:|---:|']
    for r in checks:
        if r['check']=='basis_agreement':lines.append(f"| {r['state']} | {float(r['energy_abs_diff']):.4e} | {float(r['rdm_frobenius']):.4e} |")
    lines.append('\nAll comparisons pass 1e-6; 6→10 checks also pass for both energies and RDMs. H2 orbital coefficient differences are exactly zero. Production starts only after this gate passes. An additional nonzero-displacement LiH comparison against the original number cache is in `coherent_v2_nonzero_displacement_check.csv`.')
    lines+=['', '## Production results','',
        'The rerun covers all 15 BeH2/LiH/Li2 geometry/coupling groups, with the original geometries, bases, core treatment, active-space rule, resonant FCI bright-transition frequencies, λ ladder, 2/4/6/10 photon states, and 1e-6 Eh criterion. The original 24-root cavity-free fallback is retained when the first eight roots contain no bright transition; QED calculations still request eight roots.',
        '', '| System | Geometry | λ | States | Result |','|---|---|---:|---:|---|']
    for r in groups:lines.append(f"| {r['system']} | {r['geometry']} | {float(r['lambda_value']):.2f} | {r['photon_states']} | {r['reason']} |")
    lines.append('\nAn assignment_gate_no_valid_previous result means the final assignment is acceptable but an ambiguous preceding rung reset the comparison; it is not evidence of a measured energy difference above tolerance. Failed assignment gates are kept distinct from actual photon-truncation failures.')
    old_ok={(r['system'],r['geometry'],lambda_key(r['lambda_value'])) for r in old if r['status']=='converged'}
    new_ok={(r['system'],r['geometry'],lambda_key(r['lambda_value'])) for r in groups if r['converged']=='True'}
    lines.append(f'\n{len(new_ok)}/15 groups converge in the coherent basis; {len(new_ok-old_ok)} are newly converged relative to the original number-state run. Newly recovered groups replace the entire failed old group and add ground rows only. Already-converged number-state groups remain intact, including their parity-only polaritons; their successful coherent repeats are retained separately in coherent_v2_metrics.csv. Failed coherent groups do not replace original number-state rows. No mixed-basis group is permitted. No convergence threshold was relaxed.')
    lines+=['', '## Updated caption statistics','',f"The ratio data represent {len(present)} molecules: {', '.join(present)}. Filled symbols denote equilibrium geometries and hollow symbols stretched geometries. Panel (b) excludes blue-detuned grounds and all LP/UP; the natural-occupation metric retains its 1e-8 baseline floor.",'','| Metric | λ | n | Median ratio | Range |','|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['metric']} | {float(r['lambda_value']):.2f} | {r['n']} | {float(r['median_ratio']):.6g} | {float(r['min_ratio']):.6g}–{float(r['max_ratio']):.6g} |")
    lines+=['','Complete outlier identities, both errors, remaining omitted systems/groups, and all row-count audits are in `full_v2_summary.md` and the accompanying CSV tables. Figures are `full_v2_{rdm,natural_occupations}_variant_{A,B}.{pdf,png}` (PNG at 300 dpi).']
    # Include the omission explanation directly, so this report stands alone.
    details=(RESULTS/'full_v2_summary.md').read_text().split('## Remaining omissions')[-1]
    lines+=['','## Remaining omissions',details]
    (RESULTS/'full_v2_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
