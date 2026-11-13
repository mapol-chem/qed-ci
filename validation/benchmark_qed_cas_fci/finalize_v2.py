"""Merge whole converged photon-basis groups, audit selections, render v2 outputs."""
import csv
import hashlib
import json
import numpy as np
from config import RESULTS, FIGURES, SYSTEMS, LAMBDA_MAGNITUDES, lambda_key
from plot_combined import combined_rows, selected, stats, figure, read
from run_coherent_v2 import write_csv

METRICS=[('1-RDM Frobenius','cavity_free_frobenius','ratio','Frobenius norm','rdm',False),
 ('Natural occupations','cavity_free_natural_occupation_error','natural_occupation_ratio','max occupation error','natural_occupations',True)]


def key(r):return r['system'],r['geometry'],lambda_key(r['lambda_value'])


def main():
    original=combined_rows()
    old=read(RESULTS/'full_metrics.csv')
    detuned=read(RESULTS/'offres_metrics.csv')
    new=read(RESULTS/'coherent_v2_metrics.csv')
    groups=read(RESULTS/'coherent_v2_groups.csv')
    convergence=read(RESULTS/'coherent_v2_convergence.csv')
    for r in groups:
        if r['reason']=='photon_truncation_or_ci':
            last=[c for c in convergence if key(c)==key(r)][-1]
            r['reason']=('ci_not_converged' if last['ci_converged']!='True' else
                'assignment_gate_no_valid_previous' if last['max_target_energy_change']=='' else 'photon_truncation')
    write_csv(RESULTS/'coherent_v2_groups.csv',groups)
    assert len(groups)==15, 'Production must finish all 15 requested groups'
    rerun_converged={key(r) for r in groups if r['converged']=='True'}
    old_converged={key(r) for r in old if r['status']=='converged'}
    # Only newly recovered groups need replacement; retain already validated
    # number-state groups (including their parity-only polaritons) intact.
    replacement=rerun_converged-old_converged
    # Require every active-space calculation in a group to pass as well.
    replacement={k for k in replacement if all(r['status']=='converged' for r in new if key(r)==k)}
    merged=[]
    for rows,condition in ((old,'resonant'),(detuned,'blue_detuned')):
        for r in rows:
            if key(r) in replacement:continue
            row=dict(r,condition=condition,photon_basis='bare_number_state',electronic_orbitals='canonical_cavity_free_rhf')
            if condition=='blue_detuned' and row['state']=='ground':row['state']='ground_offres'
            merged.append(row)
    for r in new:
        spec=SYSTEMS[r['system']]
        total=spec['valence_electrons']+2*spec['core_orbitals']
        r['reference_frozen_core_orbitals']=r.get('reference_frozen_core_orbitals') or r.get('frozen_core_orbitals')
        r['frozen_core_orbitals']=(total-int(r['active_electrons']))//2
    write_csv(RESULTS/'coherent_v2_metrics.csv',new)
    merged += [r for r in new if key(r) in replacement]
    for k in {key(r) for r in merged}:
        assert len({r['photon_basis'] for r in merged if key(r)==k})==1
    write_csv(RESULTS/'full_v2_metrics.csv',merged)
    summary=[];audits=[];outliers=[];presence={}
    for name,xkey,rkey,label,stem,natural in METRICS:
        ratios=selected(merged,xkey,rkey,natural=natural,ratio=True)
        identities=[(*key(r),r['active_space'],r['state']) for r in ratios]
        assert len(identities)==len(set(identities)), 'Duplicate ratio point'
        summary.extend(stats(ratios,name))
        presence[name]=sorted({r['system'] for r in ratios})
        # Isolate lambda matching: hold the old cached data and each selection fixed.
        eligible=[]
        for r in original:
            x,y=float(r[xkey]) if r.get(xkey) else 0,float(r[rkey]) if r.get(rkey) else 0
            if not (np.isfinite(x) and np.isfinite(y) and x>0 and y>0):continue
            if natural and x<1e-8:continue
            if any(np.isclose(float(r['lambda_value']),l,rtol=0,atol=1e-8) for l in LAMBDA_MAGNITUDES):eligible.append(r)
        selections={'historical resonant-only figure and summary':[r for r in eligible if r['condition']=='resonant'],
            'historical parity (all states)':eligible,
            'historical ratio/summary (duplicate grounds)': [r for r in eligible if r['state'] in {'ground','ground_offres','bright_dressed'}],
            'variant A parity; A/B ratio and summary':[r for r in eligible if r['state'] in {'ground','bright_dressed'}],
            'variant B parity':[r for r in eligible if r['state'] in {'ground','bright_dressed','LP','UP'}]}
        for selection,rows in selections.items():
            for lam in LAMBDA_MAGNITUDES:
                tolerant=sum(np.isclose(float(r['lambda_value']),lam,rtol=0,atol=1e-8) for r in rows)
                exact=sum(float(r['lambda_value'])==lam for r in rows)
                audits.append(dict(metric=name,selection=selection,lambda_value=lam,exact_n=exact,tolerant_n=int(tolerant),rows_recovered=int(tolerant-exact)))
        for variant in ('A','B'):
            parity,rr=figure(merged,xkey,rkey,label,f'full_v2_{stem}_variant_{variant}',natural,polaritons=variant=='B')
            assert len(rr)==len(ratios)
            assert all(r['state'] in {'ground','bright_dressed'} for r in rr)
            if variant=='A':assert len(parity)==len(rr)
        for r in ratios:
            if not .8<=r['ratio_num']<=1.25:
                outliers.append(dict(metric=name,**{k:r[k] for k in ('system','geometry','active_space','state','lambda_value','photon_basis')},
                    cavity_free_error=r['x_num'],qed_error=r['y_num'],ratio=r['ratio_num']))
    write_csv(RESULTS/'full_v2_summary.csv',summary)
    write_csv(RESULTS/'lambda_recovery_v2.csv',audits)
    write_csv(RESULTS/'full_v2_outliers.csv',outliers)
    lines=['# QED-CAS/QED-FCI v2 summary','',
        'Both variants use resonant ground states and blue-detuned bright-connected states in the ratio panel and all statistics. Blue-detuned ground states are excluded. Variant A has the same points in parity; Variant B adds light-gray resonant LP/UP in parity only.',
        '', '| Metric | λ | n | Median | Range |','|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['metric']} | {r['lambda_value']:.2f} | {r['n']} | {r['median_ratio']:.8g} | {r['min_ratio']:.8g}–{r['max_ratio']:.8g} |")
    lines+=['','Natural-occupation statistics retain the existing cavity-free error floor of 1e-8.','', '## Molecules represented']
    for metric,systems in presence.items():lines.append(f"\n{metric}: {len(systems)} molecules — {', '.join(systems)}.")
    lines+=['','## Points outside [0.8, 1.25]','','| Metric | System | Geometry | Active space | State | λ | Cavity-free error | QED error | Ratio |','|---|---|---|---|---|---:|---:|---:|---:|']
    for r in outliers:lines.append(f"| {r['metric']} | {r['system']} | {r['geometry']} | {r['active_space']} | {r['state']} | {float(r['lambda_value']):.2f} | {r['cavity_free_error']:.9g} | {r['qed_error']:.9g} | {r['ratio']:.9g} |")
    lines+=['','## Rows recovered by tolerant λ matching','','Counts below hold cached data and state selection fixed; they do not include coherent rerun additions or the removal of duplicate grounds. Each entry is the gain at λ=0.01 / 0.05 / 0.10.']
    for metric in presence:
        for selection in selections:
            aa=[r for r in audits if r['metric']==metric and r['selection']==selection]
            lines.append(f"\n- {metric}, {selection}: "+' / '.join(str(r['rows_recovered']) for r in aa)+f" (total {sum(r['rows_recovered'] for r in aa)}).")
    lines+=['','## Coherent rerun outcome','','| System | Geometry | λ | Photon states | Outcome |','|---|---|---:|---:|---|']
    for r in groups:lines.append(f"| {r['system']} | {r['geometry']} | {float(r['lambda_value']):.2f} | {r['photon_states']} | {r['reason']} |")
    lines+=['',f'{len(replacement)} newly converged coherent groups merged as whole groups ({len(rerun_converged)}/15 rerun groups converged). Already-converged number-state groups are retained intact, including parity-only polaritons; successful repeat calculations remain in coherent_v2_metrics.csv. Recovered groups contribute coherent ground rows only and replace all old rows in the group. Failed groups retain their original records and statuses. No detuned calculations were added.',
        '', '## Remaining omissions']
    present=set(presence['1-RDM Frobenius'])
    for system in SYSTEMS:
        if system not in present:
            statuses=sorted({r['status'] for r in merged if r['system']==system and float(r.get('lambda_value') or 0)>0})
            lines.append(f"\n- {system}: no eligible ratio points; recorded statuses: {', '.join(statuses)}.")
    remaining={key(r) for r in merged if r['condition']=='resonant' and r['state']=='ground' and r['status']!='converged'}
    lines.append('\nRemaining omitted resonant-ground groups (including partially represented systems): '+ '; '.join(f'{s}/{g}/λ={l:.2f}' for s,g,l in sorted(remaining))+'.')
    lines.append('\nThe older joint ground/LP/UP convergence CSV also flags He2 and CH4, but the supplied metrics explicitly accept their ground rows and mark their LP/UP unavailable. Those accepted ground rows are preserved. Their cached 6→10 ground-energy changes are at most 6.58e-14 Eh (He2) and 1.06e-12 Eh (CH4). They are represented in the ratio statistics, not omitted; no new ground-only acceptance rule was applied to the coherent reruns.')
    lines.append('\nHF blue-detuned bright-connected rows remain excluded by the original failed λ=0 state-identity check. Systems without a safe blue-detuned window gain no excited-state ratio points. CH4 has no screened detuned frequency because its spectrum is too short. Resonant LP/UP are never counted as ratios.')
    (RESULTS/'full_v2_summary.md').write_text('\n'.join(lines)+'\n')
    # Verify every pre-existing result/figure is byte-for-byte unchanged.
    hashes=json.loads((RESULTS/'original_checksums_v2.json').read_text())
    root=RESULTS.parent
    assert all(hashlib.sha256((root/path).read_bytes()).hexdigest()==digest for path,digest in hashes.items())
    from report_v2 import main as report
    report()
    print('Wrote v2 metrics, four figure pairs, summary, audit, and outliers; original artifacts unchanged.')

if __name__=='__main__':main()
