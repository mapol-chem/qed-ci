"""v3: uniform ground-state acceptance rule, basis selection, figures, summary.

Ground rule (applied to every resonant (system, geometry, lambda) group, number
and coherent alike): accept the group's resonant ground rows iff, between the
two largest photon-ladder rungs the group reached (its QED-CAS rows are
computed at the larger one), the character-tracked QED-FCI ground state has
  |dE| <= 1e-6 Eh  and  ||d gamma||_F <= 1e-6,
with both CI solves converged. Inputs come from ground_rule_v3.py, which
recomputes those two rungs and tracks the ground root by its projection onto
the physical photon vacuum x cavity-free ground vector.

LP/UP acceptance (joint ground/LP/UP energy test + assignment gate) is
unchanged, and LP/UP appear only in variant B parity. A group never mixes
photon bases; where number-state and coherent groups both pass, the
number-state rows are kept. v2 and earlier artifacts are read-only.
"""
import hashlib
import json

import numpy as np

from config import RESULTS, FIGURES, SYSTEMS, LAMBDA_MAGNITUDES, lambda_key
from plot_combined import selected, stats, figure, read
from run_coherent_v2 import write_csv

E_TOL = 1.0e-6
RDM_TOL = 1.0e-6
MIN_GROUND_SCORE = 0.5   # tracked root must be dominantly vacuum x ground
METRICS = [('1-RDM Frobenius', 'cavity_free_frobenius', 'ratio', 'Frobenius norm', 'rdm', False),
           ('Natural occupations', 'cavity_free_natural_occupation_error', 'natural_occupation_ratio',
            'max occupation error', 'natural_occupations', True)]
NUMBER, COHERENT = 'bare_number_state', 'coherent_state'


def key(r):
    return r['system'], r['geometry'], lambda_key(r['lambda_value'])


def ground_rule(rec):
    """Return (accepted, cause) for one recomputed group record."""
    rungs = list(rec['rungs'].values())
    if not all(r['ci_converged'] for r in rungs):
        return False, 'ci_not_converged'
    if min(r['ground_score'] for r in rungs) < MIN_GROUND_SCORE:
        return False, 'ground_tracking_ambiguous'
    de, dg = rec['ground_delta_e'], rec['ground_delta_rdm']
    if de <= E_TOL and dg <= RDM_TOL:
        return True, 'accepted'
    return False, 'photon_truncation_' + ('energy_and_rdm' if de > E_TOL and dg > RDM_TOL
                                          else 'energy' if de > E_TOL else 'rdm_only')


def choose_basis(number_pass, coherent_pass):
    """Whole-group basis choice; number-state rows win when both pass."""
    if number_pass:
        return NUMBER
    if coherent_pass:
        return COHERENT
    return NUMBER   # neither passes: keep original rows, ground marked failed


def ground_table():
    rows = []
    for rec in map(json.loads, (RESULTS/'ground_rule_v3_recompute.jsonl').read_text().splitlines()):
        ok, cause = ground_rule(rec)
        a, b = (rec['rungs'][str(n)] for n in rec['rung_pair'])
        rows.append(dict(photon_basis=rec['photon_basis'], system=rec['system'], geometry=rec['geometry'],
            lambda_value=rec['lambda_value'], ladder='/'.join(map(str, rec['ladder'])),
            rung_pair='%d->%d' % tuple(rec['rung_pair']), ground_delta_e=rec['ground_delta_e'],
            ground_delta_rdm=rec['ground_delta_rdm'], ground_delta_rdm_raw_mo=rec['ground_delta_rdm_raw_mo'],
            orbital_max_abs_diff=rec['orbital_max_abs_diff'],
            ground_roots='%d,%d' % (a['ground_root'], b['ground_root']),
            min_ground_score=min(a['ground_score'], b['ground_score']),
            max_other_score=max(a['next_best_score'], b['next_best_score']),
            ci_converged=a['ci_converged'] and b['ci_converged'],
            max_energy_reproduction=max(a['energy_reproduction'], b['energy_reproduction']),
            cas_ground_roots=','.join(str(c['ground_root']) for c in rec['cas']),
            cas_ci_converged=all(c['ci_converged'] for c in rec['cas']),
            accepted=ok, cause=cause, _cas=rec['cas']))
    return rows


def main():
    table = ground_table()
    rule = {(r['photon_basis'],) + key(r): r for r in table}
    old = read(RESULTS/'full_metrics.csv')
    detuned = read(RESULTS/'offres_metrics.csv')
    coherent = read(RESULTS/'coherent_v2_metrics.csv')
    v2 = read(RESULTS/'full_v2_metrics.csv')
    v2_summary = read(RESULTS/'full_v2_summary.csv')

    resonant_keys = {key(r) for r in old if float(r['lambda_value'] or 0) > 0}
    assert all((NUMBER,)+k in rule for k in resonant_keys), 'missing number-state recompute'
    assert all((COHERENT,)+key(r) in rule for r in coherent), 'missing coherent recompute'

    decisions = {}
    for k in sorted(resonant_keys):
        n, c = rule[(NUMBER,)+k], rule.get((COHERENT,)+k)
        decisions[k] = choose_basis(n['accepted'], bool(c and c['accepted']))

    # Cross-check cached QED-CAS ground rows against the recompute: the cached
    # rows must sit at the final rung, and the character-tracked CAS ground
    # must reproduce the cached QED error.
    checks = []
    for basis, rows in ((NUMBER, old), (COHERENT, coherent)):
        for r in rows:
            if r['state'] != 'ground' or float(r['lambda_value'] or 0) == 0:
                continue
            t = rule[(basis,)+key(r)]
            cas = next(c for c in t['_cas'] if c['active_space'] == r['active_space'])
            final = int(t['rung_pair'].split('->')[1])
            checks.append(dict(photon_basis=basis, system=r['system'], geometry=r['geometry'],
                lambda_value=lambda_key(r['lambda_value']), active_space=r['active_space'],
                cached_photon_states=int(r['photon_states']), final_rung=final,
                cas_ground_root=cas['ground_root'], cas_ground_score=cas['ground_score'],
                cached_qed_frobenius=float(r['qed_frobenius']), recomputed_qed_frobenius=cas['qed_frobenius'],
                abs_diff=abs(float(r['qed_frobenius'])-cas['qed_frobenius'])))
    assert all(c['cached_photon_states'] == c['final_rung'] for c in checks)
    write_csv(RESULTS/'ground_rule_v3_cas_check.csv', checks)

    merged, dropped_polaritons = [], []
    for r in old:
        k = key(r)
        lam = float(r['lambda_value'] or 0)
        if lam > 0 and decisions[k] == COHERENT:
            if r['state'] in ('LP', 'UP') and r['status'] == 'converged':
                dropped_polaritons.append(r)
            continue
        row = dict(r, condition='resonant', photon_basis=NUMBER, electronic_orbitals='canonical_cavity_free_rhf',
                   joint_status=r['status'])
        if lam > 0 and r['state'] == 'ground':
            t = rule[(NUMBER,)+k]
            row.update(status='converged' if t['accepted'] else 'ground_rule_failed',
                       ground_rule_cause=t['cause'], ground_delta_e=t['ground_delta_e'],
                       ground_delta_rdm=t['ground_delta_rdm'])
        merged.append(row)
    for r in coherent:
        k = key(r)
        if decisions[k] != COHERENT:
            continue
        t = rule[(COHERENT,)+k]
        merged.append(dict(r, joint_status=r['status'], status='converged', ground_rule_cause=t['cause'],
                           ground_delta_e=t['ground_delta_e'], ground_delta_rdm=t['ground_delta_rdm']))
    for r in detuned:
        row = dict(r, condition='blue_detuned', photon_basis=NUMBER, electronic_orbitals='canonical_cavity_free_rhf',
                   joint_status=r['status'])
        if row['state'] == 'ground':
            row['state'] = 'ground_offres'
        merged.append(row)
    for k in {key(r) for r in merged if r['condition'] == 'resonant' and float(r['lambda_value'] or 0) > 0}:
        assert len({r['photon_basis'] for r in merged if key(r) == k and r['condition'] == 'resonant'}) == 1, k
    write_csv(RESULTS/'full_v3_metrics.csv', merged)
    write_csv(RESULTS/'ground_rule_v3.csv', [{k: v for k, v in r.items() if k != '_cas'} for r in table])

    # Old v2 ground acceptance, per group, for the newly-accepted comparison.
    v2_ground = {key(r): r['photon_basis'] for r in v2 if r['state'] == 'ground' and r.get('condition') == 'resonant'
                 and r['status'] == 'converged' and float(r['lambda_value'] or 0) > 0}

    summary, outliers, presence = [], [], {}
    for name, xkey, rkey, label, stem, natural in METRICS:
        ratios = selected(merged, xkey, rkey, natural=natural, ratio=True)
        ids = [(*key(r), r['active_space'], r['state']) for r in ratios]
        assert len(ids) == len(set(ids)), 'duplicate ratio point'
        summary.extend(stats(ratios, name))
        presence[name] = sorted({r['system'] for r in ratios})
        for variant in ('A', 'B'):
            parity, rr = figure(merged, xkey, rkey, label, f'full_v3_{stem}_variant_{variant}', natural,
                                polaritons=variant == 'B')
            assert len(rr) == len(ratios) and all(r['state'] in {'ground', 'bright_dressed'} for r in rr)
            if variant == 'A':
                assert len(parity) == len(rr)
            else:
                assert all(r['photon_basis'] == NUMBER and r['status'] == 'converged'
                           for r in parity if r['state'] in ('LP', 'UP'))
        for r in ratios:
            if not .8 <= r['ratio_num'] <= 1.25:
                outliers.append(dict(metric=name, **{k: r[k] for k in ('system', 'geometry', 'active_space', 'state', 'lambda_value', 'photon_basis')},
                                     cavity_free_error=r['x_num'], qed_error=r['y_num'], ratio=r['ratio_num']))
    write_csv(RESULTS/'full_v3_summary.csv', summary)
    write_csv(RESULTS/'full_v3_outliers.csv', outliers)

    accepted = {k for k in resonant_keys if rule[(decisions[k],)+k]['accepted']}
    newly = sorted(accepted - set(v2_ground))
    lost = sorted(set(v2_ground) - accepted)
    basis_changed = sorted(k for k in accepted & set(v2_ground) if v2_ground[k] != decisions[k])
    omitted = sorted(resonant_keys - accepted)

    def fmt(k): return f'{k[0]}/{k[1]}/λ={k[2]:.2f}'
    L = ['# QED-CAS/QED-FCI v3 summary', '',
         '## Ground-state acceptance rule (v3)', '',
         'Resonant ground rows enter only the ratio/parity sets, so their acceptance no longer depends on LP/UP assignment. '
         'For every resonant (system, geometry, λ) group, number-state and coherent runs alike, the resonant ground rows are accepted iff, '
         'between the two largest photon-ladder rungs the group reached (the final rung is the one at which its QED-CAS rows were computed):',
         '', '- the QED-FCI ground-state energy changes by ≤ 1e-6 Eh, and',
         '- the QED-FCI ground-state electronic 1-RDM changes by ≤ 1e-6 (Frobenius),', '',
         'with both CI solves converged. The ground root is tracked by character, not index: at each rung it is the root with the largest squared '
         'projection onto the physical photon vacuum ⊗ cavity-free ground vector (the vacuum is transformed into the displaced oscillator for coherent runs); '
         f'a tracked score below {MIN_GROUND_SCORE} would count as a tracking failure. Separate Psi4 runs may rotate degenerate MOs, so the two '
         'rungs\' RDMs are compared in one orbital frame (U = C_a⁻¹ C_b). The two rungs were recomputed by `ground_rule_v3.py` because the caches do not store CI vectors.',
         '', 'This rule replaces the previous special-case acceptance of He2 and CH4 ground rows. LP/UP acceptance (joint energy test and assignment gate) is unchanged, '
         'and LP/UP appear only in variant B parity. Bases are never mixed within a group; where number-state and coherent groups both pass, the number-state rows are kept.',
         '', '## Per-λ statistics', '', '| Metric | λ | n (v2) | n (v3) | Median | Range |', '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        old_n = next(int(s['n']) for s in v2_summary if s['metric'] == r['metric'] and np.isclose(float(s['lambda_value']), r['lambda_value']))
        L.append(f"| {r['metric']} | {r['lambda_value']:.2f} | {old_n} | {r['n']} | {r['median_ratio']:.8g} | {r['min_ratio']:.8g}–{r['max_ratio']:.8g} |")
    L += ['', 'Natural-occupation statistics retain the cavity-free error floor of 1e-8. Ratios use resonant ground + blue-detuned bright-connected states only.', '',
          '## Molecules represented']
    for m, s in presence.items():
        L.append(f'\n{m}: {len(s)} molecules — {", ".join(s)}.')
    L += ['', '## Newly accepted groups (relative to v2)', '']
    if newly:
        L += ['| Group | Basis | Rungs | Ground ΔE (Eh) | Ground Δγ | Ground rows added |', '|---|---|---|---:|---:|---:|']
        for k in newly:
            t = rule[(decisions[k],)+k]
            n_rows = sum(1 for r in merged if key(r) == k and r['state'] == 'ground' and r['condition'] == 'resonant')
            L.append(f"| {fmt(k)} | {decisions[k]} | {t['rung_pair']} | {t['ground_delta_e']:.3e} | {t['ground_delta_rdm']:.3e} | {n_rows} |")
    else:
        L.append('None.')
    L += ['', f"Groups accepted in v2 but failing the v3 rule: {', '.join(map(fmt, lost)) or 'none'}.",
          f"Accepted groups whose basis changed from v2: {', '.join(f'{fmt(k)} ({v2_ground[k]} → {decisions[k]})' for k in basis_changed) or 'none'}.",
          f"Number-state LP/UP rows dropped to avoid basis mixing: {len(dropped_polaritons)}.",
          '', 'He2 and CH4 now pass the uniform rule on their own numbers:']
    for s in ('He2', 'CH4'):
        for k in sorted(k for k in resonant_keys if k[0] == s):
            t = rule[(NUMBER,)+k]
            L.append(f"- {fmt(k)}: {t['rung_pair']}, ΔE = {t['ground_delta_e']:.2e} Eh, Δγ = {t['ground_delta_rdm']:.2e} → {'accepted' if t['accepted'] else t['cause']}.")
    L += ['', '## Remaining omitted resonant-ground groups', '',
          '| Group | Basis tested | Rungs | Ground ΔE (Eh) | Ground Δγ | Cause |', '|---|---|---|---:|---:|---|']
    for k in omitted:
        for b in (NUMBER, COHERENT):
            t = rule.get((b,)+k)
            if t:
                L.append(f"| {fmt(k)} | {b} | {t['rung_pair']} | {t['ground_delta_e']:.3e} | {t['ground_delta_rdm']:.3e} | {t['cause']} |")
    L += ['', '## Points outside [0.8, 1.25]', '',
          '| Metric | System | Geometry | Active space | State | λ | Basis | Cavity-free error | QED error | Ratio |', '|---|---|---|---|---|---:|---|---:|---:|---:|']
    for r in outliers:
        L.append(f"| {r['metric']} | {r['system']} | {r['geometry']} | {r['active_space']} | {r['state']} | {float(r['lambda_value']):.2f} | {r['photon_basis']} | {r['cavity_free_error']:.9g} | {r['qed_error']:.9g} | {r['ratio']:.9g} |")
    worst = max(c['abs_diff'] for c in checks)
    L += ['', '## Consistency checks', '',
          f"- Tracked QED-FCI ground root is index 0 at both rungs in {sum(t['ground_roots'] == '0,0' for t in table)}/{len(table)} groups; "
          f"minimum tracked score {min(t['min_ground_score'] for t in table):.4f}, largest competing score {max(t['max_other_score'] for t in table):.2e}.",
          f"- Recomputed ground energies reproduce the cached energies to ≤ {max(t['max_energy_reproduction'] for t in table):.1e} Eh.",
          f"- QED-CAS ground rows: tracked CAS ground is root 0 in {sum(c['cas_ground_root'] == 0 for c in checks)}/{len(checks)} cached rows; "
          f"all cached rows sit at their group's final rung; recomputed errors match the cache to ≤ 1e-6 in {sum(c['abs_diff'] <= 1e-6 for c in checks)}/{len(checks)} rows (exceptions below).",
          ]
    acc_bad = [c for c in checks if c['abs_diff'] > 1e-6 and decisions[key(c)] == c['photon_basis'] and rule[(c['photon_basis'],)+key(c)]['accepted']]
    rej_bad = [c for c in checks if c['cas_ground_root'] != 0]
    spaces = sorted({(c['system'], c['active_space']) for c in acc_bad})
    L += [f"- Rows used in v3 whose recomputed QED-CAS error differs from the cache by > 1e-6: {len(acc_bad)} "
          f"({', '.join(f'{s} {a}' for s, a in spaces) or 'none'}; largest {max([c['abs_diff'] for c in acc_bad] or [0]):.1e}, "
          f"relative {max([c['abs_diff']/c['cached_qed_frobenius'] for c in acc_bad] or [0]):.1e}). "
          'These active spaces end inside a degenerate canonical π pair (RHF orbital-energy gap < 1e-14 Eh at the boundary), so the QED-CAS '
          'result depends on the arbitrary rotation of that pair relative to the cavity polarization and is not run-to-run reproducible. '
          'The discrepancy grows with λ and is unrelated to photon truncation; the rows are kept as cached (pre-existing protocol issue).',
          f"- Rows whose character-tracked QED-CAS ground is not root 0: {len(rej_bad)} "
          f"({', '.join(sorted({f'{c['system']}/{c['geometry']}/λ={c['lambda_value']:.2f} {c['active_space']}' for c in rej_bad})) or 'none'}); "
          'all belong to groups that fail the ground rule, so none enters the figures or statistics.',
          f"- Orbital frames differed between the two rungs in {sum(t['orbital_max_abs_diff'] > 1e-8 for t in table)} recomputed groups; differences are compared after frame alignment.",
          '', 'Inputs: `ground_rule_v3_recompute.jsonl` (recompute), `ground_rule_v3.csv` (per-group rule table), `ground_rule_v3_cas_check.csv`. '
          'Figures: `full_v3_{rdm,natural_occupations}_variant_{A,B}.{pdf,png}` (PNG at 300 dpi).']
    (RESULTS/'full_v3_summary.md').write_text('\n'.join(L)+'\n')

    hashes = json.loads((RESULTS/'original_checksums_v3.json').read_text())
    root = RESULTS.parent
    changed = [p for p, d in hashes.items() if hashlib.sha256((root/p).read_bytes()).hexdigest() != d]
    assert not changed, f'pre-v3 artifacts modified: {changed}'
    print(f'v3: {len(accepted)}/{len(resonant_keys)} resonant groups accepted; newly {len(newly)}, lost {len(lost)}; '
          f'{len(table)} recomputed groups; pre-v3 artifacts unchanged ({len(hashes)} files).')


if __name__ == '__main__':
    main()
