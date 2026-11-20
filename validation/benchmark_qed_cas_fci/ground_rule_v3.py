"""v3 ground-state acceptance: recompute the two largest computed ladder rungs.

For every resonant (system, geometry, lambda) group in the number-state run
(full_cache.jsonl) and the coherent rerun (coherent_v2_cache.jsonl), the QED-FCI
reference is recomputed at the two largest photon-ladder rungs that the group
actually reached (the final rung is where its QED-CAS rows were computed). The
ground root is tracked by character, not index: the root with the largest
squared projection onto the physical photon vacuum times the cavity-free FCI
ground vector (transformed into the displaced oscillator for coherent runs).

The QED-CAS ground root at the final rung is tracked the same way and its error
against the tracked FCI ground is recomputed, to check the cached metric rows.

Cached data are read-only. Output: results/ground_rule_v3_recompute.jsonl
(resumable, one record per group).
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

import run_pilot as rp
from config import RESULTS, SYSTEMS, lambda_key

OUT = RESULTS / 'ground_rule_v3_recompute.jsonl'
SOURCES = (('bare_number_state', 'full_cache.jsonl', 'number'),
           ('coherent_state', 'coherent_v2_cache.jsonl', 'coherent'))


def vacuum_target(owner, bare_ground):
    """Physical |0> x |Phi_0> in the calculation's photon basis."""
    n = int(owner.N_p) + 1
    if getattr(owner, 'benchmark_photon_basis', 'number') == 'coherent':
        from coherent_v2 import displacement
        alpha = displacement(owner)
        vac = np.array([np.exp(-alpha**2/2)*(-alpha)**p/np.sqrt(float(math.factorial(p))) for p in range(n)])
    else:
        vac = np.zeros(n); vac[0] = 1.0
    return np.kron(vac, bare_ground)


def bare_vector(owner, root=0):
    """Photon-vacuum electronic block of a lambda=0 root (dense CAS may carry 2 states)."""
    v = rp.root_vector(owner, root)[:int(owner.num_det)]
    return v/np.linalg.norm(v)


def to_frame(gamma, C, C_ref):
    """Express an MO 1-RDM in the orbital frame of C_ref (same AO basis).

    Separate psi4 runs can return different rotations within degenerate MO
    sets, so MO-basis RDMs from different runs are compared in one frame:
    U = C_ref^-1 C (C_ref^T S = C_ref^-1 for square orthonormal C).
    """
    U = np.linalg.solve(C_ref, C)
    return U @ gamma @ U.T


def ground_track(owner, bare_ground):
    scores = np.abs(np.asarray(owner.CIvecs) @ vacuum_target(owner, bare_ground))**2
    order = np.argsort(scores)[::-1]
    return int(order[0]), scores


def groups():
    out = defaultdict(dict)
    for basis_label, fname, _ in SOURCES:
        for line in (RESULTS/fname).read_text().splitlines():
            r = json.loads(line)
            if r.get('state') != 'photon_convergence':
                continue
            k = (basis_label, r['system'], r['geometry'], lambda_key(np.linalg.norm(r['lambda_vector'])))
            out[k][int(r['photon_states'])] = r   # last record per rung wins
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args()
    done = set()
    if OUT.exists():
        done = {(r['photon_basis'], r['system'], r['geometry'], r['lambda_value'])
                for r in map(json.loads, OUT.read_text().splitlines())}
    todo = {k: v for k, v in groups().items() if k not in done and (not args.only or k[1] in args.only)}
    # Cheapest systems first; one bare reference per geometry.
    by_geom = defaultdict(list)
    for k in todo: by_geom[(k[1], k[2])].append(k)
    def size(g): return max(r['ci_dimension'] for k in by_geom[g] for r in todo[k].values())
    for system, geometry in sorted(by_geom, key=size):
        scale = 1.0 if geometry == 'equilibrium' else next(
            s['scale'] for name, s in rp.STRETCHED.items() if name == geometry)
        mol = rp.geom_variant(system, scale)
        basis, ne, no, fc, _ = rp.selected_feasibility(system)
        method = 'casci' if fc else 'fci'
        bare, _ = rp.run_calc(mol, basis, method, ne, no, 0, (0, 0, 0), 1)
        g0 = bare_vector(bare)
        spec = SYSTEMS[system]; ve, vo = spec['valence_electrons'], spec['valence_orbitals']
        spaces = list(dict.fromkeys((ve, o) for o in (vo, vo+2, vo+4) if o < no and ve <= ne))
        cas0 = {s: rp.run_calc(mol, basis, 'casci', s[0], s[1], 0, (0, 0, 0), 1)[0] for s in spaces}
        for key in sorted(by_geom[(system, geometry)]):
            basis_label, _, _, lam = key
            pb = 'coherent' if basis_label == 'coherent_state' else 'number'
            rungs = sorted(todo[key])
            pair = rungs[-2:]
            rec = dict(photon_basis=basis_label, system=system, geometry=geometry, lambda_value=lam,
                       method=method, ladder=rungs, rung_pair=pair, rungs={})
            fci_final = None
            frames = {}
            for ns in pair:
                cached = todo[key][ns]
                owner, t = rp.run_calc(mol, basis, method, ne, no, cached['omega'], cached['lambda_vector'], ns, photon_basis=pb)
                g, scores = ground_track(owner, g0)
                gamma = rp.rdm(owner, g)
                frames[ns] = np.asarray(owner.C).copy()
                e = np.asarray(owner.CIeigs)
                second = float(np.sort(scores)[-2]) if len(scores) > 1 else 0.0
                rec['rungs'][str(ns)] = dict(
                    ground_root=g, ground_score=float(scores[g]), next_best_score=second,
                    energy=float(e[g]), ci_converged=bool(owner.result.convergence['ci_final'].get('converged')),
                    cached_energy_same_root=float(cached['energies'][g]),
                    energy_reproduction=float(abs(e[g]-cached['energies'][g])),
                    rdm_reproduction=float(np.linalg.norm(gamma-np.asarray(cached['roots'][g]['rdm']))),
                    seconds=t, rdm=gamma.tolist())
                if ns == pair[-1]:
                    fci_final = (owner, g, gamma)
            a, b = (rec['rungs'][str(n)] for n in pair)
            rec['ground_delta_e'] = abs(b['energy']-a['energy'])
            Ca, Cb = frames[pair[0]], frames[pair[1]]
            rec['orbital_max_abs_diff'] = float(np.max(np.abs(Ca-Cb)))
            rec['ground_delta_rdm_raw_mo'] = float(np.linalg.norm(np.asarray(b['rdm'])-np.asarray(a['rdm'])))
            rec['ground_delta_rdm'] = float(np.linalg.norm(to_frame(np.asarray(b['rdm']), Cb, Ca)-np.asarray(a['rdm'])))
            rec['ground_root_index_zero'] = a['ground_root'] == 0 and b['ground_root'] == 0
            # QED-CAS ground roots at the final rung, tracked by character.
            owner, g, gamma_f = fci_final
            rec['cas'] = []
            for (ae, ao), c0 in cas0.items():
                q, _ = rp.run_calc(mol, basis, 'casci', ae, ao, todo[key][pair[-1]]['omega'],
                                   todo[key][pair[-1]]['lambda_vector'], pair[-1], photon_basis=pb)
                cg, cs = ground_track(q, bare_vector(c0))
                gq = to_frame(rp.rdm(q, cg), np.asarray(q.C), np.asarray(owner.C))
                qerr, qno = rp.rdm_metrics(gq, gamma_f)
                rec['cas'].append(dict(active_space=f'CAS({ae},{ao})', ground_root=cg, ground_score=float(cs[cg]),
                                       orbital_max_abs_diff_vs_fci=float(np.max(np.abs(np.asarray(q.C)-np.asarray(owner.C)))),
                                       qed_frobenius=qerr, qed_natural_occupation_error=qno,
                                       ci_converged=bool(q.result.convergence['ci_final'].get('converged'))))
            with OUT.open('a') as f:
                f.write(json.dumps(rec)+'\n')
            print(key, pair, 'dE=%.2e dG=%.2e roots=%s' % (rec['ground_delta_e'], rec['ground_delta_rdm'],
                  [rec['rungs'][str(n)]['ground_root'] for n in pair]), flush=True)


if __name__ == '__main__':
    main()
